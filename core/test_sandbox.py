"""
F5 sandbox hardening.

Unit tests check the docker command and inner script. Integration tests run
real containers and are opt-in (they need Docker and the elab-sandbox image):

    SANDBOX_INTEGRATION=1 DOCKER_SANDBOX_DIR=/tmp/sbx HOST_SANDBOX_DIR=/tmp/sbx python manage.py test core.test_sandbox
"""
import os
import subprocess
import unittest
from unittest.mock import patch

from django.test import SimpleTestCase

from . import sandbox


class SandboxCommandTests(SimpleTestCase):
    def _cmd(self, language="python", code="print(1)", memory_limit_kb=128000):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="1", stderr="")

        import tempfile

        tmp = tempfile.mkdtemp()
        with patch("core.sandbox.subprocess.run", side_effect=fake_run), patch("core.sandbox.SANDBOX_DIR", tmp):
            sandbox.run_code(language, code, memory_limit_kb=memory_limit_kb)
        return captured["cmd"]

    def test_hardening_flags(self):
        cmd = self._cmd()
        joined = " ".join(cmd)
        for flag in ("--network none", "--cap-drop ALL", "--read-only", f"--user {sandbox.SANDBOX_USER}",
                     "no-new-privileges:true", "--pids-limit 64", "fsize=", "nofile="):
            self.assertIn(flag, joined)
        box_mount = cmd[cmd.index("-v") + 1]
        self.assertTrue(box_mount.endswith(":/box:ro"), box_mount)

    def test_native_languages_get_room_for_the_compiler(self):
        self.assertIn(f"{sandbox.COMPILER_MEMORY_MB}m", self._cmd("cpp", "int main(){}"))
        self.assertIn("125m", self._cmd("python"))

    def test_inner_script_caps_output_and_fails_closed(self):
        script = sandbox._build_inner_script(sandbox.LANGUAGES["c"], "main.c", "", 2, "int main(){}", "",
                                             program_memory_kb=144000)
        self.assertIn(f"head -c {sandbox.STDOUT_LIMIT_BYTES}", script)
        self.assertIn(f"head -c {sandbox.STDERR_LIMIT_BYTES} /tmp/stderr.txt", script)
        self.assertIn('exit "${rc:-1}"', script)
        self.assertIn("ulimit -v 144000", script)
        self.assertLess(script.index("gcc"), script.index("ulimit -v"))  # compiler not constrained


@unittest.skipUnless(os.environ.get("SANDBOX_INTEGRATION") == "1", "needs Docker + elab-sandbox image")
class SandboxIntegrationTests(SimpleTestCase):
    def run_py(self, code, **kw):
        return sandbox.run_code("python", code, **kw)

    def test_languages_still_work(self):
        self.assertEqual(self.run_py("print(int(input())*2)", stdin="21")["stdout"], "42")
        c = sandbox.run_code("c", '#include <stdio.h>\nint main(){int a;scanf("%d",&a);printf("%d",a*2);}', stdin="21")
        self.assertEqual(c["stdout"], "42")
        java = sandbox.run_code("java", "public class Main{public static void main(String[] a){System.out.println(42);}}")
        self.assertEqual(java["stdout"], "42")

    def test_bits_stdcpp_compiles_at_default_limit(self):
        r = sandbox.run_code("cpp", "#include <bits/stdc++.h>\nint main(){std::vector<int> v(10,1);"
                                    "std::cout<<std::accumulate(v.begin(),v.end(),0);}")
        self.assertEqual((r["status"], r["stdout"]), ("Accepted", "10"))

    def test_native_memory_limit_still_enforced(self):
        r = sandbox.run_code("c", "#include <stdlib.h>\n#include <string.h>\n#include <stdio.h>\n"
                                  "int main(){char*p=malloc(200<<20);if(!p){puts(\"no\");return 1;}memset(p,1,200<<20);}")
        self.assertEqual(r["status"], "Runtime Error")

    def test_stdout_flood_is_capped(self):
        r = self.run_py("while True: print('x'*1000)")
        self.assertEqual(r["status"], "Runtime Error")
        self.assertLessEqual(len(r["stdout"]), sandbox.STDOUT_LIMIT_BYTES)

    def test_stderr_flood_is_not_accepted(self):
        r = self.run_py("import sys\nwhile True: sys.stderr.write('e'*1000)", time_limit=2)
        self.assertNotEqual(r["status"], "Accepted")

    def test_cannot_write_host_mount_or_rootfs(self):
        self.assertEqual(self.run_py("open('/box/x','w').write('x')")["status"], "Runtime Error")
        self.assertEqual(self.run_py("open('/etc/x','w').write('x')")["status"], "Runtime Error")

    def test_runs_as_nobody_without_network(self):
        self.assertEqual(self.run_py("import os;print(os.getuid())")["stdout"], "65534")
        net = self.run_py("import socket;socket.create_connection(('1.1.1.1',80),timeout=2)")
        self.assertEqual(net["status"], "Runtime Error")

    def test_time_limit(self):
        self.assertEqual(self.run_py("while True: pass", time_limit=1)["status"], "Time Limit Exceeded")


class SandboxViaWorkerTests(SimpleTestCase):
    """F2: the web process runs code through a Celery task, so it needs no Docker socket."""

    def test_run_code_goes_through_celery_task(self):
        from django.test import override_settings

        from . import views

        with override_settings(SANDBOX_VIA_WORKER=True, CELERY_TASK_ALWAYS_EAGER=True), \
                patch("core.sandbox.run_code", return_value={"status_id": 3, "stdout": "ok"}) as run, \
                patch("core.views._local_run_code") as local:
            out = views.sandbox_run_code("python", "print('ok')", stdin="", time_limit=1)
        self.assertEqual(out["stdout"], "ok")
        run.assert_called_once()
        local.assert_not_called()

    def test_health_uses_worker_ping_not_docker(self):
        from django.test import Client, override_settings

        with override_settings(SANDBOX_VIA_WORKER=True), \
                patch("core.views.check_database", return_value=True), \
                patch("core.views.subprocess.run") as docker_cli, \
                patch("config.celery.app.control.ping", return_value=[{"w1": {"ok": "pong"}}]):
            ok = Client().get("/health/")
        self.assertEqual(ok.status_code, 200)
        docker_cli.assert_not_called()
        with override_settings(SANDBOX_VIA_WORKER=True), \
                patch("core.views.check_database", return_value=True), \
                patch("config.celery.app.control.ping", return_value=[]):
            down = Client().get("/health/")
        self.assertEqual(down.status_code, 503)
