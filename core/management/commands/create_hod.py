import os

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model


class Command(BaseCommand):
    help = 'Creates a Head of Department (HOD) user'

    def add_arguments(self, parser):
        parser.add_argument('--username', default='hod')
        parser.add_argument(
            '--password',
            default=os.environ.get('HOD_PASSWORD', ''),
            help='Password for the HOD account (or set HOD_PASSWORD).',
        )

    def handle(self, *args, **options):
        User = get_user_model()

        if User.objects.filter(role=User.Role.HOD).exists():
            self.stdout.write(self.style.WARNING('HOD user already exists'))
            return

        password = options['password']
        if not password:
            raise CommandError('Provide --password or set HOD_PASSWORD; no default password is used.')

        hod = User.objects.create_user(
            username=options['username'],
            password=password,
            first_name='Head',
            last_name='Department',
            email='hod@example.com',
            role=User.Role.HOD,
            is_staff=True
        )

        self.stdout.write(self.style.SUCCESS(f'Successfully created HOD user: {hod.username}'))
