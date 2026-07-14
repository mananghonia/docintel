"""Create (or fetch) an API user and print a token for authenticated access.

    python manage.py create_api_token --username reviewer --password secret

Then call the API with:  Authorization: Token <printed token>
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from rest_framework.authtoken.models import Token


class Command(BaseCommand):
    help = "Create or fetch a user and print their API token."

    def add_arguments(self, parser):
        parser.add_argument("--username", default="apiuser")
        parser.add_argument("--password", default=None,
                            help="set/reset the user's password (optional)")

    def handle(self, *args, **opts):
        User = get_user_model()
        user, created = User.objects.get_or_create(username=opts["username"])
        if opts["password"]:
            user.set_password(opts["password"])
            user.save()
        token, _ = Token.objects.get_or_create(user=user)
        self.stdout.write(self.style.SUCCESS(
            f"user {'created' if created else 'exists'}: {user.username}"))
        self.stdout.write(f"token: {token.key}")
        self.stdout.write('use header:  Authorization: Token ' + token.key)
