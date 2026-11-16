"""Invite an expert (there is no public sign-up), reset a password, or deactivate an account.

    python scripts/create_user.py alice --display-name "Alice Smith" --affiliation UVA
    python scripts/create_user.py admin --admin
    python scripts/create_user.py alice --reset-password
    python scripts/create_user.py alice --deactivate

The password is prompted for, read from stdin with --password-stdin, or generated with --generate.
"""

import argparse
import getpass
import secrets
import sys

import _path  # noqa: F401
from sqlalchemy import select

from app.auth import MIN_PASSWORD_LENGTH, USERNAME_RE, USERNAME_RULE, hash_password
from app.db import SessionLocal
from app.models import User

def get_password(args) -> tuple[str, bool]:
    """Returns (password, was_generated)."""
    if args.generate:
        return secrets.token_urlsafe(12), True
    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\n")
    else:
        password = getpass.getpass("Password: ")
        if getpass.getpass("Repeat password: ") != password:
            sys.exit("Passwords don't match")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    return password, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("username")
    parser.add_argument("--display-name")
    parser.add_argument("--affiliation")
    parser.add_argument("--expertise")
    parser.add_argument("--admin", action="store_true", help="grant admin rights")
    parser.add_argument("--reset-password", action="store_true")
    parser.add_argument("--deactivate", action="store_true")
    parser.add_argument("--activate", action="store_true")
    pw = parser.add_mutually_exclusive_group()
    pw.add_argument("--generate", action="store_true", help="generate a random password and print it")
    pw.add_argument("--password-stdin", action="store_true")
    args = parser.parse_args()

    username = args.username.strip().lower()
    if not USERNAME_RE.match(username):
        sys.exit(f"Username must be {USERNAME_RULE}")

    with SessionLocal() as db:
        user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if args.deactivate or args.activate:
            if user is None:
                sys.exit(f"No user {username!r}")
            user.is_active = args.activate
            db.commit()
            print(f"{username}: {'activated' if args.activate else 'deactivated'}")
            return
        if args.reset_password:
            if user is None:
                sys.exit(f"No user {username!r}")
        elif user is not None:
            sys.exit(f"User {username!r} already exists (use --reset-password to change the password)")

        password, generated = get_password(args)
        if user is None:
            user = User(username=username, display_name=args.display_name, affiliation=args.affiliation,
                        expertise=args.expertise, is_admin=args.admin, password_hash=hash_password(password))
            db.add(user)
            action = "created"
        else:
            user.password_hash = hash_password(password)
            if args.admin:
                user.is_admin = True
            action = "password reset"
        db.commit()
    print(f"{username}: {action}{' (admin)' if user.is_admin else ''}")
    if generated:
        print(f"password: {password}")


if __name__ == "__main__":
    main()
