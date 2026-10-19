from __future__ import annotations

import argparse
import getpass
import re

from web_app.persistence import _initialize_database, set_user_password


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an application user or reset its password."
    )
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{3,64}", args.username):
        parser.error("Username must be 3-64 letters, numbers, '.', '_', '@' or '-'.")

    password = getpass.getpass("New password (12+ characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if len(password) < 12:
        parser.error("Password must contain at least 12 characters.")
    if password != confirmation:
        parser.error("Passwords do not match.")

    _initialize_database()
    set_user_password(args.username, password)
    print(f"User '{args.username}' is ready. Existing sessions were revoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
