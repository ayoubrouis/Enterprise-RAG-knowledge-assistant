"""Admin CLI for tenants, users, and API keys (no web API required).

Passwords are auto-generated (shown once) when --password is omitted.

Usage:
    python scripts/manage.py ensure                          # show setup status
    python scripts/manage.py ensure --username admin --password 'YourPass123'  # create first admin
    python scripts/manage.py create-tenant --id acme --name "Acme Corp"
    python scripts/manage.py create-user --tenant acme --username alice --role user
    python scripts/manage.py create-admin --username root
    python scripts/manage.py create-api-key --tenant acme --label "prod"
    python scripts/manage.py list-tenants
    python scripts/manage.py list-users --tenant acme
    python scripts/manage.py reset-password --username alice
    python scripts/manage.py backup --out /mnt/backups       # DB + tenant data snapshot
    python scripts/manage.py restore --from /mnt/backups/backup_20260809_120000  # restore (--force to overwrite)
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import sqlite3
import sys
import time
from pathlib import Path

# Make `app` importable when running this script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db
from app.config import settings


def _password_or_generate(provided: str) -> tuple[str, bool]:
    """Return (password, was_generated). Empty input -> strong random password
    that is shown once (only its hash is stored)."""
    if provided:
        return provided, False
    return secrets.token_urlsafe(12), True


def _check_password(password: str) -> None:
    if len(password) < 8:
        print("Error: password must be at least 8 characters.")
        sys.exit(1)


def cmd_ensure(args: argparse.Namespace) -> None:
    db.seed_defaults()
    if db.is_bootstrapped():
        print(f"Default tenant '{settings.DEFAULT_TENANT}' and admin user ensured.")
        return
    if args.password or args.superadmin:
        role = "superadmin" if args.superadmin else "admin"
        password, generated = _password_or_generate(args.password)
        _check_password(password)
        db.run_setup(args.tenant_name, args.username, password, role=role)
        print(f"Created {role} '{args.username}' for tenant '{settings.DEFAULT_TENANT}'.")
        if generated:
            print(f"Generated password (shown once): {password}")
        return
    print(
        "No admin exists yet. Run the setup wizard in the UI at "
        "http://localhost:8501, or create one with:\n"
        "  python scripts/manage.py ensure --username admin --password 'YourPass123'\n"
        "For a platform admin (superadmin) add: --superadmin"
    )


def cmd_create_tenant(args: argparse.Namespace) -> None:
    db.seed_defaults()
    if db.get_tenant(args.tenant_id):
        print(f"Tenant '{args.tenant_id}' already exists.")
        return
    db.create_tenant(args.tenant_id, args.name)
    print(f"Created tenant '{args.tenant_id}' ({args.name}).")


def cmd_create_user(args: argparse.Namespace) -> None:
    db.seed_defaults()
    password, generated = _password_or_generate(args.password)
    _check_password(password)
    try:
        user = db.create_user(args.tenant, args.username, password, args.role)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Created {user['role']} user '{user['username']}' in tenant '{user['tenant_id']}'.")
    if generated:
        print(f"Generated password (shown once): {password}")


def cmd_create_admin(args: argparse.Namespace) -> None:
    db.seed_defaults()
    password, generated = _password_or_generate(args.password)
    _check_password(password)
    role = "superadmin" if args.superadmin else "admin"
    try:
        user = db.create_user(args.tenant, args.username, password, role)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Created {role} user '{user['username']}' in tenant '{user['tenant_id']}'.")
    if generated:
        print(f"Generated password (shown once): {password}")


def cmd_create_api_key(args: argparse.Namespace) -> None:
    db.seed_defaults()
    if not db.get_tenant(args.tenant):
        print(f"Error: tenant '{args.tenant}' does not exist.")
        sys.exit(1)
    plain, _ = db.create_api_key(args.tenant, args.label)
    print(f"API key for tenant '{args.tenant}':")
    print(f"  {plain}")
    print("Store it now - it is shown only once.")


def cmd_list_tenants(_: argparse.Namespace) -> None:
    db.seed_defaults()
    print(f"{'tenant_id':<20} {'name':<24} users docs api_keys active")
    for t in db.list_tenants():
        print(
            f"{t['tenant_id']:<20} {t['name']:<24} "
            f"{db.count_users(t['tenant_id']):<5} {db.count_documents(t['tenant_id']):<4} "
            f"{db.count_api_keys(t['tenant_id']):<8} {t['is_active']}"
        )


def cmd_list_users(args: argparse.Namespace) -> None:
    db.seed_defaults()
    for u in db.list_users(args.tenant):
        print(f"{u['username']:<24} role={u['role']:<6} active={u['is_active']}")


def cmd_reset_password(args: argparse.Namespace) -> None:
    db.seed_defaults()
    user = db.get_user_by_username(args.username)
    if user is None:
        print(f"Error: no user named '{args.username}'.")
        sys.exit(1)
    password, generated = _password_or_generate(args.password)
    _check_password(password)
    db.set_user_password(user["id"], password)
    db.clear_login_failures(args.username)  # reset password also lifts a login lockout
    print(f"Password updated for '{args.username}'.")
    if generated:
        print(f"Generated password (shown once): {password}")


def _sqlite_snapshot(src_path: Path, dst_path: Path) -> None:
    """Consistent DB copy via the sqlite3 online-backup API (WAL-safe)."""
    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(dst_path))
    try:
        with dst:
            src.backup(dst)
    finally:
        src.close()
        dst.close()


def cmd_backup(args: argparse.Namespace) -> None:
    """Snapshot the SQLite DB + all tenant data into a timestamped directory."""
    if not settings.DB_PATH.exists():
        print(f"Error: no database found at {settings.DB_PATH}. Nothing to back up.")
        sys.exit(1)

    out = Path(args.out) if args.out else settings.DATA_DIR / "backups"
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    while True:  # unique timestamped name (handles two runs in one second)
        name = time.strftime("backup_%Y%m%d_%H%M%S")
        if n:
            name += f"_{n}"
        dest = out / name
        try:
            dest.mkdir(parents=False)
            break
        except FileExistsError:
            n += 1

    _sqlite_snapshot(settings.DB_PATH, dest / settings.DB_PATH.name)
    tenants = []
    if settings.TENANTS_DIR.exists():
        shutil.copytree(settings.TENANTS_DIR, dest / "tenants", dirs_exist_ok=True)
        tenants = sorted(p.name for p in (dest / "tenants").iterdir() if p.is_dir())

    (dest / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": time.time(),
                "tool": "scripts/manage.py backup",
                "db_file": settings.DB_PATH.name,
                "tenants": tenants,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Backup written to {dest}")
    print(f"  DB snapshot : {settings.DB_PATH.name}")
    print(f"  Tenants     : {', '.join(tenants) if tenants else '(none)'}")


def cmd_restore(args: argparse.Namespace) -> None:
    """Restore the DB + tenant data from a backup directory."""
    src = Path(args.from_path)
    db_backup = src / settings.DB_PATH.name
    tenants_backup = src / "tenants"
    if not db_backup.exists() and not tenants_backup.exists():
        print(
            f"Error: '{src}' is not a backup (no {settings.DB_PATH.name} or tenants/ found)."
        )
        sys.exit(1)

    if not args.force:
        existing = settings.DB_PATH.exists() or (
            settings.TENANTS_DIR.exists() and any(settings.TENANTS_DIR.iterdir())
        )
        if existing:
            print("Error: destination already contains data. Use --force to overwrite.")
            sys.exit(1)

    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if db_backup.exists():
        for suffix in ("-wal", "-shm"):  # drop stale WAL sidecars of the current DB
            Path(f"{settings.DB_PATH}{suffix}").unlink(missing_ok=True)
        shutil.copy2(db_backup, settings.DB_PATH)

    if tenants_backup.exists():
        if settings.TENANTS_DIR.exists():
            shutil.rmtree(settings.TENANTS_DIR)
        shutil.copytree(tenants_backup, settings.TENANTS_DIR)

    if db._CONN is not None:  # drop the old handle so the restored DB is reopened
        db._CONN.close()
        db._CONN = None

    print(f"Restored from {src}")
    print(f"  DB      : {settings.DB_PATH}")
    print(f"  Tenants : {settings.TENANTS_DIR if tenants_backup.exists() else '(none in backup)'}")
    print("If the API is running, restart it so in-memory pipelines reload restored indexes.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ensure", help="Ensure default tenant + admin exist")
    p.add_argument("--tenant-name", default=settings.DEFAULT_TENANT)
    p.add_argument("--username", default=settings.ADMIN_USERNAME)
    p.add_argument("--password", default="")
    p.add_argument("--superadmin", action="store_true", help="Create a platform admin (superadmin)")
    p.set_defaults(func=cmd_ensure)

    p = sub.add_parser("create-tenant", help="Create a tenant")
    p.add_argument("--id", dest="tenant_id", required=True)
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_create_tenant)

    p = sub.add_parser("create-user", help="Create a regular user")
    p.add_argument("--tenant", default=settings.DEFAULT_TENANT)
    p.add_argument("--username", required=True)
    p.add_argument("--role", default="user", choices=["admin", "user"])
    p.add_argument("--password", default="")
    p.set_defaults(func=cmd_create_user)

    p = sub.add_parser("create-admin", help="Create an enterprise admin (or platform admin with --superadmin)")
    p.add_argument("--tenant", default=settings.DEFAULT_TENANT)
    p.add_argument("--username", required=True)
    p.add_argument("--password", default="")
    p.add_argument("--superadmin", action="store_true", help="Create a platform admin instead")
    p.set_defaults(func=cmd_create_admin)

    p = sub.add_parser("create-api-key", help="Create a tenant API key")
    p.add_argument("--tenant", default=settings.DEFAULT_TENANT)
    p.add_argument("--label", default="")
    p.set_defaults(func=cmd_create_api_key)

    sub.add_parser("list-tenants", help="List tenants").set_defaults(func=cmd_list_tenants)

    p = sub.add_parser("list-users", help="List a tenant's users")
    p.add_argument("--tenant", default=settings.DEFAULT_TENANT)
    p.set_defaults(func=cmd_list_users)

    p = sub.add_parser("reset-password", help="Reset a user's password")
    p.add_argument("--username", required=True)
    p.add_argument("--password", default="")
    p.set_defaults(func=cmd_reset_password)

    p = sub.add_parser("backup", help="Snapshot the DB + tenant data")
    p.add_argument("--out", default="", help="Destination directory (default: data/backups)")
    p.set_defaults(func=cmd_backup)

    p = sub.add_parser("restore", help="Restore the DB + tenant data from a backup")
    p.add_argument("--from", dest="from_path", required=True, help="Backup directory to restore from")
    p.add_argument("--force", action="store_true", help="Overwrite existing data")
    p.set_defaults(func=cmd_restore)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
