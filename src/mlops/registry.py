"""Command-line tool to inspect, promote and roll back registered models."""

import argparse
import getpass
from datetime import UTC, datetime

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlops import settings

NAME = settings.MODEL_NAME
CHAMPION = settings.CHAMPION_ALIAS
CHALLENGER = settings.CHALLENGER_ALIAS


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)


def _alias_version(client: MlflowClient, alias: str) -> str | None:
    """Version number an alias points to, or None if the alias is not set."""
    try:
        return str(client.get_model_version_by_alias(NAME, alias).version)
    except MlflowException:
        return None


def status() -> None:
    client = _client()
    for alias in (CHAMPION, CHALLENGER):
        version = _alias_version(client, alias)
        if version is None:
            print(f"{alias:<11} (none)")
            continue
        tags = client.get_model_version(NAME, version).tags
        print(
            f"{alias:<11} v{version}  holdout RMSE {tags.get('holdout_rmse', '?')}"
            f"  gate passed: {tags.get('gate_passed', '?')}"
        )


def promote(version: str | None) -> str:
    """Make a version the champion. Defaults to the current challenger.

    Refuses any version that did not pass the gate, and records who promoted it, when, and
    which version it replaced (so rollback knows where to go back to).
    """
    client = _client()
    version = version or _alias_version(client, CHALLENGER)
    if version is None:
        raise SystemExit("No challenger to promote. Run the training job first.")
    tags = client.get_model_version(NAME, version).tags
    if tags.get("gate_passed") != "true":
        raise SystemExit(f"v{version} did not pass the gate: {tags.get('gate_reasons')}")
    previous = _alias_version(client, CHAMPION)
    if previous == version:
        raise SystemExit(f"v{version} is already the champion.")

    client.set_registered_model_alias(NAME, CHAMPION, version)
    client.set_model_version_tag(NAME, version, "previous_champion", previous or "none")
    client.set_model_version_tag(NAME, version, "promoted_at", datetime.now(UTC).isoformat())
    client.set_model_version_tag(NAME, version, "promoted_by", getpass.getuser())
    if _alias_version(client, CHALLENGER) == version:
        client.delete_registered_model_alias(NAME, CHALLENGER)
    print(f"v{version} is now the champion (previous: {previous or 'none'}).")
    print("Restart the API to serve it: docker compose restart api")
    return version


def rollback(version: str | None) -> str:
    """Point the champion alias back to an older version. Defaults to the previous champion."""
    client = _client()
    current = _alias_version(client, CHAMPION)
    if current is None:
        raise SystemExit("There is no champion to roll back.")
    if version is None:
        version = client.get_model_version(NAME, current).tags.get("previous_champion")
    if not version or version == "none":
        raise SystemExit("The champion has no recorded previous version; pass --version.")
    client.get_model_version(NAME, version)  # fails loudly if the version does not exist

    client.set_registered_model_alias(NAME, CHAMPION, version)
    client.set_model_version_tag(NAME, version, "rolled_back_from", current)
    client.set_model_version_tag(NAME, version, "rolled_back_at", datetime.now(UTC).isoformat())
    print(f"Rolled back: v{version} is the champion again (was v{current}).")
    print("Restart the API to serve it: docker compose restart api")
    return version


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mlops-model", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show the champion and challenger versions.")
    for name in ("promote", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("--version", help="Model version (default: the obvious one).")
    args = parser.parse_args(argv)

    if args.command == "status":
        status()
    elif args.command == "promote":
        promote(args.version)
    else:
        rollback(args.version)


if __name__ == "__main__":
    main()
