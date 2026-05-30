#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VILLAGE_REPO = REPO_ROOT / "village"
if VILLAGE_REPO.exists() and str(VILLAGE_REPO) not in sys.path:
    sys.path.insert(0, str(VILLAGE_REPO))

from village.scripts.global_csv_for_subject import main as merge_subject_csv


DEFAULT_PROJECT = "COT_cannula_data"
DEFAULT_LOCAL_ROOT = Path("/home/pi/village_projects")
DEFAULT_REMOTE = "training_village@minibaps"
DEFAULT_REMOTE_ROOT = "/storage/training_village"


def run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command))
    if not dry_run:
        subprocess.run(command, check=True)


def read_deleted_sessions(project_dir: Path) -> list[str]:
    deleted_sessions: set[str] = set()
    deleted_files = sorted(project_dir.glob("village*/deleted_sessions.csv"))

    for deleted_file in deleted_files:
        with deleted_file.open(newline="") as file:
            for row in csv.DictReader(file):
                filename = (row.get("filename") or "").strip()
                if filename:
                    deleted_sessions.add(filename)

    print(
        f"Loaded {len(deleted_sessions)} deleted session filename(s) "
        f"from {len(deleted_files)} local file(s)."
    )
    for deleted_file in deleted_files:
        print(f"  deleted sessions source: {deleted_file}")

    return sorted(deleted_sessions)


def list_subjects(sessions_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in sessions_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all subject CSVs from separate session files, then sync only "
            "the merged subject CSVs to mini."
        )
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=DEFAULT_LOCAL_ROOT,
        help="Default: /home/pi/village_projects",
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_dir = args.local_root / args.project
    sessions_dir = project_dir / "data" / "sessions"
    if not sessions_dir.exists():
        sessions_dir = project_dir / "sessions"
    remote_project_dir = f"{args.remote_root.rstrip('/')}/{args.project}"

    if not sessions_dir.exists():
        raise FileNotFoundError(f"sessions directory not found: {sessions_dir}")

    deleted_sessions = read_deleted_sessions(project_dir)
    subjects = list_subjects(sessions_dir)

    print(f"Project: {args.project}")
    print(f"Local project directory: {project_dir}")
    print(f"Local sessions directory: {sessions_dir}")
    print(f"Remote project directory: {args.remote}:{remote_project_dir}")
    print(f"Subjects ({len(subjects)}): {', '.join(subjects)}")

    failures: list[tuple[str, str]] = []
    for subject in subjects:
        local_csv = sessions_dir / subject / f"{subject}.csv"
        remote_subject_dir = f"{remote_project_dir}/sessions/{subject}/"

        print(f"\n=== {subject} ===")
        print(f"merge output: {local_csv}")
        print(f"remote target: {args.remote}:{remote_subject_dir}{subject}.csv")

        try:
            if args.dry_run:
                print(
                    "dry run: skipping merge_subject_csv("
                    f"subject={subject!r}, deleted_sessions={len(deleted_sessions)})"
                )
            else:
                merge_subject_csv(
                    subject=subject,
                    sessions_directory=str(sessions_dir),
                    deleted_sessions=deleted_sessions,
                )

            run(
                ["ssh", args.remote, f"mkdir -p {shlex.quote(remote_subject_dir)}"],
                dry_run=args.dry_run,
            )
            run(
                ["rsync", "-avz", str(local_csv), f"{args.remote}:{remote_subject_dir}"],
                dry_run=args.dry_run,
            )
        except Exception as exc:
            failures.append((subject, f"{type(exc).__name__}: {exc}"))
            print(f"[failed] {subject}: {type(exc).__name__}: {exc}")

    if failures:
        print("\nFailures:")
        for subject, error in failures:
            print(f"  {subject}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
