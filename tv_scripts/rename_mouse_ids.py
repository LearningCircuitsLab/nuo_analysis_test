# %%
from __future__ import annotations

import codecs
import os
import re
from collections import Counter
from pathlib import Path
from typing import Callable

# %%
RETROCRE_ROOT = Path("/mnt/c/Users/90611/Desktop/retrocre_copy")
DATA_ROOT = RETROCRE_ROOT / "data"
SESSIONS_DIR = DATA_ROOT / "sessions"
VIDEOS_DIR = DATA_ROOT / "videos"
VILLAGE02_DIR = DATA_ROOT / "village02"

MOUSE_ID_MAP = {
    "ANR001": "ACV023",
    "ANR002": "ACV024",
    "ANR003": "ACV026",
    "ANR004": "ACV027",
    "ANR005": "ACV028",
    "ANR006": "ACV029",
    "ANR007": "ACV030",
    "ANR008": "ACV031",
    "ANR009": "ACV033",
}

# Keep this False for the first run. Change to True only after checking.
APPLY_CHANGES = True

# For your current requirement, keep this False.
# If True, CSV contents under videos/ will also be changed.
INCLUDE_VIDEO_CSVS = True

# Limit printed preview rows so the notebook does not become too noisy.
# Set to None if you want to print every planned operation.
MAX_PREVIEW_ROWS = 100

# Text file types to scan inside village02.
VILLAGE02_TEXT_SUFFIXES = {".csv", ".json", ".log", ".txt"}

# %%
def validate_mapping(raw_mapping: dict[str, str]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for old, new in raw_mapping.items():
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError("All mapping keys and values must be strings.")
        if not old or not new:
            raise ValueError("Mapping keys and values cannot be empty.")
        if old == new:
            continue
        mapping[old] = new

    if not mapping:
        raise ValueError("Mapping does not contain any real renames.")

    target_counts = Counter(mapping.values())
    duplicate_targets = {
        value for value, count in target_counts.items() if count > 1
    }
    if duplicate_targets:
        joined = ", ".join(sorted(duplicate_targets))
        raise ValueError(f"Duplicate target IDs in mapping: {joined}")

    return mapping


mapping = validate_mapping(MOUSE_ID_MAP)
mapping

# %%
def build_replacer(
    mapping: dict[str, str],
) -> tuple[re.Pattern[str], Callable[[str], str]]:
    ordered_keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(key) for key in ordered_keys))

    def replace(text: str) -> str:
        return pattern.sub(lambda match: mapping[match.group(0)], text)

    return pattern, replace


pattern, replace_text = build_replacer(mapping)

# %%
sessions_dir = SESSIONS_DIR.expanduser()
videos_dir = VIDEOS_DIR.expanduser()
village02_dir = VILLAGE02_DIR.expanduser()

print(f"Sessions: {sessions_dir}")
print(f"Videos: {videos_dir}")
print(f"Village02: {village02_dir}")

# %%
def read_text_preserving_encoding(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    if raw.startswith(codecs.BOM_UTF8):
        return raw[len(codecs.BOM_UTF8) :].decode("utf-8"), "utf-8-sig"

    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def write_text_preserving_encoding(
    path: Path,
    text: str,
    encoding: str,
) -> None:
    if encoding == "utf-8-sig":
        path.write_bytes(codecs.BOM_UTF8 + text.encode("utf-8"))
        return
    path.write_bytes(text.encode(encoding))


# %%
def find_csv_ops(
    roots: list[Path],
    pattern: re.Pattern[str],
    replace: Callable[[str], str],
) -> list[tuple[Path, int]]:
    return find_text_ops(roots, pattern, replace, suffixes={".csv"})


def find_text_ops(
    roots: list[Path],
    pattern: re.Pattern[str],
    replace: Callable[[str], str],
    suffixes: set[str],
) -> list[tuple[Path, int]]:
    text_ops: list[tuple[Path, int]] = []

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            text, _encoding = read_text_preserving_encoding(path)
            matches = len(pattern.findall(text))
            if matches == 0:
                continue
            new_text = replace(text)
            if new_text != text:
                text_ops.append((path, matches))

    return text_ops


# %%
def collect_rename_ops(
    roots: list[Path],
    replace: Callable[[str], str],
) -> list[tuple[Path, Path, str]]:
    rename_ops: list[tuple[Path, Path, str]] = []

    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            current_dir = Path(dirpath)

            for filename in filenames:
                new_name = replace(filename)
                if new_name != filename:
                    rename_ops.append(
                        (
                            current_dir / filename,
                            current_dir / new_name,
                            "file",
                        )
                    )

            for dirname in dirnames:
                new_name = replace(dirname)
                if new_name != dirname:
                    rename_ops.append(
                        (
                            current_dir / dirname,
                            current_dir / new_name,
                            "dir",
                        )
                    )

    return rename_ops


# %%
def validate_rename_ops(
    rename_ops: list[tuple[Path, Path, str]],
) -> None:
    if not rename_ops:
        return

    targets_seen: dict[Path, Path] = {}
    conflicts: list[str] = []

    for source, target, _kind in rename_ops:
        if target in targets_seen:
            conflicts.append(
                f"two paths would rename to {target}: "
                f"{targets_seen[target]} and {source}"
            )
            continue
        targets_seen[target] = source

        if target.exists():
            conflicts.append(f"target already exists: {target}")

    if conflicts:
        details = "\n".join(f"  - {item}" for item in conflicts)
        raise ValueError(f"Rename conflicts found:\n{details}")


# %%
def preview_items(items: list, formatter: Callable, max_rows: int | None) -> None:
    shown_items = items if max_rows is None else items[:max_rows]
    for item in shown_items:
        print(formatter(item))
    if max_rows is not None and len(items) > max_rows:
        remaining = len(items) - max_rows
        print(f"... {remaining} more not shown")


csv_roots = [sessions_dir]
if INCLUDE_VIDEO_CSVS:
    csv_roots.append(videos_dir)

csv_ops = find_csv_ops(csv_roots, pattern, replace_text)
village02_content_ops = find_text_ops(
    [village02_dir],
    pattern,
    replace_text,
    suffixes=VILLAGE02_TEXT_SUFFIXES,
)
rename_ops = collect_rename_ops([sessions_dir, videos_dir], replace_text)
validate_rename_ops(rename_ops)

mode = "APPLY" if APPLY_CHANGES else "DRY RUN"
print(f"Mode: {mode}")
print(f"Mapping: {mapping}")
print(f"CSV files to edit: {len(csv_ops)}")
preview_items(
    csv_ops,
    lambda op: f"CSV: {op[0]} ({op[1]} replacements)",
    MAX_PREVIEW_ROWS,
)
print()

print(f"Village02 text files to edit: {len(village02_content_ops)}")
preview_items(
    village02_content_ops,
    lambda op: f"VILLAGE02: {op[0]} ({op[1]} replacements)",
    MAX_PREVIEW_ROWS,
)
print()

print(f"Files/directories to rename: {len(rename_ops)}")
preview_items(
    rename_ops,
    lambda op: f"{op[2].upper()}: {op[0]} -> {op[1]}",
    MAX_PREVIEW_ROWS,
)

# %%
def apply_csv_ops(
    csv_ops: list[tuple[Path, int]],
    replace: Callable[[str], str],
    apply_changes: bool,
) -> None:
    for path, _replacements in csv_ops:
        if not apply_changes:
            continue
        text, encoding = read_text_preserving_encoding(path)
        new_text = replace(text)
        write_text_preserving_encoding(path, new_text, encoding)


def apply_rename_ops(
    rename_ops: list[tuple[Path, Path, str]],
    apply_changes: bool,
) -> None:
    for source, target, _kind in rename_ops:
        if apply_changes:
            source.rename(target)


if not APPLY_CHANGES:
    print("Dry run only. Set APPLY_CHANGES = True and rerun to write.")
else:
    current_csv_roots = [sessions_dir]
    if INCLUDE_VIDEO_CSVS:
        current_csv_roots.append(videos_dir)

    current_csv_ops = find_csv_ops(
        current_csv_roots,
        pattern,
        replace_text,
    )
    current_village02_content_ops = find_text_ops(
        [village02_dir],
        pattern,
        replace_text,
        suffixes=VILLAGE02_TEXT_SUFFIXES,
    )
    current_rename_ops = collect_rename_ops(
        [sessions_dir, videos_dir],
        replace_text,
    )
    validate_rename_ops(current_rename_ops)

    print(f"CSV files to edit now: {len(current_csv_ops)}")
    print(
        "Village02 text files to edit now: "
        f"{len(current_village02_content_ops)}"
    )
    print(f"Files/directories to rename now: {len(current_rename_ops)}")

    apply_csv_ops(current_csv_ops, replace_text, apply_changes=True)
    apply_csv_ops(
        current_village02_content_ops,
        replace_text,
        apply_changes=True,
    )
    apply_rename_ops(current_rename_ops, apply_changes=True)
    print("Done.")
