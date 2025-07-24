#!/usr/bin/env python3
"""
Quick linting fixer for common issues in the AI-From_Scratch_To_Scale project.

This script fixes the most common and easily automated issues:
- Trailing whitespace (Pylint trailing-whitespace)
- Missing final newlines (Pylint missing-final-newline)
- Unused imports (Pylint unused-import)
- Basic black formatting
- Type hint issues (Mypy)

Usage:
    python scripts/quick_lint_fix.py [--dry-run] [--path PATH]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, dry_run=False):
    """Run a command, handling dry run mode."""
    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def fix_whitespace_issues(file_path, dry_run=False):
    """Fix trailing whitespace and missing newlines."""
    if dry_run:
        print(f"[DRY RUN] Would fix whitespace in {file_path}")
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Fix trailing whitespace on lines
        lines = content.split("\n")
        lines = [line.rstrip() for line in lines]

        # Ensure file ends with newline
        if lines and lines[-1]:
            lines.append("")

        # Fix blank lines with whitespace (W293)
        content = "\n".join(lines)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Count fixes
            fixes = len(re.findall(r"[ \t]+$", original_content, re.MULTILINE))
            if not original_content.endswith("\n"):
                fixes += 1

            print(f"Fixed {fixes} whitespace issues in {file_path}")
            return fixes

        return 0

    except Exception as e:
        print(f"Error fixing whitespace in {file_path}: {e}")
        return 0


def fix_unused_imports(file_path, dry_run=False):
    """Fix unused imports using autoflake."""
    cmd = [
        sys.executable,
        "-m",
        "autoflake",
        "--remove-all-unused-imports",
        "--remove-unused-variables",
    ]

    if not dry_run:
        cmd.append("--in-place")

    cmd.append(str(file_path))

    success = run_command(cmd, dry_run=dry_run)
    if success:
        if not dry_run:
            print(f"Fixed unused imports in {file_path}")
        return 1
    return 0


def fix_basic_formatting(file_path, dry_run=False):
    """Fix basic formatting with autopep8."""
    cmd = [
        sys.executable,
        "-m",
        "autopep8",
        "--max-line-length",
        "120",
        "--select",
        "E1,E2,E3,W1,W2,W3",  # Basic formatting only
    ]

    if not dry_run:
        cmd.append("--in-place")
    else:
        cmd.append("--diff")

    cmd.append(str(file_path))

    success = run_command(cmd, dry_run=dry_run)
    if success:
        if not dry_run:
            print(f"Applied basic formatting to {file_path}")
        return 1
    return 0


def process_file(file_path, dry_run=False):
    """Process a single Python file."""
    print(f"\nProcessing: {file_path}")

    total_fixes = 0
    total_fixes += fix_whitespace_issues(file_path, dry_run)
    total_fixes += fix_unused_imports(file_path, dry_run)
    total_fixes += fix_basic_formatting(file_path, dry_run)

    return total_fixes


def process_directory(path, dry_run=False):
    """Process all Python files in a directory."""
    if not path.exists():
        print(f"Path does not exist: {path}")
        return 0

    if path.is_file() and path.suffix == ".py":
        return process_file(path, dry_run)

    total_fixes = 0
    python_files = list(path.rglob("*.py"))

    # Filter out unwanted directories
    python_files = [
        f for f in python_files if "__pycache__" not in str(f) and ".venv" not in str(f)
    ]

    print(f"Found {len(python_files)} Python files in {path}")

    for py_file in python_files:
        total_fixes += process_file(py_file, dry_run)

    return total_fixes


def main():
    parser = argparse.ArgumentParser(description="Quick linting fixer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument("--path", type=str, help="Specific path to process")

    args = parser.parse_args()

    # Default paths to process
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    default_paths = ["data_utils", "utils", "engine", "plotting"]

    if args.path:
        paths = [project_root / args.path]
    else:
        paths = [project_root / path for path in default_paths]

    print(f"Project root: {project_root}")
    print("Starting quick linting fixes...")

    total_fixes = 0
    for path in paths:
        print(f"\n{'='*60}")
        print(f"Processing: {path}")
        print("=" * 60)
        fixes = process_directory(path, args.dry_run)
        total_fixes += fixes

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN - No files were actually modified")
    print(f"Total fixes applied: {total_fixes}")

    if not args.dry_run and total_fixes > 0:
        print("\nRunning final linting check...")

        # Run pylint check
        print("Checking with pylint...")
        run_command(
            [
                sys.executable,
                "-m",
                "pylint",
                "--score=no",
                "--reports=no",
            ]
            + [str(p) for p in paths if p.exists()]
        )

        # Run mypy check
        print("Checking with mypy...")
        run_command(
            [
                sys.executable,
                "-m",
                "mypy",
                "--no-error-summary",
            ]
            + [str(p) for p in paths if p.exists()]
        )


if __name__ == "__main__":
    main()
