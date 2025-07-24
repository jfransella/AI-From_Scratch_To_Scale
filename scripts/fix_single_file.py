#!/usr/bin/env python3
"""
Fix linting issues for a single file using pylint and mypy.

Usage:
    python scripts/fix_single_file.py path/to/file.py [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_pylint_issues(file_path: Path) -> tuple[bool, str]:
    """Check for pylint issues."""
    result = subprocess.run(
        [sys.executable, "-m", "pylint", str(file_path)], capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout


def check_mypy_issues(file_path: Path) -> tuple[bool, str]:
    """Check for mypy issues."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", str(file_path)], capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout


def fix_file_linting(file_path: Path, dry_run: bool = False) -> int:
    """Fix linting issues for a single file."""
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return 1

    if file_path.suffix != ".py":
        print(f"Error: {file_path} is not a Python file")
        return 1

    print(f"{'[DRY RUN] ' if dry_run else ''}Fixing linting issues in: {file_path}")

    fixes_applied = 0

    # 1. Check current issues
    print("\n1. Checking current issues...")

    # Check pylint
    pylint_clean, pylint_output = check_pylint_issues(file_path)
    if not pylint_clean:
        print("   Pylint issues found:")
        print(pylint_output)

    # Check mypy
    mypy_clean, mypy_output = check_mypy_issues(file_path)
    if not mypy_clean:
        print("   Mypy issues found:")
        print(mypy_output)

    if pylint_clean and mypy_clean:
        print("   ✅ No linting issues found!")
        return 0

    if dry_run:
        print("\n[DRY RUN] Would apply the following fixes:")

    # 2. Remove unused imports
    print(f"\n2. {'[DRY RUN] ' if dry_run else ''}Removing unused imports...")
    cmd = [sys.executable, "-m", "autoflake", "--remove-all-unused-imports"]
    if not dry_run:
        cmd.append("--in-place")
    cmd.append(str(file_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        if dry_run and result.stdout:
            print("   Would remove unused imports")
            fixes_applied += 1
        elif not dry_run:
            print("   ✅ Removed unused imports")
            fixes_applied += 1

    # 3. Fix basic formatting
    print(f"\n3. {'[DRY RUN] ' if dry_run else ''}Applying basic formatting...")
    cmd = [sys.executable, "-m", "autopep8", "--max-line-length", "120"]
    if dry_run:
        cmd.append("--diff")
    else:
        cmd.append("--in-place")
    cmd.append(str(file_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        if dry_run and result.stdout.strip():
            print("   Would apply formatting fixes")
            fixes_applied += 1
        elif not dry_run:
            print("   ✅ Applied formatting fixes")
            fixes_applied += 1

    # 4. Sort imports
    print(f"\n4. {'[DRY RUN] ' if dry_run else ''}Sorting imports...")
    cmd = [sys.executable, "-m", "isort", "--line-length", "120"]
    if dry_run:
        cmd.append("--check-only")
    cmd.append(str(file_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if not dry_run and result.returncode == 0:
        print("   ✅ Sorted imports")
        fixes_applied += 1
    elif dry_run and result.returncode != 0:
        print("   Would sort imports")
        fixes_applied += 1

    # 5. Final check
    if not dry_run:
        print(f"\n5. Final check...")

        # Check pylint
        pylint_clean, pylint_output = check_pylint_issues(file_path)
        if pylint_clean:
            print("   ✅ Pylint: All issues fixed!")
        else:
            print("   ⚠️  Pylint: Remaining issues:")
            print(pylint_output)

        # Check mypy
        mypy_clean, mypy_output = check_mypy_issues(file_path)
        if mypy_clean:
            print("   ✅ Mypy: All issues fixed!")
        else:
            print("   ⚠️  Mypy: Remaining issues:")
            print(mypy_output)

        if pylint_clean and mypy_clean:
            print("   🎉 All linting tools report success!")
        else:
            print("   💡 Some issues may need manual fixing")

    print(
        f"\n{'[DRY RUN] ' if dry_run else ''}Summary: {fixes_applied} fixes {'would be ' if dry_run else ''}applied"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fix linting issues for a single file")
    parser.add_argument("file", help="Python file to fix")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )

    args = parser.parse_args()

    file_path = Path(args.file)
    return fix_file_linting(file_path, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
