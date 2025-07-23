#!/usr/bin/env python3
"""
Advanced linting fixer for remaining complex issues.

This script handles the more complex linting issues that the basic fixer can't handle:
- F841: Unused variables
- F541: f-strings without placeholders
- E701: Multiple statements on one line
- E722: Bare except clauses
- F402: Variable shadowing

Usage:
    python scripts/advanced_lint_fix.py [--dry-run] [--path PATH]
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple


def fix_unused_variables(file_path: Path, dry_run: bool = False) -> int:
    """Fix unused variables (F841) by removing assignments or using underscore."""
    if dry_run:
        print(f"[DRY RUN] Would fix unused variables in {file_path}")
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        fixes = 0

        # Pattern for simple unused variable assignments in except clauses
        # except Exception as e: -> except Exception:
        pattern = (
            r"except\s+(\w+)\s+as\s+(\w+):\s*\n(\s*)(pass|continue|break|\.\.\.|return)"
        )

        def replace_unused_except(match):
            exception_type = match.group(1)
            var_name = match.group(2)
            indent = match.group(3)
            statement = match.group(4)
            return f"except {exception_type}:\n{indent}{statement}"

        new_content = re.sub(
            pattern, replace_unused_except, content, flags=re.MULTILINE
        )

        if new_content != content:
            fixes += len(re.findall(pattern, content, flags=re.MULTILINE))
            content = new_content

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {fixes} unused variable issues in {file_path}")
            return fixes

        return 0

    except Exception as e:
        print(f"Error fixing unused variables in {file_path}: {e}")
        return 0


def fix_fstring_placeholders(file_path: Path, dry_run: bool = False) -> int:
    """Fix f-strings without placeholders (F541)."""
    if dry_run:
        print(f"[DRY RUN] Would fix f-string placeholders in {file_path}")
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        fixes = 0

        # Find f-strings without placeholders: f"text" -> "text"
        # Look for f"..." or f'...' that don't contain { }
        pattern = r'f(["\'])([^"\']*?)\1'

        def replace_fstring(match):
            quote = match.group(1)
            text = match.group(2)
            # Only replace if there are no { } placeholders
            if "{" not in text and "}" not in text:
                return f"{quote}{text}{quote}"
            return match.group(0)

        new_content = re.sub(pattern, replace_fstring, content)

        if new_content != content:
            # Count how many were actually changed
            original_fstrings = re.findall(r'f(["\'])([^"\']*?)\1', content)
            new_fstrings = re.findall(r'f(["\'])([^"\']*?)\1', new_content)
            fixes = len(original_fstrings) - len(new_fstrings)

            if fixes > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Fixed {fixes} f-string placeholder issues in {file_path}")
                return fixes

        return 0

    except Exception as e:
        print(f"Error fixing f-strings in {file_path}: {e}")
        return 0


def fix_multiple_statements(file_path: Path, dry_run: bool = False) -> int:
    """Fix multiple statements on one line (E701)."""
    if dry_run:
        print(f"[DRY RUN] Would fix multiple statements in {file_path}")
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        original_lines = lines[:]
        fixes = 0
        new_lines = []

        for line in lines:
            # Look for if statement: something
            if_pattern = r"^(\s*)(if\s+[^:]+):\s*(.+)$"
            match = re.match(if_pattern, line.rstrip())

            if match:
                indent = match.group(1)
                condition = match.group(2)
                statement = match.group(3)

                # Split into two lines
                new_lines.append(f"{indent}{condition}:\n")
                new_lines.append(f"{indent}    {statement}\n")
                fixes += 1
            else:
                new_lines.append(line)

        if fixes > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"Fixed {fixes} multiple statement issues in {file_path}")
            return fixes

        return 0

    except Exception as e:
        print(f"Error fixing multiple statements in {file_path}: {e}")
        return 0


def fix_bare_except(file_path: Path, dry_run: bool = False) -> int:
    """Fix bare except clauses (E722)."""
    if dry_run:
        print(f"[DRY RUN] Would fix bare except in {file_path}")
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Replace bare except: with except Exception:
        content = re.sub(r"\bexcept\s*:", "except Exception:", content)

        if content != original_content:
            fixes = len(re.findall(r"\bexcept\s*:", original_content))
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {fixes} bare except issues in {file_path}")
            return fixes

        return 0

    except Exception as e:
        print(f"Error fixing bare except in {file_path}: {e}")
        return 0


def fix_variable_shadowing(file_path: Path, dry_run: bool = False) -> int:
    """Fix variable shadowing (F402) by renaming variables."""
    if dry_run:
        print(f"[DRY RUN] Would fix variable shadowing in {file_path}")
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        fixes = 0

        # Common case: for field in ...: where field shadows import
        if "for field in" in content and "from " in content and " field" in content:
            # Replace 'for field in' with 'for field_item in'
            content = re.sub(r"for field in", "for field_item in", content)
            # Replace references to 'field' within the loop context
            content = re.sub(r"(\s+)field(\s*[^_\w])", r"\1field_item\2", content)
            fixes += 1

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {fixes} variable shadowing issues in {file_path}")
            return fixes

        return 0

    except Exception as e:
        print(f"Error fixing variable shadowing in {file_path}: {e}")
        return 0


def process_file(file_path: Path, dry_run: bool = False) -> int:
    """Process a single Python file with advanced fixes."""
    print(f"\nProcessing: {file_path}")

    total_fixes = 0
    total_fixes += fix_unused_variables(file_path, dry_run)
    total_fixes += fix_fstring_placeholders(file_path, dry_run)
    total_fixes += fix_multiple_statements(file_path, dry_run)
    total_fixes += fix_bare_except(file_path, dry_run)
    total_fixes += fix_variable_shadowing(file_path, dry_run)

    return total_fixes


def process_directory(path: Path, dry_run: bool = False) -> int:
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
    parser = argparse.ArgumentParser(description="Advanced linting fixer")
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
    print("Starting advanced linting fixes...")

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
    print(f"Total advanced fixes applied: {total_fixes}")


if __name__ == "__main__":
    main()
