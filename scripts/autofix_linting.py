#!/usr/bin/env python3
"""
Automated linting fixer for the AI-From_Scratch_To_Scale project.

This script automatically fixes common linting issues detected by:
- flake8 (PEP 8 style violations)
- autopep8 (automatic PEP 8 formatting)
- isort (import sorting)
- black (code formatting)

Usage:
    python scripts/autofix_linting.py [--dry-run] [--path PATH] [--aggressive]
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths to process
DEFAULT_PATHS = [
    "models/01_perceptron/src",
    "models/02_adaline/src",
    "models/03_mlp/src",
    "engine",
    "utils",
    "data_utils",
    "plotting",
    "tests",
]


class LintingFixer:
    """Automated linting fixer for Python code."""

    def __init__(
        self, project_root: Path, dry_run: bool = False, aggressive: bool = False
    ):
        self.project_root = project_root
        self.dry_run = dry_run
        self.aggressive = aggressive
        self.fixed_files: Set[str] = set()
        self.errors_fixed: Dict[str, int] = {}

    def run_command(
        self, cmd: List[str], cwd: Path = None
    ) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        if cwd is None:
            cwd = self.project_root

        logger.debug(f"Running: {' '.join(cmd)} in {cwd}")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        try:
            result = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, check=False
            )
            return result
        except Exception as e:
            logger.error(f"Command failed: {' '.join(cmd)} - {e}")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(e))

    def fix_trailing_whitespace(self, file_path: Path) -> int:
        """Fix trailing whitespace issues (W291, W293)."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would fix trailing whitespace in {file_path}")
            return 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            fixes = 0
            new_lines = []

            for line in lines:
                # Remove trailing whitespace but preserve line endings
                if line.rstrip() != line.rstrip(" \t"):
                    fixes += 1
                new_lines.append(
                    line.rstrip() + "\n" if line.endswith("\n") else line.rstrip()
                )

            # Ensure file ends with newline (W292)
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
                fixes += 1

            if fixes > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                logger.info(f"Fixed {fixes} trailing whitespace issues in {file_path}")
                self.fixed_files.add(str(file_path))

            return fixes

        except Exception as e:
            logger.error(f"Error fixing trailing whitespace in {file_path}: {e}")
            return 0

    def fix_blank_lines(self, file_path: Path) -> int:
        """Fix blank line issues (W293 - blank line contains whitespace)."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would fix blank lines in {file_path}")
            return 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Fix blank lines that contain whitespace
            original_content = content
            content = re.sub(r"^[ \t]+$", "", content, flags=re.MULTILINE)

            fixes = len(re.findall(r"^[ \t]+$", original_content, flags=re.MULTILINE))

            if fixes > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(
                    f"Fixed {fixes} blank line whitespace issues in {file_path}"
                )
                self.fixed_files.add(str(file_path))

            return fixes

        except Exception as e:
            logger.error(f"Error fixing blank lines in {file_path}: {e}")
            return 0

    def fix_unused_imports(self, file_path: Path) -> int:
        """Fix unused imports (F401) using autoflake."""
        cmd = [
            sys.executable,
            "-m",
            "autoflake",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--in-place" if not self.dry_run else "--check",
            str(file_path),
        ]

        result = self.run_command(cmd)

        if result.returncode == 0 and not self.dry_run:
            self.fixed_files.add(str(file_path))
            logger.info(f"Fixed unused imports in {file_path}")
            return 1
        elif self.dry_run and "would be reformatted" in result.stdout:
            logger.info(f"[DRY RUN] Would fix unused imports in {file_path}")
            return 1

        return 0

    def sort_imports(self, file_path: Path) -> int:
        """Sort imports using isort."""
        cmd = [
            sys.executable,
            "-m",
            "isort",
            "--check-only" if self.dry_run else "--force-single-line",
            "--line-length",
            "120",
            str(file_path),
        ]

        result = self.run_command(cmd)

        if not self.dry_run and result.returncode != 0:
            # Run without check to actually fix
            cmd = [
                sys.executable,
                "-m",
                "isort",
                "--force-single-line",
                "--line-length",
                "120",
                str(file_path),
            ]
            fix_result = self.run_command(cmd)
            if fix_result.returncode == 0:
                self.fixed_files.add(str(file_path))
                logger.info(f"Sorted imports in {file_path}")
                return 1
        elif self.dry_run and result.returncode != 0:
            logger.info(f"[DRY RUN] Would sort imports in {file_path}")
            return 1

        return 0

    def format_with_autopep8(self, file_path: Path) -> int:
        """Format code with autopep8 for basic PEP 8 compliance."""
        aggressiveness = "--aggressive" if self.aggressive else ""
        cmd = [
            sys.executable,
            "-m",
            "autopep8",
            "--in-place" if not self.dry_run else "--diff",
            "--max-line-length",
            "120",
            aggressiveness,
            str(file_path),
        ]

        # Remove empty strings from cmd
        cmd = [arg for arg in cmd if arg]

        result = self.run_command(cmd)

        if result.returncode == 0:
            if self.dry_run and result.stdout.strip():
                logger.info(f"[DRY RUN] Would format {file_path} with autopep8")
                return 1
            elif not self.dry_run:
                self.fixed_files.add(str(file_path))
                logger.info(f"Formatted {file_path} with autopep8")
                return 1

        return 0

    def fix_specific_issues(self, file_path: Path) -> int:
        """Fix specific common issues found in the codebase."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would fix specific issues in {file_path}")
            return 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            fixes = 0

            # Fix bare except (E722)
            if "except:" in content:
                content = re.sub(r"\bexcept\s*:", "except Exception:", content)
                fixes += content.count("except Exception:") - original_content.count(
                    "except Exception:"
                )

            # Fix comparison to True (E712)
            true_comparisons = re.findall(r"== True\b", content)
            if true_comparisons:
                content = re.sub(r"== True\b", "is True", content)
                fixes += len(true_comparisons)

            false_comparisons = re.findall(r"== False\b", content)
            if false_comparisons:
                content = re.sub(r"== False\b", "is False", content)
                fixes += len(false_comparisons)

            if fixes > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Fixed {fixes} specific issues in {file_path}")
                self.fixed_files.add(str(file_path))

            return fixes

        except Exception as e:
            logger.error(f"Error fixing specific issues in {file_path}: {e}")
            return 0

    def process_file(self, file_path: Path) -> Dict[str, int]:
        """Process a single Python file."""
        logger.info(f"Processing {file_path}")

        fixes = {
            "trailing_whitespace": self.fix_trailing_whitespace(file_path),
            "blank_lines": self.fix_blank_lines(file_path),
            "unused_imports": self.fix_unused_imports(file_path),
            "import_sorting": self.sort_imports(file_path),
            "autopep8_formatting": self.format_with_autopep8(file_path),
            "specific_issues": self.fix_specific_issues(file_path),
        }

        return fixes

    def process_directory(self, path: Path) -> None:
        """Process all Python files in a directory."""
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return

        if path.is_file() and path.suffix == ".py":
            fixes = self.process_file(path)
            for fix_type, count in fixes.items():
                self.errors_fixed[fix_type] = self.errors_fixed.get(fix_type, 0) + count
            return

        # Process directory recursively
        for py_file in path.rglob("*.py"):
            # Skip __pycache__ and .venv directories
            if "__pycache__" in str(py_file) or ".venv" in str(py_file):
                continue

            fixes = self.process_file(py_file)
            for fix_type, count in fixes.items():
                self.errors_fixed[fix_type] = self.errors_fixed.get(fix_type, 0) + count

    def install_dependencies(self) -> bool:
        """Install required linting tools."""
        tools = ["autopep8", "isort", "autoflake", "flake8"]

        for tool in tools:
            logger.info(f"Installing {tool}...")
            result = self.run_command([sys.executable, "-m", "pip", "install", tool])
            if result.returncode != 0:
                logger.error(f"Failed to install {tool}: {result.stderr}")
                return False

        logger.info("All linting tools installed successfully")
        return True

    def run_final_check(self, paths: List[Path]) -> None:
        """Run final flake8 check to see remaining issues."""
        logger.info("Running final flake8 check...")

        all_paths = []
        for path in paths:
            if path.exists():
                all_paths.append(str(path))

        if not all_paths:
            logger.warning("No valid paths found for final check")
            return

        cmd = [
            sys.executable,
            "-m",
            "flake8",
            "--statistics",
            "--count",
            "--max-line-length=120",
            "--extend-ignore=E203,E501,W503",  # Common ignores
        ] + all_paths

        result = self.run_command(cmd)

        if result.returncode == 0:
            logger.info("✅ No flake8 issues remaining!")
        else:
            logger.info("Remaining flake8 issues:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)

    def print_summary(self) -> None:
        """Print a summary of fixes applied."""
        logger.info("\n" + "=" * 60)
        logger.info("LINTING FIX SUMMARY")
        logger.info("=" * 60)

        if self.dry_run:
            logger.info("DRY RUN - No files were actually modified")

        logger.info(f"Files processed: {len(self.fixed_files)}")

        total_fixes = sum(self.errors_fixed.values())
        logger.info(f"Total fixes applied: {total_fixes}")

        if self.errors_fixed:
            logger.info("\nFixes by type:")
            for fix_type, count in sorted(self.errors_fixed.items()):
                if count > 0:
                    logger.info(f"  {fix_type}: {count}")

        if self.fixed_files:
            logger.info(f"\nFiles modified ({len(self.fixed_files)}):")
            for file_path in sorted(self.fixed_files):
                logger.info(f"  {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Automated linting fixer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument(
        "--path", type=str, help="Specific path to process (default: all project paths)"
    )
    parser.add_argument(
        "--aggressive", action="store_true", help="Use aggressive autopep8 formatting"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required linting dependencies",
    )
    parser.add_argument(
        "--final-check", action="store_true", help="Run final flake8 check after fixes"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    logger.info(f"Project root: {project_root}")

    fixer = LintingFixer(project_root, dry_run=args.dry_run, aggressive=args.aggressive)

    # Install dependencies if requested
    if args.install_deps:
        if not fixer.install_dependencies():
            sys.exit(1)
        return

    # Determine paths to process
    if args.path:
        paths = [project_root / args.path]
    else:
        paths = [project_root / path for path in DEFAULT_PATHS]

    # Process all paths
    logger.info("Starting automated linting fixes...")

    for path in paths:
        logger.info(f"\nProcessing path: {path}")
        fixer.process_directory(path)

    # Print summary
    fixer.print_summary()

    # Run final check if requested
    if args.final_check and not args.dry_run:
        fixer.run_final_check(paths)


if __name__ == "__main__":
    main()
