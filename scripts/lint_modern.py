#!/usr/bin/env python3
"""
Modern linting and fixing script for AI-From_Scratch_To_Scale project.

This script provides comprehensive linting using pylint + mypy + black + isort,
with automatic fixing capabilities for common issues.

Usage:
    python scripts/lint_modern.py [--fix] [--path PATH] [--tool TOOL]

Examples:
    python scripts/lint_modern.py                    # Check all files
    python scripts/lint_modern.py --fix              # Fix all auto-fixable issues
    python scripts/lint_modern.py --path src/        # Check specific path
    python scripts/lint_modern.py --tool pylint      # Run only pylint
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class LintingTool:
    """Base class for linting tools."""

    def __init__(self, name: str, command: List[str], can_fix: bool = False):
        self.name = name
        self.command = command
        self.can_fix = can_fix

    def check(self, paths: List[Path]) -> tuple[bool, str]:
        """Run the linting tool and return (success, output)."""
        cmd = self.command + [str(p) for p in paths]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = stdout + stderr
            
            # Special handling for mypy package name issues
            if self.name == "Mypy" and "is not a valid Python package name" in output:
                # Filter out the package name error and check if there are other issues
                lines = output.split('\n')
                filtered_lines = [line for line in lines if "is not a valid Python package name" not in line]
                filtered_output = '\n'.join(filtered_lines).strip()
                
                # If only package name error, consider it successful
                if not filtered_output:
                    return True, ""
                else:
                    return result.returncode == 0, filtered_output
            
            return result.returncode == 0, output
        except FileNotFoundError:
            return (
                False,
                f"❌ {self.name} not found. Install with: pip install {self.name.lower()}",
            )


class ModernLinter:
    """Modern linting system using pylint + mypy + black + isort."""

    def __init__(self):
        self.tools = {
            "black": LintingTool(
                "Black",
                [sys.executable, "-m", "black", "--check", "--diff"],
                can_fix=True,
            ),
            "isort": LintingTool(
                "isort",
                [sys.executable, "-m", "isort", "--check-only", "--diff"],
                can_fix=True,
            ),
            "pylint": LintingTool(
                "Pylint", [sys.executable, "-m", "pylint", "--score=no"]
            ),
            "mypy": LintingTool(
                "Mypy", [sys.executable, "-m", "mypy", "--no-error-summary", "--ignore-missing-imports", "--disable-error-code=misc"]
            ),
        }

    def get_python_files(self, path: Path) -> List[Path]:
        """Get all Python files in the given path."""
        if path.is_file() and path.suffix == ".py":
            return [path]
        elif path.is_dir():
            return list(path.rglob("*.py"))
        else:
            return []

    def fix_formatting(self, paths: List[Path]) -> int:
        """Fix auto-fixable formatting issues."""
        fixes_applied = 0

        print("🔧 Applying automatic fixes...")

        # 1. Remove unused imports with autoflake
        print("  📦 Removing unused imports...")
        cmd = [
            sys.executable,
            "-m",
            "autoflake",
            "--remove-all-unused-imports",
            "--in-place",
        ] + [str(p) for p in paths]
        try:
            result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                print("     ✅ Removed unused imports")
                fixes_applied += 1
        except FileNotFoundError:
            print("     ⚠️  autoflake not found, skipping unused import removal")

        # 2. Sort imports with isort
        print("  📋 Sorting imports...")
        cmd = [sys.executable, "-m", "isort"] + [str(p) for p in paths]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print("     ✅ Sorted imports")
            fixes_applied += 1

        # 3. Format code with black
        print("  🎨 Formatting code...")
        cmd = [sys.executable, "-m", "black"] + [str(p) for p in paths]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print("     ✅ Formatted code")
            fixes_applied += 1

        return fixes_applied

    def run_linting(
        self, paths: List[Path], tools: Optional[List[str]] = None, fix: bool = False
    ) -> int:
        """Run linting tools on the given paths."""
        if not paths:
            print("❌ No Python files found to lint")
            return 1

        print(f"🔍 Linting {len(paths)} Python files...")

        # Apply fixes first if requested
        if fix:
            fixes_applied = self.fix_formatting(paths)
            print(f"✅ Applied {fixes_applied} automatic fixes\n")

        # Determine which tools to run
        tools_to_run = tools or ["black", "isort", "pylint", "mypy"]

        overall_success = True
        results = {}

        # Run each linting tool
        for tool_name in tools_to_run:
            if tool_name not in self.tools:
                print(f"❌ Unknown tool: {tool_name}")
                continue

            tool = self.tools[tool_name]
            print(f"🔍 Running {tool.name}...")

            success, output = tool.check(paths)
            results[tool_name] = (success, output)

            if success:
                print(f"   ✅ {tool.name}: No issues found")
            else:
                print(f"   ❌ {tool.name}: Issues found")
                if output.strip():
                    print(f"      {output[:200]}{'...' if len(output) > 200 else ''}")
                overall_success = False

        # Summary
        print("\n" + "=" * 60)
        print("LINTING SUMMARY")
        print("=" * 60)

        for tool_name, (success, output) in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{tool_name:>10}: {status}")

        if overall_success:
            print("\n🎉 All linting checks passed!")
            return 0
        else:
            print("\n💡 Some issues found. Use --fix to apply automatic fixes.")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Modern linting for AI-From_Scratch_To_Scale"
    )
    parser.add_argument(
        "--path", default=".", help="Path to lint (default: current directory)"
    )
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes")
    parser.add_argument(
        "--tool",
        choices=["black", "isort", "pylint", "mypy"],
        help="Run only specific tool",
    )

    args = parser.parse_args()

    # Get files to lint
    path = Path(args.path)
    linter = ModernLinter()
    files = linter.get_python_files(path)

    # Filter out common directories to skip
    skip_dirs = {
        ".venv", "__pycache__", ".git", "node_modules",  # Common dev directories
        ".mypy_cache", ".pytest_cache", ".coverage",     # Cache directories
        ".vscode", ".idea",                              # IDE directories
        "outputs", "logs", "tmp", "temp",                # Output directories
        "dist", "build", "*.egg-info"                    # Build directories
    }
    files = [f for f in files if not any(skip_dir in f.parts for skip_dir in skip_dirs)]

    if not files:
        print(f"❌ No Python files found in {path}")
        return 1

    # Run linting
    tools = [args.tool] if args.tool else None
    return linter.run_linting(files, tools, args.fix)


if __name__ == "__main__":
    sys.exit(main())
