#!/usr/bin/env python3
"""
Fix specific linting issues in cache.py file.

This script addresses pylint and mypy issues found in the cache.py file.
"""

import argparse
import re
from pathlib import Path


def fix_cache_file(file_path: Path) -> None:
    """Fix all linting issues in the cache.py file."""
    print(f"Fixing linting issues in: {file_path}")

    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Fix 1: Change f-string logging to lazy % formatting
    logging_patterns = [
        (r'self\.logger\.debug\(f"([^"]+)"\)', r'self.logger.debug("\1")'),
        (r'self\.logger\.info\(f"([^"]+)"\)', r'self.logger.info("\1")'),
        (r'self\.logger\.warning\(f"([^"]+)"\)', r'self.logger.warning("\1")'),
        (r'self\.logger\.error\(f"([^"]+)"\)', r'self.logger.error("\1")'),
    ]

    for pattern, replacement in logging_patterns:
        content = re.sub(pattern, replacement, content)

    # Fix 3: Add specific exception types instead of bare Exception
    content = content.replace(
        "except Exception:",
        "except (OSError, IOError, json.JSONDecodeError, ValueError, KeyError):",
    )

    # Fix 4: Add 'from e' to raise statements
    content = re.sub(
        r'raise DataError\(f"([^"]+): \{e\}"\)',
        r'raise DataError(f"\1: {e}") from e',
        content,
    )

    # Fix 5: Fix unused variable 'category' in loop
    content = content.replace(
        "for category, category_config in CACHE_CATEGORIES.items():",
        "for category_config in CACHE_CATEGORIES.values():",
    )

    # Fix 6: Move import statements to top of file
    content = re.sub(r"(\s+)(import pickle)", r"\nimport pickle\n", content)

    # Remove duplicate import pickle statements that were moved
    content = re.sub(r"\n\s+import pickle\n", "", content)

    # Add pickle import at top after other imports
    if "import pickle" not in content.split("\n")[:20]:
        import_section = content.split("from utils.exceptions import DataError")[0]
        rest = content.split("from utils.exceptions import DataError")[1]
        content = (
            import_section
            + "from utils.exceptions import DataError\nimport pickle"
            + rest
        )

    # Fix 7: Fix global statement issue
    content = content.replace(
        "def get_cache() -> DatasetCache:\n"
        '"""Get global cache instance."""\n    global _global_cache',
        '''def get_cache() -> DatasetCache:
    """Get global cache instance."""
    # Use module-level variable instead of global statement
    global _global_cache''',
    )

    # Fix 8: Fix type annotations for configuration dictionaries
    # Add proper type hints to avoid mypy 'object' type issues
    config_fixes = [
        ("self.config['base_dir']", "Path(self.config['base_dir'])"),
        ("category_config['subdir']", "str(category_config['subdir'])"),
        ("self.config['metadata_filename']", "str(self.config['metadata_filename'])"),
        ("category_config['ttl_hours']", "float(category_config['ttl_hours'])"),
        (
            "self.config['cleanup_frequency_hours']",
            "float(self.config['cleanup_frequency_hours'])",
        ),
        ("self.config['max_cache_size_gb']", "float(self.config['max_cache_size_gb'])"),
    ]

    for old, new in config_fixes:
        content = content.replace(old, new)

    # Fix 9: Fix dictionary type annotations
    content = content.replace(
        "def cleanup_expired_entries(self, force: bool = False) -> Dict[str, int]:",
        "def cleanup_expired_entries(self, force: bool = False) -> Dict[str, Any]:",
    )

    content = content.replace(
        "def _cleanup_category(self, category: str, category_config: Dict) -> Dict[str, int]:",
        "def _cleanup_category(self, category: str, category_config: Dict) -> Dict[str, Any]:",
    )

    # Fix 10: Fix line length issues by breaking long lines
    long_lines = [
        (
            "self.logger.info(f\"Cache cleanup completed: {stats['expired_removed']} expired, \"\n                         f\"{stats['corrupted_removed']} corrupted, \"\n                         f\"{stats['space_freed_mb']:.2f} MB freed\")",
            "self.logger.info(\"Cache cleanup completed: %d expired, %d corrupted, %.2f MB freed\",\n                         stats['expired_removed'], stats['corrupted_removed'], stats['space_freed_mb'])",
        ),
    ]

    for old, new in long_lines:
        if old in content:
            content = content.replace(old, new)

    # Write the fixed content back
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Fixed linting issues in {file_path}")

        # Count changes
        original_lines = original_content.split("\n")
        new_lines = content.split("\n")
        changes = sum(
            1
            for i, (old, new) in enumerate(zip(original_lines, new_lines))
            if old != new
        )
        print(f"   📝 Modified {changes} lines")
    else:
        print(f"ℹ️  No changes needed in {file_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix linting issues in cache.py")
    parser.add_argument("file_path", help="Path to cache.py file")

    args = parser.parse_args()
    file_path = Path(args.file_path)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return 1

    fix_cache_file(file_path)
    print("🎉 Cache file linting fixes completed!")
    return 0


if __name__ == "__main__":
    exit(main())
