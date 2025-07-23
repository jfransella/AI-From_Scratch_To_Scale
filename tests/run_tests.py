#!/usr/bin/env python3
"""
Test runner for AI From Scratch to Scale project.

This script runs the complete test suite and provides a summary of results.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run the complete test suite."""
    # Get the project root
    project_root = Path(__file__).parent.parent

    # Change to project root
    os.chdir(project_root)

    print("🧪 Running AI From Scratch to Scale Test Suite")
    print("=" * 60)

    # Test categories
    test_categories = [
        ("Smoke Tests", "tests/smoke/"),
        ("Unit Tests", "tests/unit/"),
        ("Integration Tests", "tests/integration/"),
    ]

    results = {}

    for category, test_path in test_categories:
        if Path(test_path).exists():
            print(f"\n📋 Running {category}...")
            try:
                # Run pytest for this category
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        test_path,
                        "-v",
                        "--tb=short",
                        "--color=yes",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=300,  # 5 minute timeout
                )

                results[category] = {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

                if result.returncode == 0:
                    print(f"✅ {category} passed")
                else:
                    print(f"❌ {category} failed")
                    print(result.stdout)
                    if result.stderr:
                        print("Errors:")
                        print(result.stderr)

            except subprocess.TimeoutExpired:
                print(f"⏰ {category} timed out")
                results[category] = {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "Test timed out",
                }
            except Exception as e:
                print(f"💥 {category} failed to run: {e}")
                results[category] = {"returncode": -1, "stdout": "", "stderr": str(e)}
        else:
            print(f"⚠️  {category} directory not found: {test_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for category, result in results.items():
        if result["returncode"] == 0:
            print(f"✅ {category}: PASSED")
            passed_tests += 1
        else:
            print(f"❌ {category}: FAILED")
            failed_tests += 1
        total_tests += 1

    print("\n📈 Overall Results:")
    print(f"   Total Categories: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")

    if failed_tests == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed_tests} test category(ies) failed")
        return 1


def run_specific_test(test_path):
    """Run a specific test file."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print(f"🧪 Running specific test: {test_path}")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                test_path,
                "-v",
                "--tb=long",
                "--color=yes",
            ],
            check=False,
            timeout=300,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return -1
    except Exception as e:
        print(f"💥 Test failed to run: {e}")
        return -1


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Run specific test
        test_path = sys.argv[1]
        return run_specific_test(test_path)
    else:
        # Run all tests
        return run_tests()


if __name__ == "__main__":
    sys.exit(main())
