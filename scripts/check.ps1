# Comprehensive code quality check
# Runs formatting, linting, and tests
# Usage: .\scripts\check.ps1 [path]

param(
    [string]$Path = ".",
    [switch]$Fix = $false,
    [switch]$SkipTests = $false
)

Write-Host "🚀 Running comprehensive code quality checks..." -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

$ErrorCount = 0

# Check if we're in a virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "⚠️  No virtual environment detected. Consider activating one first."
}

# 1. Import organization check
Write-Host "`n📚 1. Checking import organization (isort)..." -ForegroundColor Blue
try {
    isort $Path --diff --check-only --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Imports are properly organized" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Import organization issues found" -ForegroundColor Red
        if ($Fix) {
            Write-Host "🔧 Auto-fixing imports..." -ForegroundColor Yellow
            isort $Path
            Write-Host "✅ Imports fixed" -ForegroundColor Green
        }
        else {
            $ErrorCount++
        }
    }
}
catch {
    Write-Host "❌ Error checking imports: $_" -ForegroundColor Red
    $ErrorCount++
}

# 2. Code formatting check
Write-Host "`n🎨 2. Checking code formatting (black)..." -ForegroundColor Blue
try {
    black $Path --diff --check --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Code is properly formatted" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Code formatting issues found" -ForegroundColor Red
        if ($Fix) {
            Write-Host "🔧 Auto-fixing formatting..." -ForegroundColor Yellow
            black $Path --quiet
            Write-Host "✅ Formatting fixed" -ForegroundColor Green
        }
        else {
            $ErrorCount++
        }
    }
}
catch {
    Write-Host "❌ Error checking formatting: $_" -ForegroundColor Red
    $ErrorCount++
}

# 3. Linting check
Write-Host "`n🔍 3. Checking code quality (flake8)..." -ForegroundColor Blue
try {
    $flakeOutput = flake8 $Path 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ No linting errors found" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Linting issues found:" -ForegroundColor Red
        Write-Host $flakeOutput -ForegroundColor Red
        $ErrorCount++
    }
}
catch {
    Write-Host "❌ Error running linter: $_" -ForegroundColor Red
    $ErrorCount++
}

# 4. Enhanced linting (pylint)
Write-Host "`n🔍 4. Checking code quality (pylint)..." -ForegroundColor Blue
try {
    $pylintAvailable = Get-Command pylint -ErrorAction SilentlyContinue
    if ($pylintAvailable) {
        $pylintOutput = pylint $Path --rcfile=.pylintrc 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Pylint found no issues" -ForegroundColor Green
        }
        else {
            Write-Host "⚠️  Pylint found some issues:" -ForegroundColor Yellow
            # Only show first few lines to avoid spam
            $pylintOutput | Select-Object -First 10 | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
            if ($pylintOutput.Count -gt 10) {
                Write-Host "... (and $($pylintOutput.Count - 10) more issues)" -ForegroundColor Gray
            }
            # Don't count as error since pylint can be very strict
        }
    }
    else {
        Write-Host "⏭️  Pylint not available, skipping enhanced linting" -ForegroundColor Gray
    }
}
catch {
    Write-Host "⚠️  Pylint check skipped: $_" -ForegroundColor Yellow
}

# 5. Type checking (mypy)
Write-Host "`n🏷️  5. Checking types (mypy)..." -ForegroundColor Blue
try {
    $mypyAvailable = Get-Command mypy -ErrorAction SilentlyContinue
    if ($mypyAvailable) {
        $mypyOutput = mypy $Path --config-file=mypy.ini 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ No type errors found" -ForegroundColor Green
        }
        else {
            Write-Host "⚠️  Type checking issues found:" -ForegroundColor Yellow
            # Only show first few lines
            $mypyOutput | Select-Object -First 5 | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
            if ($mypyOutput.Count -gt 5) {
                Write-Host "... (run 'mypy $Path' for full details)" -ForegroundColor Gray
            }
            # Don't count as error since type checking is gradual
        }
    }
    else {
        Write-Host "⏭️  Mypy not installed, skipping type checking" -ForegroundColor Gray
    }
}
catch {
    Write-Host "⚠️  Type checking skipped: $_" -ForegroundColor Yellow
}

# 6. Tests
if (-not $SkipTests) {
    Write-Host "`n🧪 6. Running tests (pytest)..." -ForegroundColor Blue
    try {
        $testsExist = Test-Path "tests" -PathType Container
        if ($testsExist) {
            pytest tests/ --quiet --tb=short
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ All tests passed" -ForegroundColor Green
            }
            else {
                Write-Host "❌ Some tests failed" -ForegroundColor Red
                $ErrorCount++
            }
        }
        else {
            Write-Host "⏭️  No tests directory found, skipping tests" -ForegroundColor Gray
        }
    }
    catch {
        Write-Host "❌ Error running tests: $_" -ForegroundColor Red
        $ErrorCount++
    }
}
else {
    Write-Host "`n🧪 6. Skipping tests (--SkipTests flag used)" -ForegroundColor Gray
}

# Summary
Write-Host "`n" + "=" * 60 -ForegroundColor Gray
if ($ErrorCount -eq 0) {
    Write-Host "🎉 All checks passed! Code is ready for commit." -ForegroundColor Green
    exit 0
}
else {
    Write-Host "❌ $ErrorCount check(s) failed. Please fix the issues above." -ForegroundColor Red
    if (-not $Fix) {
        Write-Host "💡 Tip: Run with -Fix flag to auto-fix formatting issues" -ForegroundColor Cyan
    }
    exit 1
} 