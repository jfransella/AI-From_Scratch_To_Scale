# Lint code using flake8
# Usage: .\scripts\lint.ps1 [path]
# If no path is provided, lints the entire project

param(
    [string]$Path = ".",
    [switch]$Fix = $false
)

Write-Host "🔍 Linting Python code..." -ForegroundColor Cyan

# Check if we're in a virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "⚠️  No virtual environment detected. Consider activating one first."
}

# Run flake8
Write-Host "🐍 Running flake8..." -ForegroundColor Yellow
try {
    $flakeOutput = flake8 $Path 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ No linting errors found!" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Linting errors found:" -ForegroundColor Red
        Write-Host $flakeOutput -ForegroundColor Red
        
        if ($Fix) {
            Write-Host "🔧 Auto-fixing what we can..." -ForegroundColor Yellow
            Write-Host "Running format script..." -ForegroundColor Yellow
            & ".\scripts\format.ps1" $Path
            
            Write-Host "Re-running flake8 after formatting..." -ForegroundColor Yellow
            $flakeOutput2 = flake8 $Path 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ All auto-fixable issues resolved!" -ForegroundColor Green
            }
            else {
                Write-Host "⚠️  Some issues remain and need manual fixing:" -ForegroundColor Yellow
                Write-Host $flakeOutput2 -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "💡 Tip: Run with -Fix flag to auto-fix formatting issues" -ForegroundColor Cyan
        }
        exit 1
    }
}
catch {
    Write-Error "❌ Error running flake8: $_"
    exit 1
}

Write-Host "🎉 Linting complete!" -ForegroundColor Green 