# Lint code using pylint
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

# Run pylint
Write-Host "🐍 Running pylint..." -ForegroundColor Yellow
try {
    $pylintOutput = pylint $Path 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ No linting errors found!" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Linting errors found:" -ForegroundColor Red
        Write-Host $pylintOutput -ForegroundColor Red
        
        if ($Fix) {
            Write-Host "🔧 Auto-fixing what we can..." -ForegroundColor Yellow
            Write-Host "Running format script..." -ForegroundColor Yellow
            & ".\scripts\format.ps1" $Path
            
            Write-Host "Re-running pylint after formatting..." -ForegroundColor Yellow
            $pylintOutput2 = pylint $Path 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ All auto-fixable issues resolved!" -ForegroundColor Green
            }
            else {
                Write-Host "⚠️  Some issues remain and need manual fixing:" -ForegroundColor Yellow
                Write-Host $pylintOutput2 -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "💡 Tip: Run with -Fix flag to auto-fix formatting issues" -ForegroundColor Cyan
        }
        exit 1
    }
}
catch {
    Write-Error "❌ Error running pylint: $_"
    exit 1
}

Write-Host "🎉 Linting complete!" -ForegroundColor Green 