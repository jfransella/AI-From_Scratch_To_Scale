# Format code using black and isort
# Usage: .\scripts\format.ps1 [path]
# If no path is provided, formats the entire project

param(
    [string]$Path = "."
)

Write-Host "🎨 Formatting Python code..." -ForegroundColor Cyan

# Check if we're in a virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "⚠️  No virtual environment detected. Consider activating one first."
}

# Run isort first (organize imports)
Write-Host "📝 Organizing imports with isort..." -ForegroundColor Yellow
try {
    isort $Path --diff --check-only
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Imports are already organized" -ForegroundColor Green
    }
    else {
        Write-Host "🔧 Organizing imports..." -ForegroundColor Yellow
        isort $Path
        Write-Host "✅ Imports organized" -ForegroundColor Green
    }
}
catch {
    Write-Error "❌ Error running isort: $_"
    exit 1
}

# Run black (code formatting)
Write-Host "🖤 Formatting code with black..." -ForegroundColor Yellow
try {
    black $Path --diff --check
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Code is already formatted" -ForegroundColor Green
    }
    else {
        Write-Host "🔧 Formatting code..." -ForegroundColor Yellow
        black $Path
        Write-Host "✅ Code formatted" -ForegroundColor Green
    }
}
catch {
    Write-Error "❌ Error running black: $_"
    exit 1
}

Write-Host "🎉 Formatting complete!" -ForegroundColor Green 