#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Bulk markdown linting and fixing script for the AI From Scratch To Scale project.

.DESCRIPTION
    This script automatically fixes markdown linting issues across all markdown files
    in the project. It can be run manually or as part of CI/CD pipelines.

.PARAMETER Fix
    Automatically fix issues that can be fixed automatically.

.PARAMETER CheckOnly
    Only check for issues without fixing them.

.PARAMETER Verbose
    Show detailed output including all linting rules and their status.

.EXAMPLE
    .\fix_markdown.ps1
    # Check and fix all markdown files

.EXAMPLE
    .\fix_markdown.ps1 -CheckOnly
    # Only check for issues without fixing

.EXAMPLE
    .\fix_markdown.ps1 -Verbose
    # Show detailed output while fixing
#>

param(
    [switch]$Fix = $true,
    [switch]$CheckOnly = $false,
    [switch]$Verbose = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Get the script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "🔍 Markdown Linting and Fixing Tool" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Yellow
Write-Host ""

# Check if markdownlint is installed
try {
    $markdownlintVersion = npx markdownlint --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "markdownlint not found"
    }
    Write-Host "✅ markdownlint found: $markdownlintVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ markdownlint not found. Installing..." -ForegroundColor Red
    npm install -g markdownlint-cli
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install markdownlint. Please install it manually:" -ForegroundColor Red
        Write-Host "   npm install -g markdownlint-cli" -ForegroundColor Yellow
        exit 1
    }
}

# Find all markdown files
Write-Host "📁 Scanning for markdown files..." -ForegroundColor Blue
$markdownFiles = Get-ChildItem -Path $ProjectRoot -Recurse -Filter "*.md" | 
Where-Object { 
    $_.FullName -notmatch "\\node_modules\\" -and
    $_.FullName -notmatch "\\.git\\" -and
    $_.FullName -notmatch "\\venv\\" -and
    $_.FullName -notmatch "\\.venv\\" -and
    $_.FullName -notmatch "\\__pycache__\\" -and
    $_.FullName -notmatch "\\.pytest_cache\\" -and
    $_.FullName -notmatch "\\build\\" -and
    $_.FullName -notmatch "\\dist\\" -and
    $_.FullName -notmatch "\\.egg-info\\" -and
    $_.FullName -notmatch "\\.mypy_cache\\" -and
    $_.FullName -notmatch "\\test_outputs\\" -and
    $_.FullName -notmatch "\\outputs\\" -and
    $_.FullName -notmatch "\\wandb\\" -and
    $_.FullName -notmatch "\\mlruns\\"
}

Write-Host "📄 Found $($markdownFiles.Count) markdown files" -ForegroundColor Green

if ($markdownFiles.Count -eq 0) {
    Write-Host "❌ No markdown files found to process." -ForegroundColor Red
    exit 1
}

# Build the markdownlint command
$markdownlintArgs = @(
    "--config", "$ProjectRoot\.markdownlint.json"
)

if ($CheckOnly) {
    $markdownlintArgs += "--list-files"
    Write-Host "🔍 Checking markdown files for issues..." -ForegroundColor Blue
}
else {
    $markdownlintArgs += "--fix"
    Write-Host "🔧 Fixing markdown files..." -ForegroundColor Blue
}

if ($Verbose) {
    $markdownlintArgs += "--verbose"
}

# Add all markdown files to the command
$markdownlintArgs += $markdownFiles.FullName

# Run markdownlint
Write-Host ""
Write-Host "Running: npx markdownlint $($markdownlintArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    $output = & npx markdownlint @markdownlintArgs 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host "✅ All markdown files passed linting!" -ForegroundColor Green
        if (-not $CheckOnly) {
            Write-Host "✅ All auto-fixable issues have been resolved." -ForegroundColor Green
        }
    }
    else {
        Write-Host "⚠️  Some markdown files have issues:" -ForegroundColor Yellow
        Write-Host $output
        Write-Host ""
        Write-Host "💡 To see detailed information about specific rules:" -ForegroundColor Cyan
        Write-Host "   npx markdownlint --help" -ForegroundColor Gray
        Write-Host ""
        Write-Host "💡 To fix issues manually:" -ForegroundColor Cyan
        Write-Host "   npx markdownlint --fix <filename>" -ForegroundColor Gray
    }
    
    exit $exitCode
}
catch {
    Write-Host "❌ Error running markdownlint: $_" -ForegroundColor Red
    exit 1
} 