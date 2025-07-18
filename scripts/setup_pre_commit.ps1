#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup script for pre-commit markdown linting hooks.

.DESCRIPTION
    This script installs and configures pre-commit hooks for markdown linting.
    It creates the necessary hook files and makes them executable.

.PARAMETER Force
    Force reinstallation of hooks even if they already exist.

.EXAMPLE
    .\setup_pre_commit.ps1
    # Install pre-commit hooks

.EXAMPLE
    .\setup_pre_commit.ps1 -Force
    # Force reinstall pre-commit hooks
#>

param(
    [switch]$Force = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🔧 Setting up pre-commit markdown linting hooks..." -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Get the script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$HooksDir = Join-Path $ProjectRoot ".git" "hooks"

Write-Host "Project Root: $ProjectRoot" -ForegroundColor Yellow
Write-Host "Hooks Directory: $HooksDir" -ForegroundColor Yellow
Write-Host ""

# Check if we're in a git repository
if (-not (Test-Path (Join-Path $ProjectRoot ".git"))) {
    Write-Host "❌ Error: Not in a git repository. Please run this from the project root." -ForegroundColor Red
    exit 1
}

# Check if hooks directory exists
if (-not (Test-Path $HooksDir)) {
    Write-Host "❌ Error: Git hooks directory not found. Please ensure this is a valid git repository." -ForegroundColor Red
    exit 1
}

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
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install markdownlint. Please install it manually:" -ForegroundColor Red
        Write-Host "   npm install" -ForegroundColor Yellow
        exit 1
    }
}

# Define hook files to install
$Hooks = @{
    "pre-commit" = @{
        Source      = "pre-commit.ps1"
        Description = "Markdown linting before commit"
    }
}

# Install each hook
foreach ($HookName in $Hooks.Keys) {
    $HookPath = Join-Path $HooksDir $HookName
    $SourcePath = Join-Path $ScriptDir $Hooks[$HookName].Source
    
    Write-Host "Installing $HookName hook..." -ForegroundColor Blue
    
    # Check if hook already exists
    if (Test-Path $HookPath) {
        if ($Force) {
            Write-Host "  - Replacing existing hook..." -ForegroundColor Yellow
        }
        else {
            Write-Host "  - Hook already exists. Use -Force to replace." -ForegroundColor Yellow
            continue
        }
    }
    
    # Copy the hook file
    try {
        Copy-Item -Path $SourcePath -Destination $HookPath -Force
        Write-Host "  ✅ Hook installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "  ❌ Failed to install hook: $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Pre-commit hooks setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 What happens now:" -ForegroundColor Cyan
Write-Host "  - Every time you commit, markdown files will be automatically linted" -ForegroundColor Gray
Write-Host "  - Auto-fixable issues will be fixed and staged automatically" -ForegroundColor Gray
Write-Host "  - Non-fixable issues will prevent the commit until resolved" -ForegroundColor Gray
Write-Host ""
Write-Host "🧪 Test the setup:" -ForegroundColor Cyan
Write-Host "  1. Make a change to any .md file" -ForegroundColor Gray
Write-Host "  2. Stage it: git add <filename>" -ForegroundColor Gray
Write-Host "  3. Try to commit: git commit -m 'test'" -ForegroundColor Gray
Write-Host "  4. The hook will run automatically" -ForegroundColor Gray
Write-Host ""
Write-Host "🔧 Manual testing:" -ForegroundColor Cyan
Write-Host "  .\scripts\fix_markdown.ps1 -CheckOnly" -ForegroundColor Gray
Write-Host ""
Write-Host "📚 For more information, see: docs/Markdown_Linting_Guide.md" -ForegroundColor Cyan 