#!/usr/bin/env pwsh

# Fix remaining markdownlint errors
$files = Get-ChildItem -Recurse -Filter "*.md" | 
Where-Object { 
    $_.FullName -notmatch "node_modules" -and 
    $_.FullName -notmatch "venv" -and 
    $_.FullName -notmatch "\.venv"
}

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    
    # Fix MD032: Add blank lines around lists
    $content = $content -replace '(\n)(\d+\.\s)', '$1$2'
    $content = $content -replace '(\n)(\*\s)', '$1$2'
    $content = $content -replace '(\d+\.\s.*?)(\n\d+\.)', '$1$2'
    $content = $content -replace '(\*\s.*?)(\n\*)', '$1$2'
    
    # Fix MD031: Add blank lines around fenced code blocks
    $content = $content -replace '(\n)(```)', '$1$2'
    $content = $content -replace '(```\n)(\n)', '$1$2'
    
    # Fix MD029: Fix ordered list numbering
    $lines = $content -split "`n"
    $newLines = @()
    $listCounter = 1
    
    foreach ($line in $lines) {
        if ($line -match '^\d+\.\s') {
            $newLines += $line -replace '^\d+\.\s', "$listCounter. "
            $listCounter++
        }
        else {
            $newLines += $line
            $listCounter = 1
        }
    }
    
    $content = $newLines -join "`n"
    
    # Fix MD024: Make duplicate headings unique
    $headingCounts = @{}
    $lines = $content -split "`n"
    $newLines = @()
    
    foreach ($line in $lines) {
        if ($line -match '^(#{1,6})\s+(.+)$') {
            $headingText = $matches[2].Trim()
            if ($headingCounts.ContainsKey($headingText)) {
                $headingCounts[$headingText]++
                $newLines += $line -replace '^(#{1,6})\s+(.+)$', "`$1 `$2 ($($headingCounts[$headingText]))"
            }
            else {
                $headingCounts[$headingText] = 1
                $newLines += $line
            }
        }
        else {
            $newLines += $line
        }
    }
    
    $content = $newLines -join "`n"
    
    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Fixed remaining issues in: $($file.Name)" -ForegroundColor Green
    }
}

Write-Host "Remaining fixes completed!" -ForegroundColor Green 