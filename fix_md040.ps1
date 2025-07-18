#!/usr/bin/env pwsh

# Fix MD040: Add language specification to code blocks
$files = Get-ChildItem -Recurse -Filter "*.md" | 
Where-Object { 
    $_.FullName -notmatch "node_modules" -and 
    $_.FullName -notmatch "venv" -and 
    $_.FullName -notmatch "\.venv"
}

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    
    # Fix MD040: Add language specification to code blocks
    $content = $content -replace '```\s*\n', '```text`n'
    $content = $content -replace '```\s*\r\n', '```text`r`n'
    
    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Fixed MD040 in: $($file.Name)" -ForegroundColor Green
    }
}

Write-Host "MD040 fixes completed!" -ForegroundColor Green 