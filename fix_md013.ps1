#!/usr/bin/env pwsh

# Fix MD013: Line length violations by breaking long lines
$files = Get-ChildItem -Recurse -Filter "*.md" | 
Where-Object { 
    $_.FullName -notmatch "node_modules" -and 
    $_.FullName -notmatch "venv" -and 
    $_.FullName -notmatch "\.venv"
}

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    
    # Split into lines
    $lines = $content -split "`n"
    $newLines = @()
    
    foreach ($line in $lines) {
        if ($line.Length -gt 120 -and $line.Trim() -ne "") {
            # Don't break code blocks, URLs, or already broken lines
            if ($line -notmatch '^```' -and $line -notmatch '^    ' -and $line -notmatch '^#') {
                # Try to break at natural points
                $words = $line -split " "
                $currentLine = ""
                $brokenLine = ""
                
                foreach ($word in $words) {
                    if (($currentLine + " " + $word).Length -gt 120) {
                        if ($currentLine -ne "") {
                            $brokenLine += $currentLine.Trim() + "`n"
                            $currentLine = $word
                        }
                        else {
                            # Single word is too long, break it
                            $brokenLine += $word + "`n"
                        }
                    }
                    else {
                        $currentLine += " " + $word
                    }
                }
                
                if ($currentLine -ne "") {
                    $brokenLine += $currentLine.Trim()
                }
                
                $newLines += $brokenLine
            }
            else {
                $newLines += $line
            }
        }
        else {
            $newLines += $line
        }
    }
    
    $newContent = $newLines -join "`n"
    
    if ($newContent -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $newContent -NoNewline
        Write-Host "Fixed MD013 in: $($file.Name)" -ForegroundColor Green
    }
}

Write-Host "MD013 fixes completed!" -ForegroundColor Green 