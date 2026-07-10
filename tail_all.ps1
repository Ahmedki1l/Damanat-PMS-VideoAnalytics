#Requires -Version 5.1
<#
  tail_all.ps1 - live-follow every VA group log at once, in ONE window, with
  each line prefixed by its group so the streams are distinguishable.

  USAGE
    .\tail_all.ps1              # follow *.out.log (default)
    .\tail_all.ps1 -All         # follow both .out.log and .err.log
    .\tail_all.ps1 -Err         # follow only .err.log
    Ctrl+C to stop (does NOT stop the pipeline; use .\stop_all.ps1 for that).
#>
param(
    [switch]$All,   # follow .out.log AND .err.log
    [switch]$Err    # follow only .err.log
)

$Root   = $PSScriptRoot
$LogDir = Join-Path $Root 'logs'
if (-not (Test-Path $LogDir)) { Write-Host "No logs\ dir yet - start run_all.ps1 first."; return }

# Which files to follow.
if ($All)      { $pattern = '*.log' }
elseif ($Err)  { $pattern = '*.err.log' }
else           { $pattern = '*.out.log' }

$files = Get-ChildItem (Join-Path $LogDir $pattern) -ErrorAction SilentlyContinue
if (-not $files) { Write-Host "No logs matching '$pattern' in $LogDir."; return }

# Colour each group consistently so the merged stream is easy to read.
$palette = 'Cyan','Green','Yellow','Magenta','White','Blue','Red','DarkCyan'
$colour  = @{}
$i = 0
foreach ($f in $files) {
    $tag = $f.BaseName        # e.g. va_b1-gate.out
    $colour[$f.FullName] = $palette[$i % $palette.Count]
    $i++
}

# Start each read at end-of-file so we only show NEW lines (like tail -f).
$pos = @{}
foreach ($f in $files) { $pos[$f.FullName] = $f.Length }

Write-Host "[tail_all] following $($files.Count) log(s) matching '$pattern'. Ctrl+C to stop.`n"

while ($true) {
    foreach ($f in Get-ChildItem (Join-Path $LogDir $pattern) -ErrorAction SilentlyContinue) {
        $path = $f.FullName
        $last = if ($pos.ContainsKey($path)) { $pos[$path] } else { 0 }
        $len  = $f.Length
        if ($len -lt $last) { $last = 0 }   # file was truncated/rotated
        if ($len -gt $last) {
            $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
            try {
                [void]$fs.Seek($last, 'Begin')
                $sr = New-Object System.IO.StreamReader($fs)
                $chunk = $sr.ReadToEnd()
                $sr.Dispose()
            } finally { $fs.Dispose() }

            $tag = $f.BaseName -replace '^va_', '' -replace '\.(out|err)$', ''
            $c   = if ($colour.ContainsKey($path)) { $colour[$path] } else { 'Gray' }
            foreach ($line in ($chunk -split "`r?`n")) {
                if ($line -ne '') { Write-Host ("[{0}] {1}" -f $tag, $line) -ForegroundColor $c }
            }
            $pos[$path] = $len
        }
    }
    Start-Sleep -Milliseconds 400
}
