#Requires -Version 5.1
<#
  stop_all.ps1 — stop every VA process launched by run_all.ps1.
  Reads the PIDs recorded in run_all.pids and terminates them.
#>

$ErrorActionPreference = 'SilentlyContinue'
$Root = $PSScriptRoot
$pidFile = Join-Path $Root 'run_all.pids'

if (-not (Test-Path $pidFile)) {
    Write-Host "[stop_all] no run_all.pids found — nothing to stop."
    return
}

Get-Content $pidFile | ForEach-Object {
    $procId = $_.Trim()
    if ($procId) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($p) {
            Write-Host "[stop_all] stopping PID $procId ($($p.ProcessName))"
            Stop-Process -Id $procId -Force
        }
    }
}
Remove-Item $pidFile -ErrorAction SilentlyContinue
Write-Host "[stop_all] done."
