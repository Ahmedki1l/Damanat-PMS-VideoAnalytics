#Requires -Version 5.1
<#
  run_all.ps1 - launch the Video Analytics pipeline as parallel per-process
  camera groups. This is the supported scaling lever (it replaces the reverted
  in-process threading). Each process is ONE serial inference worker, so
  fps/camera scales as that budget is split over fewer cameras. Keep <= 6
  cameras per group to approach ~5 fps/camera on a high-core box.

  TOPOLOGY RULES (read before editing $Groups)
    * EXACTLY ONE group runs --api. That process receives the ANPR webhooks
      and binds plates in its OWN in-memory registry, so it MUST own the
      entrance / Park_Entry camera. Set Api=$true on that group only.
    * All groups share the same database and the on-disk gallery, so slot
      status, parking sessions and per-plate gallery folders are visible
      across every process.
    * Cameras in DIFFERENT groups do NOT share live in-memory track state;
      cross-process identity flows only through the DB + gallery. Keep cameras
      that must hand identity off live (typically the same floor) in the SAME
      group.

  USAGE
    .\run_all.ps1              # start all groups
    Get-Content -Wait .\logs\va_b1-gate.out.log     # follow a group's log
    .\stop_all.ps1             # stop everything started by this script
#>

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

# Prefer the venv's WINDOWLESS python (pythonw.exe): it has no console, so
# neither it nor the child processes it spawns pop up console windows - that is
# what was opening "tons of PowerShell/console windows". stdout/stderr are
# redirected to log files below, so losing the console costs nothing. Fall back
# to python.exe, then to whatever 'python' resolves to.
$Python = Join-Path $Root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $Python)) { $Python = Join-Path $Root '.venv\Scripts\python.exe' }
if (-not (Test-Path $Python)) { $Python = 'python' }

# Total logical cores, used to hand each group a slice of the CPU thread budget
# so the parallel processes PARTITION the cores instead of each grabbing all of
# them (5 processes x all cores = oversubscription = every inference slower).
$TotalCores = [int][Environment]::ProcessorCount

$LogDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ---------------------------------------------------------------------------
# EDIT THESE to match your DB camera roster.
#   Name : short label used for log filenames
#   Cams : comma-separated camera IDs passed to --cameras
#   Api  : $true on EXACTLY ONE group - the one that owns the entrance camera
#
# Roster below = the 26 enabled non-ANPR cameras from the DB (CAM-00..CAM-25).
# ANPR-Entry / ANPR-Exit are read by the external ANPR server, not here.
#
# CAM-23 is the B1 Park_Entry gate, so its group is the --api host: ANPR
# webhooks bind the plate in that process's in-memory registry. Each group is
# <= 6 cameras (one serial inference worker per process). Floors are kept
# together; cross-process identity flows via the shared DB + gallery.
# ---------------------------------------------------------------------------
$Groups = @(
    @{ Name = 'b1-gate';  Cams = 'CAM-23,CAM-03,CAM-04,CAM-05,CAM-06,CAM-07'; Api = $true  }
    @{ Name = 'b1-areas'; Cams = 'CAM-08,CAM-20,CAM-21,CAM-22,CAM-24';        Api = $false }
    @{ Name = 'b2-1';     Cams = 'CAM-09,CAM-10,CAM-11,CAM-12,CAM-13,CAM-14'; Api = $false }
    @{ Name = 'b2-2';     Cams = 'CAM-15,CAM-16,CAM-17,CAM-18,CAM-19,CAM-25'; Api = $false }
    @{ Name = 'ground';   Cams = 'CAM-00,CAM-01,CAM-02';                      Api = $false }
)
$ApiPort = 8000
# ---------------------------------------------------------------------------

# Sanity: exactly one API host. NOTE the @(...) wrap - without it, a single
# match returns the lone hashtable and .Count reads its KEY count (3), not the
# collection size. @() forces an array so .Count is the number of API groups.
$apiCount = @($Groups | Where-Object { $_.Api }).Count
if ($apiCount -ne 1) {
    throw "Exactly one group must have Api=`$true (found $apiCount). Fix `$Groups."
}

$pidFile = Join-Path $Root 'run_all.pids'
Remove-Item $pidFile -ErrorAction SilentlyContinue
$started = @()

# Total cameras across all groups - used to slice the CPU thread budget
# proportionally to each group's camera count.
$totalCams = ($Groups | ForEach-Object { ($_.Cams -split ',').Count } | Measure-Object -Sum).Sum

foreach ($g in $Groups) {
    $cliArgs = @('main.py', '--cameras', $g.Cams)
    if ($g.Api) { $cliArgs += @('--api', '--port', "$ApiPort") }

    # Give this group a slice of the cores proportional to its camera count so
    # the parallel processes don't each spin up all-core thread pools and thrash
    # (BLAS/OpenMP/OpenVINO all read these). At least 1.
    $camCount = ($g.Cams -split ',').Count
    $threads = [Math]::Max(1, [int][Math]::Floor($TotalCores * $camCount / $totalCams))
    $env:OMP_NUM_THREADS      = "$threads"
    $env:OPENBLAS_NUM_THREADS = "$threads"
    $env:MKL_NUM_THREADS      = "$threads"
    $env:NUMEXPR_NUM_THREADS  = "$threads"
    $env:VECLIB_MAXIMUM_THREADS = "$threads"

    $out = Join-Path $LogDir ("va_{0}.out.log" -f $g.Name)
    $err = Join-Path $LogDir ("va_{0}.err.log" -f $g.Name)
    $label = if ($g.Api) { "$($g.Name) (+API :$ApiPort)" } else { $g.Name }

    Write-Host "[run_all] starting group '$label' ($threads threads/$TotalCores cores) -> $($g.Cams)"
    $p = Start-Process -FilePath $Python -ArgumentList $cliArgs -WorkingDirectory $Root `
            -WindowStyle Hidden `
            -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    $p.Id | Add-Content -Path $pidFile
    $started += [pscustomobject]@{ Group = $g.Name; PID = $p.Id; Cameras = $g.Cams; Log = $out }
}

Write-Host ''
$started | Format-Table -AutoSize
Write-Host "[run_all] PIDs written to $pidFile"
Write-Host "[run_all] follow a log:  Get-Content -Wait '$LogDir\va_b1-gate.out.log'"
Write-Host "[run_all] stop all:      .\stop_all.ps1"
