# scripts/sync_to_pi.ps1 - Fast sync of ReflectAI workspace to Raspberry Pi
param (
    [string]$PiHost = "192.168.173.9",
    [string]$PiUser = "reflectai",
    [string]$PiDest = "~/ReflectAI",
    [switch]$NewOnly = $true,         # Defaults to syncing ONLY new/changed files
    [switch]$All = $false,             # Set to $true to force full sync of all folders
    [string[]]$Files = @(),            # Specific file(s) to sync
    [string]$Since = "",               # Optional git ref or commit to diff against (e.g. HEAD~1, origin/main)
    [switch]$IncludeModel = $false,    # Optional: sync the 1GB GGUF model
    [switch]$SkipPing = $false         # Skip ping test if ICMP is blocked
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " ReflectAI -> Raspberry Pi Fast Sync   " -ForegroundColor Cyan
Write-Host " Target: ${PiUser}@${PiHost}:${PiDest} " -ForegroundColor Cyan
Write-Host " Mode  : $(if ($All) { 'FULL FOLDER SYNC' } else { 'INCREMENTAL (NEW/CHANGED ONLY)' })" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Test connectivity
if (-not $SkipPing) {
    Write-Host "[*] Checking connectivity to $PiHost..." -ForegroundColor Gray
    $ping = Test-Connection -ComputerName $PiHost -Count 1 -Quiet -ErrorAction SilentlyContinue
    if (-not $ping) {
        # Check if SSH port 22 is open even if ICMP ping is blocked
        $tcp = Test-NetConnection -ComputerName $PiHost -Port 22 -WarningAction SilentlyContinue -InformationLevel Quiet
        if (-not $tcp) {
            Write-Host "[WARNING] Cannot reach Raspberry Pi at $PiHost." -ForegroundColor Red
            Write-Host "  1. Ensure the Pi is powered on and connected to your network/hotspot." -ForegroundColor Yellow
            Write-Host "  2. If the Pi's IP address has changed, specify it with: -PiHost <NEW_IP>" -ForegroundColor Yellow
            Write-Host "     Example: .\scripts\sync_to_pi.ps1 -PiHost 192.168.1.50" -ForegroundColor Yellow
            Write-Host "  3. To bypass ping checks, add: -SkipPing" -ForegroundColor Yellow
            Write-Host ""
            $choice = Read-Host "Try proceeding anyway? (y/N)"
            if ($choice -ne "y" -and $choice -ne "Y") {
                exit 1
            }
        } else {
            Write-Host "[OK] SSH port 22 is open on $PiHost." -ForegroundColor Green
        }
    } else {
        Write-Host "[OK] Raspberry Pi is online." -ForegroundColor Green
    }
}

# -------------------------------------------------------------------------
# Determine files to sync
# -------------------------------------------------------------------------
$filesToSync = @()

if ($Files.Count -gt 0) {
    # User specified exact files
    $filesToSync = $Files
} elseif (-not $All) {
    # 1. Detect uncommitted modified and newly added files via git
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $porcelain = git status --porcelain
        foreach ($line in $porcelain) {
            if ($line -match '^[ MADRCU?]{2}\s+(.+)$') {
                $f = $Matches[1].Trim().Trim('"')
                if ($f -notmatch '\.pyc$' -and $f -notmatch '__pycache__' -and $f -notmatch '\.db$' -and $f -notmatch '^\.git') {
                    if (Test-Path $f) {
                        $filesToSync += $f
                    }
                }
            }
        }
    }

    # 1B. If $Since is specified, diff against that commit/branch
    if ($Since -and (Get-Command git -ErrorAction SilentlyContinue)) {
        $sinceFiles = git diff --name-only $Since HEAD 2>$null
        if ($sinceFiles) {
            foreach ($f in $sinceFiles) {
                if (Test-Path $f -and $filesToSync -notcontains $f) {
                    $filesToSync += $f
                }
            }
        }
    }

    # 2. If git working tree is clean, look for files in recent commit
    if ($filesToSync.Count -eq 0 -and (Get-Command git -ErrorAction SilentlyContinue)) {
        $recentCommitFiles = git diff --name-only HEAD~1 HEAD 2>$null
        if ($recentCommitFiles) {
            foreach ($f in $recentCommitFiles) {
                if (Test-Path $f) {
                    $filesToSync += $f
                }
            }
        }
    }

    # 3. Fallback: files modified in the last 48 hours
    if ($filesToSync.Count -eq 0) {
        Write-Host "[INFO] No git changes detected. Searching for files modified in last 48 hours..." -ForegroundColor Gray
        $cutoff = (Get-Date).AddDays(-2)
        $recentFiles = Get-ChildItem -Path "app", "voice_engine", "face_engine", "tests" -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.LastWriteTime -gt $cutoff -and $_.FullName -notmatch '__pycache__' -and $_.Extension -ne '.pyc' -and $_.Extension -ne '.db' } |
            ForEach-Object { Resolve-Path -Relative $_.FullName } |
            ForEach-Object { $_.TrimStart(".\") -replace "\\", "/" }

        $filesToSync = @($recentFiles)
    }
}

# -------------------------------------------------------------------------
# Perform Sync
# -------------------------------------------------------------------------
if (-not $All) {
    if ($filesToSync.Count -eq 0) {
        Write-Host "[INFO] No new or modified files found to sync." -ForegroundColor Green
        Write-Host "To sync all project folders, run: .\scripts\sync_to_pi.ps1 -All" -ForegroundColor Yellow
        exit 0
    }

    Write-Host ""
    Write-Host "Found $($filesToSync.Count) new/modified file(s) to sync:" -ForegroundColor Cyan
    foreach ($f in $filesToSync) {
        Write-Host "  + $f" -ForegroundColor Yellow
    }
    Write-Host ""

    # Ensure remote parent directories exist on Pi
    $remoteDirs = $filesToSync | ForEach-Object {
        $parent = Split-Path ($_ -replace "\\", "/") -Parent
        if ($parent) { "${PiDest}/${parent}" -replace "\\", "/" } else { "${PiDest}" }
    } | Select-Object -Unique

    Write-Host "[*] Creating remote directory structure on Raspberry Pi..." -ForegroundColor Gray
    $mkdirArgs = ($remoteDirs | ForEach-Object { "'$_'" }) -join " "
    ssh -o ConnectTimeout=8 "${PiUser}@${PiHost}" "mkdir -p $mkdirArgs"

    # Transfer each changed file
    $successCount = 0
    foreach ($f in $filesToSync) {
        $linuxPath = $f -replace "\\", "/"
        $parentDir = Split-Path $linuxPath -Parent
        $targetDir = if ($parentDir) { "${PiDest}/${parentDir}/" -replace "\\", "/" } else { "${PiDest}/" }

        Write-Host "--> Uploading: $linuxPath -> $targetDir" -ForegroundColor Yellow
        scp "$f" "${PiUser}@${PiHost}:${targetDir}"
        if ($LASTEXITCODE -eq 0) {
            $successCount++
        } else {
            Write-Host "[ERROR] Failed to transfer $f" -ForegroundColor Red
        }
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " Incremental Sync Complete ($successCount/$($filesToSync.Count) uploaded) " -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

} else {
    # Full folder sync mode
    Write-Host "[*] Performing FULL folder sync..." -ForegroundColor Yellow
    $folders = @("app", "face_engine", "tests", "voice_engine")
    foreach ($f in $folders) {
        Write-Host "--> Syncing folder '$f'..." -ForegroundColor Yellow
        scp -r $f "${PiUser}@${PiHost}:${PiDest}/"
    }

    $rootFiles = @("requirements.txt", "requirements-pi.txt", ".env.example", "run_dashboard.py", "run.py")
    foreach ($rf in $rootFiles) {
        if (Test-Path $rf) {
            Write-Host "--> Syncing file '$rf'..." -ForegroundColor Yellow
            scp $rf "${PiUser}@${PiHost}:${PiDest}/"
        }
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " Full Sync Complete! " -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}

# Optional: Model Sync
if ($IncludeModel) {
    $modelFile = "voice_engine/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    if (Test-Path $modelFile) {
        Write-Host "--> Syncing Qwen LLM model over Wi-Fi..." -ForegroundColor Yellow
        scp $modelFile "${PiUser}@${PiHost}:${PiDest}/voice_engine/"
    }
}

Write-Host ""
Write-Host "Next steps on Raspberry Pi:" -ForegroundColor White
Write-Host "  ssh ${PiUser}@${PiHost}" -ForegroundColor Cyan
Write-Host "  cd ${PiDest}" -ForegroundColor Cyan
Write-Host "  python3 -m app.dashboard.dashboard" -ForegroundColor Cyan
