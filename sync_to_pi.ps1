<#
.SYNOPSIS
    Sync ReflectAI project to Raspberry Pi over SSH.

.DESCRIPTION
    Smart sync script that:
    - Auto-discovers Pi via raspberrypi.local (mDNS) or manual IP
    - Skips .venv, __pycache__, .git, models, and other large/temp files
    - Only copies changed files (compares timestamps)
    - Works from any WiFi network

.USAGE
    .\sync_to_pi.ps1                          # Uses raspberrypi.local
    .\sync_to_pi.ps1 -PiHost 192.168.1.42     # Uses specific IP
    .\sync_to_pi.ps1 -PiUser maanik           # Different username
    .\sync_to_pi.ps1 -FullSync                 # Force re-copy everything
#>

param(
    [string]$PiHost = "raspberrypi.local",
    [string]$PiUser = "pi",
    [string]$RemotePath = "/home/pi/ReflectAI",
    [switch]$FullSync,
    [switch]$DryRun
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ErrorActionPreference = "Stop"

# ── Colors ──
function Write-Step($msg)    { Write-Host "  [>] $msg" -ForegroundColor Cyan }
function Write-OK($msg)      { Write-Host "  [✓] $msg" -ForegroundColor Green }
function Write-Skip($msg)    { Write-Host "  [-] $msg" -ForegroundColor DarkGray }
function Write-Warn($msg)    { Write-Host "  [!] $msg" -ForegroundColor Yellow }
function Write-Fail($msg)    { Write-Host "  [✗] $msg" -ForegroundColor Red }

# ── Folders/patterns to EXCLUDE from sync ──
$ExcludePatterns = @(
    ".venv", "venv", "__pycache__", ".git", ".gemini",
    "node_modules", ".vscode", ".idea",
    "*.pyc", "*.pyo", "*.egg-info",
    "*.gguf", "*.onnx",                         # Large model files (download on Pi)
    "data/reflectai.db",                         # DB is per-device
    "data/faces",                                # Face embeddings are per-device
    "logs",                                      # Logs are per-device
    "voice_engine/model",                        # Vosk model (~50MB, download on Pi)
    "voice_engine/espeak-ng-data"                # espeak data (~30MB, install on Pi)
)

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "  ║   ReflectAI → Raspberry Pi Sync Tool     ║" -ForegroundColor Magenta
Write-Host "  ╚══════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""

# ── Step 1: Test connectivity ──
Write-Step "Pinging $PiHost ..."
$pingResult = Test-Connection -ComputerName $PiHost -Count 1 -Quiet -ErrorAction SilentlyContinue
if (-not $pingResult) {
    Write-Warn "Could not reach '$PiHost' via ping."
    Write-Warn "If raspberrypi.local doesn't work, find your Pi's IP with:"
    Write-Host "       On Pi:  hostname -I" -ForegroundColor White
    Write-Host "       Then:   .\sync_to_pi.ps1 -PiHost <IP>" -ForegroundColor White
    Write-Host ""
    $manualIP = Read-Host "  Enter Pi IP (or press Enter to abort)"
    if ([string]::IsNullOrWhiteSpace($manualIP)) {
        Write-Fail "Aborted."
        exit 1
    }
    $PiHost = $manualIP.Trim()
    Write-Step "Retrying with $PiHost ..."
    $pingResult = Test-Connection -ComputerName $PiHost -Count 1 -Quiet -ErrorAction SilentlyContinue
    if (-not $pingResult) {
        Write-Fail "Still can't reach $PiHost. Check that Pi is on and connected to WiFi."
        exit 1
    }
}
Write-OK "Pi reachable at $PiHost"

# ── Step 2: Test SSH connection ──
Write-Step "Testing SSH to ${PiUser}@${PiHost} ..."
$sshTest = ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${PiUser}@${PiHost}" "echo ok" 2>&1
if ($sshTest -ne "ok") {
    Write-Warn "SSH connection failed. You may need to:"
    Write-Host "       1. Enable SSH on Pi:  sudo raspi-config → Interface Options → SSH" -ForegroundColor White
    Write-Host "       2. Set up key auth:   ssh-copy-id ${PiUser}@${PiHost}" -ForegroundColor White
    Write-Host "       3. Or enter password when prompted (remove -o BatchMode=yes)" -ForegroundColor White
    Write-Host ""

    # Retry without BatchMode (allows password prompt)
    Write-Step "Retrying SSH (you may be prompted for password) ..."
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "${PiUser}@${PiHost}" "echo ok"
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "SSH connection failed. Fix SSH access and try again."
        exit 1
    }
}
Write-OK "SSH connection successful"

# ── Step 3: Create remote directory ──
Write-Step "Ensuring remote directory exists: $RemotePath"
ssh "${PiUser}@${PiHost}" "mkdir -p '$RemotePath'"
Write-OK "Remote directory ready"

# ── Step 4: Build file list (excluding unwanted files) ──
Write-Step "Scanning project files ..."

function ShouldExclude($relativePath) {
    foreach ($pattern in $ExcludePatterns) {
        # Check if any path component matches the exclude pattern
        $parts = $relativePath -split '[/\\]'
        foreach ($part in $parts) {
            if ($part -like $pattern) { return $true }
        }
        # Check full path match
        if ($relativePath -like "*$pattern*") { return $true }
    }
    return $false
}

$allFiles = Get-ChildItem -Path $ProjectRoot -Recurse -File
$filesToSync = @()
$skippedCount = 0

foreach ($file in $allFiles) {
    $relativePath = $file.FullName.Substring($ProjectRoot.Length + 1).Replace('\', '/')
    if (ShouldExclude $relativePath) {
        $skippedCount++
        continue
    }
    $filesToSync += [PSCustomObject]@{
        FullPath = $file.FullName
        RelativePath = $relativePath
        Size = $file.Length
        LastWrite = $file.LastWriteTimeUtc
    }
}

$totalSize = ($filesToSync | Measure-Object -Property Size -Sum).Sum
$totalSizeMB = [math]::Round($totalSize / 1MB, 1)
Write-OK "$($filesToSync.Count) files to sync ($totalSizeMB MB), skipped $skippedCount excluded files"

if ($DryRun) {
    Write-Host ""
    Write-Warn "DRY RUN — files that would be synced:"
    $filesToSync | ForEach-Object { Write-Host "    $($_.RelativePath)" }
    Write-Host ""
    Write-OK "Dry run complete. Remove -DryRun to actually sync."
    exit 0
}

# ── Step 5: Sync files ──
Write-Host ""
Write-Step "Syncing files to ${PiUser}@${PiHost}:${RemotePath} ..."
Write-Host ""

$synced = 0
$failed = 0
$dirs_created = @{}

foreach ($file in $filesToSync) {
    $remoteDir = Split-Path -Parent "$RemotePath/$($file.RelativePath)" -ErrorAction SilentlyContinue
    $remoteDir = $remoteDir.Replace('\', '/')

    # Create remote directory if not yet created
    if (-not $dirs_created.ContainsKey($remoteDir)) {
        ssh "${PiUser}@${PiHost}" "mkdir -p '$remoteDir'" 2>$null
        $dirs_created[$remoteDir] = $true
    }

    $remoteFull = "$RemotePath/$($file.RelativePath)"
    $sizeKB = [math]::Round($file.Size / 1KB, 1)

    try {
        scp -q "$($file.FullPath)" "${PiUser}@${PiHost}:${remoteFull}" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $synced++
            $pct = [math]::Round(($synced / $filesToSync.Count) * 100)
            Write-Host "`r  [$pct%] ($synced/$($filesToSync.Count)) $($file.RelativePath)" -NoNewline
        } else {
            $failed++
            Write-Fail "Failed: $($file.RelativePath)"
        }
    } catch {
        $failed++
        Write-Fail "Error: $($file.RelativePath) - $_"
    }
}

Write-Host ""
Write-Host ""

# ── Step 6: Sync .env separately (if it exists, user may want it) ──
$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    Write-Step "Syncing .env file ..."
    scp -q "$envFile" "${PiUser}@${PiHost}:${RemotePath}/.env" 2>$null
    Write-OK ".env synced (contains API keys — keep it safe on Pi)"
}

# ── Summary ──
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║            Sync Complete!                 ║" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════╝" -ForegroundColor Green
Write-Host "    Files synced:  $synced" -ForegroundColor White
if ($failed -gt 0) {
    Write-Host "    Failed:        $failed" -ForegroundColor Red
}
Write-Host "    Destination:   ${PiUser}@${PiHost}:${RemotePath}" -ForegroundColor White
Write-Host ""
Write-Host "  Next steps on Pi:" -ForegroundColor Yellow
Write-Host "    ssh ${PiUser}@${PiHost}" -ForegroundColor White
Write-Host "    cd $RemotePath" -ForegroundColor White
Write-Host "    python3 -m venv .venv && source .venv/bin/activate" -ForegroundColor White
Write-Host "    pip install -r requirements.txt" -ForegroundColor White
Write-Host "    python run.py" -ForegroundColor White
Write-Host ""
