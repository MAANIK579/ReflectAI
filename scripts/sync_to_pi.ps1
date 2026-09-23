# scripts/sync_to_pi.ps1 - Fast sync of ReflectAI workspace to Raspberry Pi
param (
    [string]$PiHost = "192.168.173.9",
    [string]$PiUser = "reflectai",
    [string]$PiDest = "~/ReflectAI",
    [switch]$IncludeModel = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " ReflectAI -> Raspberry Pi Fast Sync   " -ForegroundColor Cyan
Write-Host " Target: ${PiUser}@${PiHost}:${PiDest} " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Test ping
$ping = Test-Connection -ComputerName $PiHost -Count 1 -Quiet
if (-not $ping) {
    Write-Host "[ERROR] Cannot ping Raspberry Pi at $PiHost. Ensure it is powered on and on the same Wi-Fi." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Raspberry Pi is online." -ForegroundColor Green

# Folders to sync
$folders = @("app", "face_engine", "tests", "voice_engine")
foreach ($f in $folders) {
    Write-Host "--> Syncing folder '$f'..." -ForegroundColor Yellow
    scp -r $f "${PiUser}@${PiHost}:${PiDest}/"
}

# Sync root files
$rootFiles = @("requirements.txt", "requirements-pi.txt", ".env.example")
foreach ($rf in $rootFiles) {
    if (Test-Path $rf) {
        Write-Host "--> Syncing file '$rf'..." -ForegroundColor Yellow
        scp $rf "${PiUser}@${PiHost}:${PiDest}/"
    }
}

# Optional: Sync 1GB LLM model directly over Wi-Fi
if ($IncludeModel) {
    $modelFile = "voice_engine/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    if (Test-Path $modelFile) {
        Write-Host "--> Syncing 1.06GB Qwen LLM model over Wi-Fi (takes ~30-60s)..." -ForegroundColor Yellow
        scp $modelFile "${PiUser}@${PiHost}:${PiDest}/voice_engine/"
    } else {
        Write-Host "[INFO] Model file $modelFile not found locally." -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Sync Complete! " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "To finalize on Raspberry Pi, SSH in and run:" -ForegroundColor White
Write-Host "  ssh ${PiUser}@${PiHost}" -ForegroundColor Cyan
Write-Host "  cd ${PiDest}" -ForegroundColor Cyan
Write-Host "  pip install faster-whisper piper-tts" -ForegroundColor Cyan
if (-not $IncludeModel) {
    Write-Host "  python3 voice_engine/download_llm.py" -ForegroundColor Cyan
}
Write-Host "  python3 -m app.dashboard.dashboard" -ForegroundColor Cyan
