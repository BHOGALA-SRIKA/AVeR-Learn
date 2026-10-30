# One command to measure SLM accuracy on this laptop (Windows PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File experiments\run_slm_eval.ps1          # quick: 6 topics
#   powershell -ExecutionPolicy Bypass -File experiments\run_slm_eval.ps1 -Full    # all 18 topics
#
# What it does:
#   1. installs Ollama with winget if it is not installed
#   2. keeps the models on D: (D:\ollama-models) when a D: drive exists, so C: stays free
#   3. starts Ollama if it is not running
#   4. downloads each small model only if there is enough free disk space
#   5. runs experiments/slm_eval.py and prints the accuracy table
# After the downloads everything runs offline.

param(
    [switch]$Full,
    [string]$ModelDir = "D:\ollama-models",
    [string[]]$Models = @("qwen2.5:1.5b", "gemma3:1b", "llama3.2:1b", "qwen3:1.7b")
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# approximate download sizes in GB
$Size = @{ "qwen2.5:1.5b" = 1.0; "gemma3:1b" = 0.9; "llama3.2:1b" = 1.4; "qwen3:1.7b" = 1.5 }
$KeepFreeGB = 3

# 1. Ollama installed?
$ollamaDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    if (Test-Path (Join-Path $ollamaDir "ollama.exe")) {
        $env:Path += ";$ollamaDir"
    } else {
        Write-Host "Installing Ollama (one time)..." -ForegroundColor Cyan
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
        $env:Path += ";$ollamaDir"
        if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
            Write-Host "Ollama did not install. Install it from https://ollama.com/download and run this again." -ForegroundColor Red
            exit 1
        }
    }
}

# 2. where to keep the models
$useModelDir = (Test-Path (Split-Path $ModelDir -Qualifier))
if ($useModelDir) {
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    $changed = ([Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User") -ne $ModelDir)
    [Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ModelDir, "User")
    $env:OLLAMA_MODELS = $ModelDir
    Write-Host "Models will be stored in $ModelDir" -ForegroundColor Cyan
    if ($changed) {
        # a running Ollama still uses the old folder; restart it with the new one
        Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
}

# 3. Ollama running?
function Test-Ollama {
    try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null; return $true }
    catch { return $false }
}
if (-not (Test-Ollama)) {
    Write-Host "Starting Ollama..." -ForegroundColor Cyan
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 20 -and -not (Test-Ollama); $i++) { Start-Sleep -Seconds 1 }
    if (-not (Test-Ollama)) { Write-Host "Ollama is not responding. Open the Ollama app and run this again." -ForegroundColor Red; exit 1 }
}

# 4. download models that fit on the disk
$have = (Invoke-RestMethod -Uri "http://localhost:11434/api/tags").models | ForEach-Object { $_.name }
$ready = @()
foreach ($m in $Models) {
    if ($have -contains $m) { $ready += $m; continue }
    $driveLetter = $env:USERPROFILE.Substring(0, 1)
    if ($useModelDir) { $driveLetter = $ModelDir.Substring(0, 1) }
    $drive = Get-PSDrive -Name $driveLetter
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    $need = 2.0
    if ($Size.ContainsKey($m)) { $need = $Size[$m] }
    if ($freeGB -lt ($need + $KeepFreeGB)) {
        Write-Host "Skipping $m - needs about $need GB, only $freeGB GB free." -ForegroundColor Yellow
        continue
    }
    Write-Host "Downloading $m (about $need GB, $freeGB GB free)..." -ForegroundColor Cyan
    ollama pull $m
    if ($LASTEXITCODE -eq 0) { $ready += $m } else { Write-Host "Download of $m failed." -ForegroundColor Yellow }
}
if ($ready.Count -eq 0) { Write-Host "No model available. Free some disk space and run again." -ForegroundColor Red; exit 1 }

# 5. measure
$args2 = @("experiments/slm_eval.py", "--models") + $ready
if (-not $Full) { $args2 += "--quick" }
Write-Host "Testing: $($ready -join ', ')" -ForegroundColor Cyan
python @args2
