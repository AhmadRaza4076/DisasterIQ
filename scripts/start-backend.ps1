# Start backend (Windows Python venv)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$backendDir = Join-Path $repoRoot "backend"
$envFile = Join-Path $repoRoot ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"
$systemPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating venv..."
    if (-not (Test-Path $systemPython)) {
        Write-Host "Python 3.12 not found. Install with: winget install Python.Python.3.12" -ForegroundColor Red
        exit 1
    }
    & $systemPython -m venv (Join-Path $backendDir ".venv")
    & $venvPython -m pip install -r (Join-Path $backendDir "requirements.txt")
}

Set-Location $backendDir
& $venvPython -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
