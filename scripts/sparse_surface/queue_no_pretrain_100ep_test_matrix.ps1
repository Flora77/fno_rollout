param(
    [int]$WaitForProcessId = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExe = "D:\software\miniconda3\envs\fno\python.exe"
$MatrixScript = Join-Path $ProjectRoot "scripts\sparse_surface\run_no_pretrain_100ep_test_matrix.py"
$ResultDir = Join-Path $ProjectRoot "results\no_pretrain_100ep_test_matrix_20260731_v1"
$ConsoleLog = Join-Path $ResultDir "orchestrator_console.log"
$QueueLog = Join-Path $ResultDir "queue.log"

New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
try {
    "queue_started pid=$PID wait_for=$WaitForProcessId" |
        Out-File -FilePath $QueueLog -Append -Encoding utf8
    if ($WaitForProcessId -gt 0) {
        while ($null -ne (
            Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
        )) {
            Start-Sleep -Seconds 30
        }
    }

    "launching_matrix" | Out-File -FilePath $QueueLog -Append -Encoding utf8
    Push-Location $ProjectRoot
    try {
        & $PythonExe $MatrixScript --allow-test *>> $ConsoleLog
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}
catch {
    ($_ | Out-String) | Out-File -FilePath $QueueLog -Append -Encoding utf8
    exit 99
}
