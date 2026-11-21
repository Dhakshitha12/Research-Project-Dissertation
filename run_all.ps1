$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$runtime = if ($env:RUNTIME) { $env:RUNTIME } else { "python3.12" }
$workload = if ($env:WORKLOAD) { $env:WORKLOAD } else { "image_resize" }
$calibration = if ($env:CALIBRATION) {
    $env:CALIBRATION
} else {
    "data\aws_calibration.csv"
}
$memoryGrid = if ($env:MEMORY_GRID) {
    $env:MEMORY_GRID -split "\s+"
} else {
    @("128", "256", "512", "1024")
}
if (-not (Test-Path $calibration -PathType Leaf)) {
    throw "Missing calibration CSV '$calibration'. Run the billable calibration command in README.md first."
}

py -m pip install stable-baselines3 gymnasium numpy pandas scipy matplotlib
if ($LASTEXITCODE -ne 0) { throw "Could not install experiment dependencies." }

$memoryArgs = @()
foreach ($memory in $memoryGrid) {
    $memoryArgs += [string]$memory
}
$profileArgs = @(
    "--calibration", $calibration,
    "--runtime", $runtime,
    "--workload", $workload,
    "--memory-grid-mb"
) + $memoryArgs

foreach ($lambda in @("0.005", "0.01", "0.02", "0.05", "0.1")) {
    Write-Host "Training PPO λ=$lambda, seeds 0..4, 40000 steps"
    py train.py --algo ppo --lam $lambda --seeds 0 1 2 3 4 --steps 40000 `
        @profileArgs --outdir ..\models
    if ($LASTEXITCODE -ne 0) { throw "PPO training failed for lambda $lambda." }
}

Write-Host "Training DQN λ=0.02, seeds 0..4, 40000 steps"
py train.py --algo dqn --lam 0.02 --seeds 0 1 2 3 4 --steps 40000 `
    @profileArgs --outdir ..\models
if ($LASTEXITCODE -ne 0) { throw "DQN training failed." }

py evaluate.py --models ..\models --out ..\results --require-full-protocol `
    @profileArgs
if ($LASTEXITCODE -ne 0) { throw "Paired evaluation failed." }

Write-Host "Full protocol complete. See ..\results for evaluation and confidence intervals."
