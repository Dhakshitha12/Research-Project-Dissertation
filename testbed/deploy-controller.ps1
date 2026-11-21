param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath
)
$ErrorActionPreference = "Stop"

foreach ($tool in @("docker", "kind", "kubectl")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required command '$tool' was not found on PATH."
    }
}
if (-not (Test-Path $ModelPath -PathType Leaf)) {
    throw "Model checkpoint does not exist: $ModelPath"
}

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
docker build -f "$PSScriptRoot\controller\Dockerfile" `
    -t prewarm-policy-controller:latest $repoRoot
if ($LASTEXITCODE -ne 0) { throw "Controller image build failed." }

kind load docker-image prewarm-policy-controller:latest --name prewarm
if ($LASTEXITCODE -ne 0) { throw "Could not load controller image into kind." }
kubectl apply -f "$PSScriptRoot\controller\controller.yaml"
if ($LASTEXITCODE -ne 0) { throw "Could not apply controller resources." }

$pod = kubectl -n openfaas-fn get pods `
    -l app=prewarm-controller-image-resize `
    -o jsonpath="{.items[0].metadata.name}"
if ($LASTEXITCODE -ne 0 -or -not $pod) {
    throw "Could not find the policy controller pod."
}
kubectl -n openfaas-fn cp $ModelPath "openfaas-fn/$($pod):/models/policy.zip" `
    -c wait-for-policy
if ($LASTEXITCODE -ne 0) { throw "Could not copy the policy checkpoint to the PVC." }

kubectl -n openfaas-fn rollout status deployment/prewarm-controller-image-resize `
    --timeout=5m
if ($LASTEXITCODE -ne 0) { throw "Policy controller did not become ready." }
Write-Host "Controller deployed. Health endpoint is available inside openfaas-fn on port 8081."
