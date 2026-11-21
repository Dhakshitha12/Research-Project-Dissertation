$ErrorActionPreference = "Stop"

foreach ($tool in @("docker", "kind", "kubectl", "helm")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required command '$tool' was not found on PATH."
    }
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not running. Start Docker Desktop and retry."
}

$cluster = "prewarm"
$existing = kind get clusters
if ($existing -notcontains $cluster) {
    kind create cluster --config "$PSScriptRoot\kind-config.yaml"
    if ($LASTEXITCODE -ne 0) { throw "kind cluster creation failed." }
}

kubectl create namespace openfaas --dry-run=client -o yaml | kubectl apply -f -
if ($LASTEXITCODE -ne 0) { throw "Could not create the openfaas namespace." }
kubectl create namespace openfaas-fn --dry-run=client -o yaml | kubectl apply -f -
if ($LASTEXITCODE -ne 0) { throw "Could not create the openfaas-fn namespace." }

helm repo add openfaas https://openfaas.github.io/faas-netes/
if ($LASTEXITCODE -ne 0) { throw "Could not add the OpenFaaS Helm repository." }
helm repo update
if ($LASTEXITCODE -ne 0) { throw "Could not update Helm repositories." }
helm upgrade --install openfaas openfaas/openfaas `
    --namespace openfaas `
    --set functionNamespace=openfaas-fn `
    --set generateBasicAuth=true `
    --wait --timeout 5m
if ($LASTEXITCODE -ne 0) { throw "OpenFaaS Helm installation failed." }

kubectl -n openfaas get deployments
Write-Host ""
Write-Host "OpenFaaS is installed. In a separate terminal run:"
Write-Host "kubectl -n openfaas port-forward svc/gateway 8080:8080"
