$ErrorActionPreference = "Stop"

foreach ($tool in @("docker", "kind", "kubectl", "faas-cli")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required command '$tool' was not found on PATH."
    }
}

Push-Location $PSScriptRoot
try {
    faas-cli template store pull python3-http
    if ($LASTEXITCODE -ne 0) { throw "Could not fetch the python3-http template." }

    faas-cli build -f stack.yml
    if ($LASTEXITCODE -ne 0) { throw "OpenFaaS function image build failed." }

    kind load docker-image image-resize:local --name prewarm
    if ($LASTEXITCODE -ne 0) { throw "Could not load image-resize into kind." }
    kind load docker-image json-transform:local --name prewarm
    if ($LASTEXITCODE -ne 0) { throw "Could not load json-transform into kind." }

    $encoded = kubectl -n openfaas get secret basic-auth `
        -o jsonpath="{.data.basic-auth-password}"
    if ($LASTEXITCODE -ne 0 -or -not $encoded) {
        throw "Could not read the local OpenFaaS basic-auth secret."
    }
    $password = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encoded))
    faas-cli login --username admin --password $password --gateway http://127.0.0.1:8080
    if ($LASTEXITCODE -ne 0) {
        throw "OpenFaaS login failed. Confirm the gateway port-forward is active."
    }
    faas-cli deploy -f stack.yml --gateway http://127.0.0.1:8080
    if ($LASTEXITCODE -ne 0) { throw "OpenFaaS function deployment failed." }
    faas-cli list --gateway http://127.0.0.1:8080
}
finally {
    Pop-Location
}
