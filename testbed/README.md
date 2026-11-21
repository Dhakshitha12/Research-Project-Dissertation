# kind + OpenFaaS testbed

This is a local-cluster integration testbed for the calibrated image-resize and
JSON-transform handlers. A running Docker engine is required. The kind cluster
is isolated and does not use cloud credentials.

## Provision

Install Docker Desktop, kind, kubectl, Helm, and faas-cli, then run:

```powershell
.\bootstrap.ps1
```

In a second PowerShell window, forward the local gateway:

```powershell
kubectl -n openfaas port-forward svc/gateway 8080:8080
```

In the first window, deploy the functions:

```powershell
.\deploy-functions.ps1
```

The local basic-auth secret is read from the newly provisioned OpenFaaS
namespace for `faas-cli login`; it is not saved or printed. Do not copy
credentials from `user.txt` into this cluster or into repository files.

## Paired replay

Replay the same held-out traces to both functions and collect raw response
latencies and handler-reported cold-start indicators:

```powershell
py replay.py `
  --function image-resize=http://127.0.0.1:8080/function/image-resize `
  --function json-transform=http://127.0.0.1:8080/function/json-transform `
  --out ..\results\openfaas_replay.csv
```

The default 1x replay preserves the one-second simulator clock and can take
multiple hours across all held-out cases. `--family poisson_ho` limits a run to
one workload family. `--speedup 10` compresses arrival times for a smoke test;
do not treat that run as a latency-equivalent 1x evaluation.

Each function returns a `cold_start` field that is true for the first request
handled by a process. This is a process-restart indicator, not a platform
initialization metric. For cluster cold-start comparisons, also retain
Prometheus data and correlate it with pod creation/readiness events.

## Policy sidecar deployment

The controller polls Prometheus every second, reconstructs the trained
observation, runs the selected PPO/DQN checkpoint, and patches only the
configured OpenFaaS function Deployment. Its Kubernetes Role uses
`resourceNames` to constrain access. First, expose/scrape the OpenFaaS
invocation counter and confirm the metric name and labels in the Prometheus
UI; adjust the `PROMQL_JSON` entries in
`controller\controller.yaml` if the chart version differs. Empty or ambiguous
Prometheus results are reported as unhealthy rather than replaced with
synthetic values.

Build and deploy a controller trained with the matching runtime, workload,
calibration file, and memory grid:

```powershell
.\deploy-controller.ps1 -ModelPath ..\models\ppo_lam0.02_seed0.zip
```

The script loads the checkpoint into a PVC before waiting for controller
readiness. The controller image includes the calibration CSV present at build
time, so build it after replacing the legacy CPU-loop CSV with the required
measured runtime/workload/memory rows.

The simulator can represent mixed memory tiers in one pool; a Kubernetes
Deployment has one memory limit for all replicas. The adapter therefore maps a
memory action to a memory-limit rollout of that function Deployment. Report
rollout time separately from simulator per-instance scale-up time; they are
not equivalent. A multi-tier cluster experiment needs separate deployments
behind a routing layer.

The Docker and Kubernetes tools were unavailable in the authoring environment,
so no real-cluster deployment or replay results are included.

## Teardown

```powershell
kind delete cluster --name prewarm
```
