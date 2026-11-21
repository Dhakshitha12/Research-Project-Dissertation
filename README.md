# Cost-aware serverless pre-warming experiments

This project trains PPO and DQN policies against a trace-driven serverless
simulator, evaluates them on paired held-out arrival traces, and records
confidence intervals for the cost/cold-start frontier. The simulator supports
memory-aware actions and per-runtime/per-handler calibration measurements.

## Requirements

- Python 3.10+
- `stable-baselines3`, `gymnasium`, `numpy`, `pandas`, `scipy`, and `matplotlib`
- `boto3` if you choose to run Lambda calibration; credentials are read
  directly from `user.txt` (AWS credentials-file INI) in the repository root,
  with fallback to the standard AWS credential chain (`~/.aws`, env vars,
  `--profile`) when `user.txt` is absent or has no usable keys
- Docker, `kind`, `kubectl`, Helm, and `faas-cli` only for the optional
  OpenFaaS testbed

Install the local experiment dependencies:

```powershell
py -m pip install stable-baselines3 gymnasium numpy pandas scipy matplotlib
```

## Calibration

The checked-in `data/aws_calibration.csv` contains the older CPU-loop
measurements at 128 MB and 512 MB. It is not a substitute for the requested
image-resize/JSON-transform calibration grid. Calibration creates or updates
Lambda functions, publishes versions, and invokes billable resources. Run it
only when you intend to incur those AWS charges:

```powershell
py -m pip install boto3
py aws\aws_calibration.py --out data\aws_calibration.csv `
  --runtimes python3.12 nodejs20.x `
  --workloads image_resize json_transform `
  --memories 128 256 512 1024 --colds 8 --warms 20
```

Credentials must be configured outside this repository before calibration.
The `--profile` option selects an existing AWS CLI profile; it does not create
one. If you see `ProfileNotFound`, check the configured profile names:

```powershell
aws configure list-profiles
```

For AWS IAM Identity Center (SSO), create and sign in to a profile outside the
repository:

```powershell
aws configure sso --profile thesis-lab
aws sso login --profile thesis-lab
```

Then verify the profile can authenticate before starting the billable
calibration:

```powershell
aws sts get-caller-identity --profile thesis-lab
```

If your organization provides credentials through environment variables or
another AWS credential provider, use that provider instead and omit
`--profile`. For a configured profile, run:

```powershell
py aws\aws_calibration.py --profile thesis-lab --out data\aws_calibration.csv `
  --runtimes python3.12 --workloads image_resize `
  --memories 128 256 512 1024 --colds 8 --warms 20
```

Calibration credentials are loaded directly from `user.txt` in the repository
root (or an explicit path via `--user-credentials`). The file uses the standard
AWS credentials-file INI layout (`[default]` section with `aws_access_key_id`,
`aws_secret_access_key`, and optionally `aws_session_token`; a `--profile`
value selects the `[section]`). Values are parsed in memory and never logged.
If `user.txt` is missing or incomplete, the script falls back to the standard
AWS credential chain. Keep `user.txt` out of version control and restrict its
file permissions (`chmod 600 user.txt`). Never paste credentials into source
files, command history, or this repository. Calibration rows are keyed by
runtime, workload, and memory size; training fails explicitly if a requested
combination has no cold and warm samples.

The two handlers are deterministic, CPU-bound representative tasks:

- `image_resize`: nearest-neighbour resize of a 64x64 RGB image to 32x32
- `json_transform`: filters and transforms 1,000 JSON records and aggregates
  their values

## Full offline protocol

After obtaining calibration rows for the selected handler/runtime and all four
memory sizes, run the full experiment from this directory:

```powershell
$env:CALIBRATION = "data\aws_calibration.csv"
$env:RUNTIME = "python3.12"
$env:WORKLOAD = "image_resize"
.\run_all.ps1
```

The run trains 25 PPO models (40,000 steps x 5 seeds x 5 lambda values) and
five DQN models (40,000 steps x 5 seeds, lambda 0.02), then evaluates all
checkpoints against identical held-out traces. Evaluation checks the complete
model matrix and writes paired statistics, regime-shift results, and
`../results/frontier_points.csv` with bootstrap 95% confidence intervals over
training seeds and held-out workload cases. `../results/frontier.png` shows
the corresponding uncertainty bars.

On Linux/macOS, `bash run_all.sh` runs the same matrix. Both scripts assume the
calibration file already contains the requested runtime/workload/memory rows;
they never provision billable AWS resources automatically.

For a single model during development:

```powershell
py train.py --algo ppo --lam 0.02 --seeds 0 --steps 1000 `
  --runtime python3.12 --workload image_resize `
  --memory-grid-mb 128 256 512 1024 --outdir ..\models
```

## Memory-aware environment

The observation includes ready and starting instance counts for each calibrated
memory size. Actions are: hold, release one idle instance, add one instance at
each memory size, or add three instances at each memory size. The requested
memory grid must have cold-start and warm-execution measurements. Runtime
constants are not extrapolated across workloads or runtimes.

The cost term uses memory-weighted instance-seconds (normalized to a 512 MB
instance for the existing lambda scale); reported dollar cost uses measured
GB-seconds and the configured price proxy.

## OpenFaaS testbed

The cluster testbed assets live under `testbed\`. The setup script provisions
a local kind cluster and OpenFaaS, while the paired replay command sends the
same workload schedule to each selected OpenFaaS function and records request
latency and handler cold-start indicators. The controller deployment example
uses the trained MLP checkpoint and Prometheus metrics; it is a deployment
starting point, not an already-run production rollout.

The testbed requires a running Docker engine and local Kubernetes tooling.
This project does not embed cluster credentials. See `testbed\README.md` for
setup, teardown, replay, and controller deployment details.
