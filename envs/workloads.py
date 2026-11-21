"""
Workload generators and the Azure Functions 2019 trace loader.

Generators produce per-second arrival-count arrays, following the proposal's
assumption that per-minute Azure trace counts are spread to per-second
arrivals with a Poisson draw (stated openly in Section 3.4 / Data sources).

The azure_like_stream is a synthetic stand-in calibrated to the published
statistics of Shahrad et al. (2020): 45% of applications invoked <=1/hour,
81% <=1/minute, and the most popular 18.6% of applications account for 99.6%
of invocations. load_azure_invocations() consumes the REAL trace when the
CSV is supplied, so the calibrated stand-in can be replaced 1:1.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
import csv
import statistics

# Published AWS Lambda x86 price per GB-second ($0.0000166667) used as the
# FaaS cost proxy per the proposal ("published FaaS price as the proxy").
GB_S_PRICE_USD = 0.0000166667


@dataclass
class FunctionProfile:
    """Per-function runtime constants, optionally calibrated by memory size."""
    name: str
    mem_gb: float
    exec_s: float
    cold_s: float
    runtime: str = "python3.12"
    workload: str = "cpu_loop"
    memory_profiles: dict[int, tuple[float, float]] = field(default_factory=dict)

    @property
    def memory_grid_mb(self):
        return tuple(sorted(self.memory_profiles)) or (round(self.mem_gb * 1024),)

    def constants_for(self, memory_mb: int) -> tuple[float, float]:
        if memory_mb in self.memory_profiles:
            return self.memory_profiles[memory_mb]
        if memory_mb == round(self.mem_gb * 1024):
            return self.exec_s, self.cold_s
        raise ValueError(
            f"No calibration for {self.runtime}/{self.workload} at {memory_mb} MB"
        )


def load_function_profile(csv_path, runtime="python3.12", workload="cpu_loop",
                          memory_grid_mb=None) -> FunctionProfile:
    """Load median warm execution and cold-init measurements from calibration CSV."""
    measurements = {}
    with open(csv_path, newline="") as stream:
        for row in csv.DictReader(stream):
            row_runtime = row.get("runtime") or "python3.12"
            row_workload = row.get("workload") or "cpu_loop"
            if row_runtime != runtime or row_workload != workload:
                continue
            memory_mb = int(row["memory_mb"])
            values = measurements.setdefault(memory_mb, {"cold": [], "warm": []})
            kind = row["kind"]
            if kind == "cold":
                values["cold"].append(float(row["init_ms"]) / 1000)
            elif kind == "warm":
                values["warm"].append(float(row["duration_ms"]) / 1000)

    if not measurements:
        raise ValueError(f"No calibration samples found for {runtime}/{workload} in {csv_path}")
    requested = sorted(set(memory_grid_mb or measurements))
    missing = [m for m in requested if m not in measurements
               or not measurements[m]["cold"] or not measurements[m]["warm"]]
    if missing:
        raise ValueError(
            f"Missing cold and warm calibration samples for {runtime}/{workload} "
            f"at {missing} MB in {csv_path}"
        )

    profiles = {
        memory_mb: (statistics.median(measurements[memory_mb]["warm"]),
                    statistics.median(measurements[memory_mb]["cold"]))
        for memory_mb in requested
    }
    default_memory = 512 if 512 in profiles else requested[0]
    exec_s, cold_s = profiles[default_memory]
    return FunctionProfile(
        name=f"{workload}-{runtime}",
        mem_gb=default_memory / 1024,
        exec_s=exec_s,
        cold_s=cold_s,
        runtime=runtime,
        workload=workload,
        memory_profiles=profiles,
    )


# Default profile; cold_s is overwritten by the AWS calibration measurement.
DEFAULT_PROFILE = FunctionProfile("demo-fn", mem_gb=0.5, exec_s=0.0415, cold_s=0.082)  # AWS Lambda (us-east-1, 512 MB) measured medians, see data/aws_calibration.csv


# --------------------------------------------------------------------------
# Per-minute -> per-second spreading (Poisson thinning), as in the proposal
# --------------------------------------------------------------------------
def spread_minutes_to_seconds(minute_counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    minute_counts = np.asarray(minute_counts, dtype=np.int64)
    secs = np.zeros(minute_counts.size * 60, dtype=np.int64)
    for m, c in enumerate(minute_counts):
        if c > 0:
            idx = rng.integers(0, 60, int(c)) + m * 60
            np.add.at(secs, idx, 1)
    return secs


# --------------------------------------------------------------------------
# Synthetic generators: Poisson, bursty, periodic, diurnal, sparse
# --------------------------------------------------------------------------
def poisson_stream(T, rng, rate=0.2):
    return rng.poisson(rate, T).astype(np.int64)


def bursty_stream(T, rng, base=0.02, burst_rate=6.0, p_burst=0.012, dmin=3, dmax=12):
    a = rng.poisson(base, T).astype(np.int64)
    t = 0
    while t < T:
        if rng.random() < p_burst:
            d = int(min(rng.integers(dmin, dmax), T - t))
            if d > 0:
                a[t:t + d] += rng.poisson(burst_rate, d)
            t += d + int(rng.integers(60, 180))
        else:
            t += 1
    return a


def periodic_stream(T, rng, period_s=300, base=0.02, peak=4.0):
    rate = base + (peak - base) * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(T) / period_s))
    return rng.poisson(rate).astype(np.int64)


def diurnal_stream(T, rng, period_s=3600, base=0.01, peak=3.0):
    # Two-humped daily-ish shape compressed into the episode window.
    x = np.arange(T)
    shape = 0.5 + 0.35 * np.sin(2 * np.pi * x / period_s) + 0.25 * np.sin(4 * np.pi * x / period_s + 1.0)
    shape = np.clip(shape, 0, 1)
    rate = base + (peak - base) * shape
    return rng.poisson(rate).astype(np.int64)


def sparse_stream(T, rng, rate=0.005):
    return rng.poisson(rate, T).astype(np.int64)


def azure_like_stream(T, rng):
    """Synthetic stand-in calibrated to Shahrad et al. (2020) statistics.
    Piecewise-constant per-minute rates drawn from a heavy-tailed mixture:
    mostly near-zero (45% of apps <=1/hour) with rare heavy minutes
    (top 18.6% of apps carry 99.6% of invocations)."""
    a = np.zeros(T, dtype=np.int64)
    t = 0
    while t < T:
        w = rng.random()
        if w < 0.45:                     # cold, rarely invoked app
            rate = rng.uniform(0.0, 1.0 / 60 / 2)
            seg = int(rng.integers(300, 1200))
        elif w < 0.81:                   # <= 1/minute band
            rate = rng.uniform(1.0 / 120, 1.0 / 30)
            seg = int(rng.integers(300, 1200))
        else:                            # busy apps: heavy tail
            rate = rng.uniform(0.5, 8.0) if rng.random() < 0.186 else rng.uniform(0.02, 0.5)
            seg = int(rng.integers(60, 600))
        seg = min(seg, T - t)
        a[t:t + seg] = rng.poisson(rate, seg)
        t += seg
    return a


# --------------------------------------------------------------------------
# Azure Functions 2019 real-trace loader (use when the CSV is available)
# --------------------------------------------------------------------------
def load_azure_invocations(csv_path, function_row=0, rng=None, T=None):
    """Parse the Azure Functions 2019 'Invocations' CSV (AzurePublicDataset).

    Each data row is: HashOwner, HashApp, HashFunction, Trigger, then 1440
    per-minute invocation counts. Returns a per-second array produced by
    Poisson spreading (the proposal's stated sub-minute assumption).
    """
    import csv
    rng = rng or np.random.default_rng(0)
    with open(csv_path, "r") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        n_meta = sum(1 for h in header if not h.strip().isdigit())
        for i, row in enumerate(rdr):
            if i == function_row:
                counts = np.array([float(x) for x in row[n_meta:]], dtype=np.int64)
                break
        else:
            raise IndexError(f"function_row {function_row} beyond file")
    secs = spread_minutes_to_seconds(counts, rng)
    return secs[:T] if T else secs


# --------------------------------------------------------------------------
# Episode samplers used by the training environment
# --------------------------------------------------------------------------
TRAIN_GENERATORS = ("poisson_low", "poisson_mid", "bursty", "periodic", "diurnal", "sparse", "azure_like")
TEST_GENERATORS = ("poisson_ho", "bursty_ho", "periodic_ho", "diurnal_ho", "sparse_ho", "azure_like_ho")


def train_arrival_fn(rng: np.random.Generator, T: int = 3600):
    """One training episode: pick a generator family at random and sample."""
    g = rng.choice(len(TRAIN_GENERATORS))
    if g == 0:  return poisson_stream(T, rng, rate=rng.uniform(0.05, 0.15))
    if g == 1:  return poisson_stream(T, rng, rate=rng.uniform(0.2, 0.8))
    if g == 2:  return bursty_stream(T, rng, burst_rate=rng.uniform(3, 9))
    if g == 3:  return periodic_stream(T, rng, period_s=int(rng.uniform(180, 600)), peak=rng.uniform(2, 6))
    if g == 4:  return diurnal_stream(T, rng, peak=rng.uniform(1.5, 4))
    if g == 5:  return sparse_stream(T, rng, rate=rng.uniform(0.002, 0.01))
    return azure_like_stream(T, rng)


def regime_shift_stream(T=3600, seed=7):
    """Slide requirement: change regime mid-run -- sparse, then burst, then sparse."""
    rng = np.random.default_rng(seed)
    a = sparse_stream(T // 3, rng, rate=0.005)
    b = bursty_stream(T // 3, rng, burst_rate=8.0, p_burst=0.03)
    c = sparse_stream(T - 2 * (T // 3), rng, rate=0.005)
    return np.concatenate([a, b, c])
