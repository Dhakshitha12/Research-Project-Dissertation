"""Patch envs/workloads.py DEFAULT_PROFILE with the AWS-measured medians."""
import csv, statistics, sys

cal = sys.argv[1] if len(sys.argv) > 1 else "/home/user/artifacts/data/aws_calibration.csv"
rows = list(csv.DictReader(open(cal)))
by_mem = {}
for r in rows:
    d = by_mem.setdefault(int(r["memory_mb"]), {"cold": [], "warm": []})
    d["cold" if r["kind"] == "cold" else "warm"].append(float(r["init_ms"] if r["kind"] == "cold" else r["duration_ms"]))

# pick the 512 MB profile if present, else the largest memory with both kinds
cand = [m for m, v in by_mem.items() if v["cold"] and v["warm"]]
m = 512 if 512 in cand else max(cand)
l_cold = statistics.median(by_mem[m]["cold"]) / 1000.0
exec_s = statistics.median(by_mem[m]["warm"]) / 1000.0
print(f"measured: mem={m}MB  L_cold={l_cold:.3f}s  exec_s={exec_s:.4f}s  "
      f"(n_cold={len(by_mem[m]['cold'])}, n_warm={len(by_mem[m]['warm'])})")

path = "/home/user/artifacts/code/envs/workloads.py"
src = open(path).read()
old = 'DEFAULT_PROFILE = FunctionProfile("demo-fn", mem_gb=0.5, exec_s=0.05, cold_s=0.75)'
new = (f'DEFAULT_PROFILE = FunctionProfile("demo-fn", mem_gb=0.5, exec_s={exec_s:.4f}, '
       f'cold_s={l_cold:.3f})  # AWS Lambda (us-east-1, {m} MB) measured medians, '
       f'see data/aws_calibration.csv')
assert old in src
open(path, "w").write(src.replace(old, new))
print("patched", path)
