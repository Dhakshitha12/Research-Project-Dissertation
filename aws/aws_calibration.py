"""
AWS validation / calibration stage (boto3).

Measures REAL cold-start and warm latency of AWS Lambda functions across
memory sizes, using the version-publishing trick to force cold starts
(Each published version runs on a fresh execution environment. Image-resize
and JSON-transform handlers are measured for each runtime and memory size;
the output CSV is consumed directly by the simulator.

Credentials are read directly from user.txt (an AWS credentials-file INI in
the repository root, or the path passed via --user-credentials). If user.txt
is missing or has no usable keys, the script falls back to the standard AWS
credential chain (environment variables, ~/.aws/credentials, --profile).
Values are parsed in memory and never logged, echoed, or written to output.

Usage:
  python3 aws_calibration.py --out ../data/aws_calibration.csv
  python3 aws_calibration.py --profile thesis-lab --out ../data/aws_calibration.csv
  python3 aws_calibration.py --user-credentials /path/to/user.txt --out ../data/aws_calibration.csv
  python3 aws_calibration.py --runtimes python3.12 --workloads json_transform \
      --memories 128 512 --colds 3 --warms 10
"""
from __future__ import annotations
import argparse, base64, configparser, io, json, re, sys, time, zipfile, os
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
ROLE_NAME = "prewarm-lab-lambda-role"
FN_BASE = "prewarm-calib"
USER_CREDENTIALS_FILENAME = "user.txt"
GB_S_PRICE_USD = 0.0000166667  # Lambda x86, published price (cost proxy)

REPORT_RE = {
    "duration_ms": re.compile(r"Duration: ([\d.]+) ms"),
    "init_ms": re.compile(r"Init Duration: ([\d.]+) ms"),
    "mem_mb": re.compile(r"Max Memory Used: ([\d.]+) MB"),
}
RUNTIMES = {
    "python3.12": ("handler.handler", "handler.py"),
    "nodejs20.x": ("index.handler", "index.js"),
}
WORKLOADS = ("image_resize", "json_transform")


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def handler_source(runtime: str, workload: str, nonce: str | None = None) -> str:
    if workload == "image_resize":
        if runtime == "python3.12":
            src = '''\
def handler(event, context):
    width, height = event["width"], event["height"]
    pixels = event["pixels"]
    out_width, out_height = event["out_width"], event["out_height"]
    checksum = 0
    for y in range(out_height):
        source_y = y * height // out_height
        for x in range(out_width):
            source_x = x * width // out_width
            offset = (source_y * width + source_x) * 3
            checksum += pixels[offset] + pixels[offset + 1] + pixels[offset + 2]
    return {"width": out_width, "height": out_height, "checksum": checksum}
'''
        else:
            src = '''\
exports.handler = async (event) => {
  const {width, height, pixels, out_width: ow, out_height: oh} = event;
  let checksum = 0;
  for (let y = 0; y < oh; y++) {
    const sy = Math.floor(y * height / oh);
    for (let x = 0; x < ow; x++) {
      const sx = Math.floor(x * width / ow);
      const offset = (sy * width + sx) * 3;
      checksum += pixels[offset] + pixels[offset + 1] + pixels[offset + 2];
    }
  }
  return {width: ow, height: oh, checksum};
};
'''
    elif workload == "json_transform":
        if runtime == "python3.12":
            src = '''\
def handler(event, context):
    records = event["records"]
    transformed = [
        {"id": row["id"], "value": round(row["value"] * 1.15, 2),
         "group": row["group"].strip().lower()}
        for row in records if row["active"]
    ]
    return {"count": len(transformed),
            "total": round(sum(row["value"] for row in transformed), 2)}
'''
        else:
            src = '''\
exports.handler = async (event) => {
  const transformed = event.records
    .filter(row => row.active)
    .map(row => ({id: row.id, value: Number((row.value * 1.15).toFixed(2)),
                  group: row.group.trim().toLowerCase()}));
  const total = transformed.reduce((sum, row) => sum + row.value, 0);
  return {count: transformed.length, total: Number(total.toFixed(2))};
};
'''
    else:
        raise ValueError(f"Unsupported workload: {workload}")
    return (f"# {nonce}\n" if nonce and runtime == "python3.12" else
            f"// {nonce}\n" if nonce else "") + src


def event_payload(workload: str) -> bytes:
    if workload == "image_resize":
        event = {"width": 64, "height": 64, "out_width": 32, "out_height": 32,
                 "pixels": [(i * 17) % 256 for i in range(64 * 64 * 3)]}
    elif workload == "json_transform":
        event = {"records": [
            {"id": i, "value": (i % 997) / 7, "group": f"Group-{i % 17}",
             "active": i % 3 != 0}
            for i in range(1000)
        ]}
    else:
        raise ValueError(f"Unsupported workload: {workload}")
    return json.dumps(event, separators=(",", ":")).encode()


def zip_handler(runtime: str, workload: str, nonce: str | None = None) -> bytes:
    handler, filename = RUNTIMES[runtime]
    src = handler_source(runtime, workload, nonce)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(filename, src)
    return buf.getvalue()


def parse_report(payload_b64: str) -> dict:
    raw = base64.b64decode(payload_b64).decode("utf-8", "ignore")
    out = {}
    for k, rx in REPORT_RE.items():
        m = rx.search(raw)
        if m:
            out[k] = float(m.group(1))
    out["cold"] = "Init Duration" in raw
    return out


def measure_memory(lam, role_arn, runtime, workload, mem_mb, rows, n_colds, n_warms):
    runtime_tag = runtime.replace(".", "-")
    fn = f"{FN_BASE}-{runtime_tag}-{workload}-{mem_mb}"
    handler, _ = RUNTIMES[runtime]
    payload = event_payload(workload)
    try:
        cfg = lam.get_function(FunctionName=fn)["Configuration"]
        if cfg["MemorySize"] != mem_mb or cfg["Runtime"] != runtime:
            if cfg["Runtime"] != runtime:
                raise ValueError(f"Existing calibration function {fn} has runtime {cfg['Runtime']}")
            lam.update_function_configuration(FunctionName=fn, MemorySize=mem_mb)
            lam.get_waiter("function_updated").wait(FunctionName=fn)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        log(f"Creating {fn} ({mem_mb} MB) ...")
        lam.create_function(FunctionName=fn, Runtime=runtime, Role=role_arn,
                            Handler=handler, Code={"ZipFile": zip_handler(runtime, workload)},
                            MemorySize=mem_mb, Timeout=30,
                            Description="NCI MSc cold-start calibration")
        lam.get_waiter("function_active_v2").wait(FunctionName=fn)
    time.sleep(2)

    for i in range(n_colds):  # cold start: unique code + fresh version
        lam.update_function_code(
            FunctionName=fn,
            ZipFile=zip_handler(runtime, workload, f"cold-{i}-{time.time_ns()}")
        )
        lam.get_waiter("function_updated_v2").wait(FunctionName=fn)
        ver = lam.publish_version(FunctionName=fn)["Version"]
        r = lam.invoke(FunctionName=fn, Qualifier=ver, LogType="Tail", Payload=payload)
        rep = parse_report(r["LogResult"])
        if rep.get("init_ms") is not None:
            rows.append({"runtime": runtime, "workload": workload, "memory_mb": mem_mb,
                         "kind": "cold", "duration_ms": rep["duration_ms"],
                         "init_ms": rep["init_ms"], "total_ms": rep["duration_ms"] + rep["init_ms"],
                         "ts": datetime.now(timezone.utc).isoformat()})
        log(f"  cold {runtime}/{workload} {mem_mb}MB #{i + 1}: "
            f"init={rep.get('init_ms')}ms dur={rep.get('duration_ms')}ms")
        time.sleep(1)

    for i in range(n_warms):  # warm: repeated $LATEST invocations
        r = lam.invoke(FunctionName=fn, LogType="Tail", Payload=payload)
        rep = parse_report(r["LogResult"])
        if not rep["cold"]:
            rows.append({"runtime": runtime, "workload": workload, "memory_mb": mem_mb,
                         "kind": "warm", "duration_ms": rep["duration_ms"],
                         "init_ms": 0.0, "total_ms": rep["duration_ms"],
                         "ts": datetime.now(timezone.utc).isoformat()})
        time.sleep(0.2)


def find_user_credentials(explicit_path: str | None = None) -> str | None:
    """Locate user.txt: explicit path, next to this script, repository root,
    or the current working directory. Returns the first existing path or None."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, USER_CREDENTIALS_FILENAME))
    candidates.append(os.path.join(here, os.pardir, USER_CREDENTIALS_FILENAME))
    candidates.append(os.path.join(os.getcwd(), USER_CREDENTIALS_FILENAME))
    for cand in candidates:
        cand = os.path.normpath(cand)
        if os.path.isfile(cand):
            return cand
    return None


def load_user_credentials(path: str | None, profile_name: str | None = None) -> dict:
    """Parse an AWS credentials-file INI (user.txt) into boto3.Session kwargs.

    Accepts the standard layout ([profile] sections with aws_access_key_id,
    aws_secret_access_key, and optionally aws_session_token) and also a bare
    ``key=value`` file without section headers. Returns {} when the file is
    missing, unreadable, or has no access key id + secret key pair, so the
    caller can fall back to the standard AWS credential chain. Values are
    never logged.
    """
    if not path or not os.path.isfile(path):
        return {}
    sections = {}
    try:
        parser = configparser.RawConfigParser(strict=False)
        with open(path, "r", encoding="utf-8-sig") as stream:
            parser.read_file(stream)
        for section in parser.sections():
            sections[section.strip()] = {k.strip(): v.strip()
                                         for k, v in parser.items(section)}
    except (configparser.Error, OSError, UnicodeDecodeError):
        # Loose fallback: regex parse key=value pairs (tolerates stray text).
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as stream:
                text = stream.read()
        except OSError:
            return {}
        current = "default"
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            header = re.match(r"^\[([^\]]+)\]$", line)
            if header:
                current = header.group(1).strip()
                sections.setdefault(current, {})
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                sections.setdefault(current, {})[key.strip()] = value.strip()

    wanted = (profile_name or "default").strip()
    section = sections.get(wanted)
    if section is None and len(sections) == 1:
        section = next(iter(sections.values()))  # single unnamed/renamed profile
    if not section:
        return {}

    out = {}
    for key in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
        value = section.get(key)
        if value:
            out[key] = value
    region = section.get("region") or section.get("aws_region")
    if region:
        out["region"] = region
    if "aws_access_key_id" not in out or "aws_secret_access_key" not in out:
        return {}
    return out


def get_exec_role(session) -> str:
    """Return a Lambda-executable role ARN. Lab (voclabs) sessions cannot
    create IAM roles, so reuse an existing role: prefer 'voclabs', then any
    role whose trust includes lambda.amazonaws.com, else fail informatively."""
    iam = session.client("iam")
    for name in ("voclabs", "LabRole", "labRole"):
        try:
            arn = iam.get_role(RoleName=name)["Role"]["Arn"]
            log(f"Using existing role {name} as Lambda execution role")
            return arn
        except ClientError:
            continue
    try:
        for r in iam.list_roles().get("Roles", []):
            trust = json.dumps(r.get("AssumeRolePolicyDocument", {}))
            if "lambda.amazonaws.com" in trust:
                log(f"Using existing role {r['RoleName']} as Lambda execution role")
                return r["Arn"]
    except ClientError:
        pass
    raise SystemExit(
        "No usable Lambda execution role found and iam:CreateRole is denied "
        "(expected in lab sessions). Pass a role ARN via --role.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/aws_calibration.csv")
    ap.add_argument("--user-credentials", default=None,
                    help="Path to user.txt (AWS credentials-file INI); "
                         "default: ./user.txt or the repository root")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"),
                    help="Profile: selects the [section] inside user.txt; "
                         "used as an AWS CLI profile only when falling back "
                         "to the standard credential chain")
    ap.add_argument("--role", default=None, help="Existing role name/ARN to use as Lambda execution role")
    ap.add_argument("--runtimes", nargs="+", choices=RUNTIMES, default=list(RUNTIMES))
    ap.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    ap.add_argument("--memories", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--colds", type=int, default=8)
    ap.add_argument("--warms", type=int, default=20)
    args = ap.parse_args()

    user_creds_path = find_user_credentials(args.user_credentials)
    user_creds = load_user_credentials(user_creds_path, args.profile) if user_creds_path else {}
    if user_creds:
        log(f"Using AWS credentials from {user_creds_path} "
            f"(profile '{args.profile or 'default'}'); values are never logged")
        region = user_creds.pop("region", None) or REGION
        session = boto3.Session(region_name=region, **user_creds)
    else:
        if user_creds_path:
            log(f"{user_creds_path} has no usable AWS access keys; "
                f"falling back to the standard credential chain")
        session = boto3.Session(profile_name=args.profile, region_name=REGION)
    ident = session.client("sts").get_caller_identity()
    log(f"Authenticated as {ident['Arn']} (account {ident['Account']})")

    role_arn = args.role or get_exec_role(session)
    if not role_arn.startswith("arn:"):
        role_arn = f"arn:aws:iam::{ident['Account']}:role/{role_arn}"

    lam = session.client("lambda", region_name=REGION)
    rows = []
    for runtime in args.runtimes:
        for workload in args.workloads:
            for mem in args.memories:
                measure_memory(lam, role_arn, runtime, workload, mem, rows,
                               args.colds, args.warms)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["runtime", "workload", "memory_mb", "kind",
                                          "duration_ms", "init_ms", "total_ms", "ts"])
        w.writeheader()
        w.writerows(rows)
    log(f"Wrote {len(rows)} measurements -> {args.out}")

    import statistics as st
    for runtime in args.runtimes:
        for workload in args.workloads:
            for mem in args.memories:
                colds = [r["init_ms"] for r in rows if r["runtime"] == runtime
                         and r["workload"] == workload and r["memory_mb"] == mem
                         and r["kind"] == "cold"]
                warms = [r["duration_ms"] for r in rows if r["runtime"] == runtime
                         and r["workload"] == workload and r["memory_mb"] == mem
                         and r["kind"] == "warm"]
                if colds and warms:
                    log(f"SUMMARY {runtime}/{workload} {mem}MB: "
                        f"L_cold={st.median(colds) / 1000:.3f}s  "
                        f"exec_s={st.median(warms) / 1000:.3f}s  "
                        f"n_cold={len(colds)} n_warm={len(warms)}")


if __name__ == "__main__":
    try:
        main()
    except ClientError as e:
        log(f"AWS error: {e}")
        sys.exit(1)
