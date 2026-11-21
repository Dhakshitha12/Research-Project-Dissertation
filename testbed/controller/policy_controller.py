from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from stable_baselines3 import DQN, PPO

from envs.workloads import load_function_profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("policy-controller")

FUNCTION_NAME = os.environ["FUNCTION_NAME"]
RUNTIME = os.environ["RUNTIME"]
WORKLOAD = os.environ["WORKLOAD"]
ALGORITHM = os.environ["ALGORITHM"].lower()
MEMORY_GRID_MB = [int(value) for value in os.environ["MEMORY_GRID_MB"].split(",")]
PROMETHEUS_URL = os.environ["PROMETHEUS_URL"].rstrip("/")
PROMQL = json.loads(os.environ["PROMQL_JSON"])
MODEL_PATH = os.environ["MODEL_PATH"]
INTERVAL_SECONDS = float(os.environ.get("INTERVAL_SECONDS", "5"))
NAMESPACE = os.environ.get("K8S_NAMESPACE", "openfaas-fn")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME", f"function-{FUNCTION_NAME}")
CONTAINER_NAME = os.environ.get("FUNCTION_CONTAINER_NAME", "function")

profile = load_function_profile(
    os.environ["CALIBRATION_CSV"], RUNTIME, WORKLOAD, MEMORY_GRID_MB
)
expected_queries = {
    "rate10", "rate60", "last_arrival", "variance30",
}
missing_queries = expected_queries - set(PROMQL)
if missing_queries:
    raise ValueError(f"PROMQL_JSON is missing queries: {sorted(missing_queries)}")

if ALGORITHM == "ppo":
    model = PPO.load(MODEL_PATH, device="cpu")
elif ALGORITHM == "dqn":
    model = DQN.load(MODEL_PATH, device="cpu")
else:
    raise ValueError("ALGORITHM must be 'ppo' or 'dqn'")

session_token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text()
service_ca = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
KUBERNETES_URL = "https://kubernetes.default.svc"
last_error = None
last_decision = None
last_traffic_time = time.time()
controller_started_at = time.monotonic()
app = FastAPI()


def prometheus_value(query_name: str) -> float:
    expression = PROMQL[query_name].replace("{function}", FUNCTION_NAME)
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": expression},
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query {query_name!r} failed: {payload}")
    results = payload["data"]["result"]
    if len(results) != 1:
        raise RuntimeError(
            f"Prometheus query {query_name!r} returned {len(results)} values; expected one"
        )
    value = float(results[0]["value"][1])
    if not math.isfinite(value):
        raise RuntimeError(f"Prometheus query {query_name!r} returned {value}")
    return value


def observation() -> np.ndarray:
    global last_traffic_time
    measurements = {name: prometheus_value(name) for name in expected_queries}
    _, deployment = current_deployment()
    containers = deployment["spec"]["template"]["spec"]["containers"]
    container = next(
        (item for item in containers if item["name"] == CONTAINER_NAME), None
    )
    if container is None:
        raise RuntimeError(f"Deployment has no container named {CONTAINER_NAME!r}")
    memory_limit = container.get("resources", {}).get("limits", {}).get("memory")
    if not memory_limit or not memory_limit.endswith("Mi"):
        raise RuntimeError(f"Unsupported function memory limit: {memory_limit!r}")
    current_memory = int(memory_limit[:-2])
    if current_memory not in MEMORY_GRID_MB:
        raise RuntimeError(f"Function memory {current_memory} MB is outside the trained grid")
    replicas = int(deployment["spec"].get("replicas", 1))
    ready = int(deployment.get("status", {}).get("readyReplicas", 0))
    ready_counts = [ready if memory == current_memory else 0
                    for memory in MEMORY_GRID_MB]
    starting_counts = [max(0, replicas - ready) if memory == current_memory else 0
                       for memory in MEMORY_GRID_MB]
    if measurements["last_arrival"] > 0:
        last_traffic_time = time.time()
    capacity = sum(
        count * max(1, int(1.0 / profile.constants_for(memory)[0]))
        for count, memory in zip(ready_counts, MEMORY_GRID_MB)
    )
    measurements["idle_age"] = time.time() - last_traffic_time
    measurements["utilization"] = min(
        measurements["last_arrival"], capacity
    ) / max(capacity, 1)
    measurements["capacity_deficit"] = max(
        0.0, measurements["last_arrival"] - capacity
    )
    elapsed = time.monotonic() - controller_started_at
    tod = (elapsed % 3600) / 3600.0
    return np.asarray([
        math.log1p(max(measurements["rate10"], 0.0)),
        math.log1p(max(measurements["rate60"], 0.0)),
        math.log1p(max(measurements["last_arrival"], 0.0)),
        math.log1p(max(measurements["variance30"], 0.0)),
        *[count / 10.0 for count in ready_counts],
        *[count / 10.0 for count in starting_counts],
        measurements["idle_age"] / 300.0,
        min(1.0, measurements["utilization"]),
        measurements["capacity_deficit"] / 5.0,
        math.sin(2 * math.pi * tod),
        math.cos(2 * math.pi * tod),
        profile.mem_gb,
        math.log1p(profile.exec_s),
    ], dtype=np.float32)


def kubernetes_request(method: str, path: str, body=None):
    response = requests.request(
        method,
        f"{KUBERNETES_URL}{path}",
        headers={
            "Authorization": f"Bearer {session_token}",
            "Content-Type": "application/strategic-merge-patch+json",
        },
        json=body,
        verify=service_ca,
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def current_deployment():
    path = (f"/apis/apps/v1/namespaces/{NAMESPACE}/deployments/"
            f"{DEPLOYMENT_NAME}")
    return path, kubernetes_request("GET", path)


def apply_action(action: int) -> dict:
    memory_count = len(MEMORY_GRID_MB)
    path, deployment = current_deployment()
    spec = deployment["spec"]
    replicas = int(spec.get("replicas", 1))
    template = spec["template"]["spec"]
    containers = template["containers"]
    function_container = next(
        (container for container in containers if container["name"] == CONTAINER_NAME),
        None,
    )
    if function_container is None:
        raise RuntimeError(f"Deployment has no container named {CONTAINER_NAME!r}")

    memory_mb = None
    replica_delta = 0
    if action == 1:
        replica_delta = -1
    elif 2 <= action < 2 + memory_count:
        memory_mb = MEMORY_GRID_MB[action - 2]
        replica_delta = 1
    elif 2 + memory_count <= action < 2 + 2 * memory_count:
        memory_mb = MEMORY_GRID_MB[action - 2 - memory_count]
        replica_delta = 3
    elif action != 0:
        raise ValueError(f"Model returned action {action} outside the trained action space")
    if action == 0:
        return {"action": action, "replicas": replicas}

    patch = {"spec": {"replicas": max(0, replicas + replica_delta)}}
    if memory_mb is not None:
        memory_limit = f"{memory_mb}Mi"
        patch["spec"]["template"] = {"spec": {"containers": [{
            "name": CONTAINER_NAME,
            "resources": {
                "requests": {"memory": memory_limit},
                "limits": {"memory": memory_limit},
            },
        }]}}
    kubernetes_request("PATCH", path, patch)
    return {
        "action": action,
        "replicas": max(0, replicas + replica_delta),
        "memory_mb": memory_mb,
    }


async def control_loop():
    global last_error, last_decision
    while True:
        try:
            obs = observation()
            action, _ = model.predict(obs, deterministic=True)
            last_decision = apply_action(int(action))
            last_error = None
        except Exception as error:
            last_error = str(error)
            logger.exception("Policy control iteration failed")
        await asyncio.sleep(INTERVAL_SECONDS)


@app.on_event("startup")
async def start_controller():
    if INTERVAL_SECONDS <= 0:
        raise ValueError("INTERVAL_SECONDS must be positive")
    asyncio.create_task(control_loop())


@app.get("/health")
def health():
    if last_error:
        raise HTTPException(status_code=503, detail=last_error)
    if last_decision is None:
        raise HTTPException(status_code=503, detail="No policy decision has completed")
    return {"status": "ok", "last_decision": last_decision}
