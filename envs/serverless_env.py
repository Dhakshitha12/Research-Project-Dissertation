"""
Trace-driven pre-warming MDP environment.

Implements the MDP of Tables 2/4 of the proposal exactly:

  Time step   : one second (configurable dt).
  State       : arrival statistics, per-memory ready/starting pool sizes,
                idle age, utilisation, capacity deficit, time-of-day sin/cos.
  Actions     : hold; release; pre-warm one or three instances at a selected
                calibrated memory size.
  Transition  : trace-driven. Ready instances serve requests up to capacity;
                excess requests incur a cold start and trigger instance
                creation; a starting instance becomes ready after the
                cold-start delay; a platform keep-alive (300 s) applies to
                every policy.
  Reward      : r_t = -( w_L * C_t * L_cold + lambda * N_t )
  Monetary    : GB-seconds x price per GB-second.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .workloads import FunctionProfile, GB_S_PRICE_USD


class PreWarmEnv(gym.Env):
    """Gymnasium environment for per-function pre-warming."""

    metadata = {"render_modes": []}

    def __init__(self,
                 profile: FunctionProfile,
                 lam: float,
                 w_L: float = 1.0,
                 arrivals: np.ndarray | None = None,
                 arrival_fn=None,
                 T: int = 3600,
                 cap_per_instance: float = 1.0,
                 keep_alive_s: int = 300,
                 seed: int | None = None):
        super().__init__()
        self.profile = profile
        self.lam = float(lam)
        self.w_L = float(w_L)
        self._arrivals_arg = arrivals
        self.arrival_fn = arrival_fn
        self.T = int(T)
        self.cap = float(cap_per_instance)
        self.keep_alive_s = int(keep_alive_s)
        self.dt = 1.0
        self._seed0 = seed

        self.memory_grid_mb = profile.memory_grid_mb
        self.default_memory_mb = 512 if 512 in self.memory_grid_mb else self.memory_grid_mb[0]
        self.action_space = spaces.Discrete(2 + 2 * len(self.memory_grid_mb))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(11 + 2 * len(self.memory_grid_mb),),
            dtype=np.float32
        )

    # ------------------------------------------------------------------ obs
    def _obs(self):
        h = np.array(self._hist[-60:] if len(self._hist) >= 60 else self._hist, dtype=float)
        h10 = np.array(self._hist[-10:] if len(self._hist) >= 10 else self._hist, dtype=float)
        rate10, rate60 = h10.mean() if h10.size else 0.0, h.mean() if h.size else 0.0
        var30 = float(np.var(self._hist[-30:])) if len(self._hist) >= 2 else 0.0
        ready_counts = [sum(memory == m for _, memory in self.ready) for m in self.memory_grid_mb]
        starting_counts = [sum(memory == m for _, memory in self.starting)
                           for m in self.memory_grid_mb]
        idle_age = (self._t - self.idle_since[0][0]) if self.idle_since else 0.0
        util = self._last_util
        deficit = max(0.0, self._last_arr - self.ready_capacity)
        tod = (self._t % 3600) / 3600.0
        return np.array([
            np.log1p(rate10), np.log1p(rate60), np.log1p(self._last_arr),
            np.log1p(var30),
            *[n / 10.0 for n in ready_counts],
            *[n / 10.0 for n in starting_counts],
            idle_age / self.keep_alive_s,
            util, deficit / 5.0, np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
            self.profile.mem_gb, np.log1p(self.profile.exec_s),
        ], dtype=np.float32)

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed if seed is not None else self._seed0)
        if options and "arrivals" in options:
            self.arrivals = np.asarray(options["arrivals"], dtype=np.int64)
        elif self._arrivals_arg is not None:
            self.arrivals = np.asarray(self._arrivals_arg, dtype=np.int64)
        elif self.arrival_fn is not None:
            self.arrivals = np.asarray(self.arrival_fn(self.rng), dtype=np.int64)
        else:
            self.arrivals = np.zeros(self.T, dtype=np.int64)
        self.T = max(self.T, len(self.arrivals))
        self._t = 0
        self._hist = []
        self.ready = []        # (ready_at, memory_mb)
        self.starting = []     # (seconds_remaining, memory_mb)
        self.idle_since = []
        self._selected_memory_mb = self.default_memory_mb
        self._last_arr = 0.0
        self._last_util = 0.0
        self._cold_total = 0
        self._arr_total = 0
        self._reward_sum = 0.0
        return self._obs(), {}

    # ----------------------------------------------------------------- step
    def step(self, action):
        t = self._t
        action = int(action)
        memory_count = len(self.memory_grid_mb)
        if action == 1:
            if self.idle_since:
                self.idle_since.pop()
                self.ready.pop()
        elif 2 <= action < 2 + memory_count:
            self._selected_memory_mb = self.memory_grid_mb[action - 2]
            self.starting.append((self.profile.constants_for(self._selected_memory_mb)[1],
                                  self._selected_memory_mb))
        elif 2 + memory_count <= action < 2 + 2 * memory_count:
            self._selected_memory_mb = self.memory_grid_mb[action - 2 - memory_count]
            cold_s = self.profile.constants_for(self._selected_memory_mb)[1]
            self.starting.extend([(cold_s, self._selected_memory_mb)] * 3)
        elif action != 0:
            raise ValueError(f"Action {action} is outside the configured action space")

        # --- arrivals are served by ready instances ---
        arr = int(self.arrivals[t]) if t < len(self.arrivals) else 0
        capacity_by_memory = {}
        for _, memory_mb in self.ready:
            exec_s, _ = self.profile.constants_for(memory_mb)
            capacity_by_memory[memory_mb] = capacity_by_memory.get(memory_mb, 0) + max(
                1, int(self.cap / exec_s)
            )
        capacity = sum(capacity_by_memory.values())
        served = min(arr, capacity)
        cold = int(arr - served)

        # cold requests trigger instance creation (becomes ready after delay)
        for _ in range(cold):
            cold_s = self.profile.constants_for(self._selected_memory_mb)[1]
            self.starting.append((cold_s, self._selected_memory_mb))

        cold_latencies = []
        if cold:
            exec_s, cold_s = self.profile.constants_for(self._selected_memory_mb)
            cold_latencies = [cold_s + exec_s] * cold
        warm_latencies = []
        remaining_served = served
        for memory_mb in sorted(capacity_by_memory,
                                key=lambda m: self.profile.constants_for(m)[0]):
            exec_s, _ = self.profile.constants_for(memory_mb)
            count = min(remaining_served, capacity_by_memory[memory_mb])
            warm_latencies.extend([exec_s] * count)
            remaining_served -= count
            if remaining_served == 0:
                break
        if served:
            self.idle_since = [(t, memory) for _, memory in self.idle_since]

        # --- bookkeeping: starting -> ready, keep-alive expiry ---
        still_starting = []
        for rem, memory_mb in self.starting:
            rem -= self.dt
            if rem <= 0:
                self.ready.append((t, memory_mb))
                self.idle_since.append((t, memory_mb))
            else:
                still_starting.append((rem, memory_mb))
        self.starting = still_starting

        keep_ready, keep_idle = [], []
        for ready, idle in zip(self.ready, self.idle_since):
            if t - idle[0] < self.keep_alive_s:
                keep_ready.append(ready)
                keep_idle.append(idle)
        self.ready, self.idle_since = keep_ready, keep_idle

        held = len(self.ready) + len(self.starting)
        gb_seconds = sum(memory for _, memory in self.ready) / 1024
        gb_seconds += sum(memory for _, memory in self.starting) / 1024
        reward = -(self.w_L * sum(cold_latencies) +
                   self.lam * gb_seconds / 0.5)
        self._reward_sum += reward

        self._last_util = served / max(capacity, 1)
        self._last_arr = float(arr)
        self._hist.append(arr)
        self._cold_total += cold
        self._arr_total += arr

        info = {"colds": cold, "arrivals": arr, "held": held, "gb_seconds": gb_seconds,
                "served": served, "cold_latencies": cold_latencies,
                "warm_latencies": warm_latencies, "reward": reward, "t": t}
        self._t = t + 1
        terminated = self._t >= self.T
        return self._obs(), float(reward), bool(terminated), False, info

    @property
    def ready_capacity(self):
        return sum(max(1, int(self.cap / self.profile.constants_for(memory)[0]))
                   for _, memory in self.ready)

    def action_for_add(self, count: int, memory_mb: int | None = None) -> int:
        if count not in (1, 3):
            raise ValueError("Pre-warm count must be 1 or 3")
        memory_mb = self.default_memory_mb if memory_mb is None else memory_mb
        try:
            memory_index = self.memory_grid_mb.index(memory_mb)
        except ValueError as exc:
            raise ValueError(f"Memory size {memory_mb} MB is not in the action grid") from exc
        return 2 + memory_index + (len(self.memory_grid_mb) if count == 3 else 0)

    # ------------------------------------------------------- cost helpers
    def dollar_cost(self, gb_seconds: float) -> float:
        """GB-seconds x published price per GB-second."""
        return gb_seconds * GB_S_PRICE_USD


def run_policy_on_arrivals(policy, arrivals, profile, lam, w_L=1.0, keep_alive_s=300):
    """Run any policy (RL model with .predict, or baseline .act(env)) on a
    FIXED arrival array -- the paired design of the proposal (every policy
    meets identical arrivals). Returns a metrics dict."""
    env = PreWarmEnv(profile, lam=lam, w_L=w_L, keep_alive_s=keep_alive_s)
    obs, _ = env.reset(options={"arrivals": np.asarray(arrivals)})
    colds, arrs, cold_latencies, warm_latencies = [], [], [], []
    held_N, gb_seconds, rew_sum = 0, 0.0, 0.0
    done = False
    while not done:
        if hasattr(policy, "predict"):                    # SB3 model
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)
        elif hasattr(policy, "act"):                      # baseline
            action = policy.act(env)
        else:
            action = int(policy)
        obs, r, terminated, truncated, info = env.step(action)
        colds.append(info["colds"]); arrs.append(info["arrivals"])
        cold_latencies.extend(info["cold_latencies"])
        warm_latencies.extend(info["warm_latencies"])
        held_N += info["held"]; gb_seconds += info["gb_seconds"]; rew_sum += r
        done = terminated or truncated
    colds = np.array(colds); arrs = np.array(arrs)
    total_arr = arrs.sum()
    inst_seconds = held_N * 1.0
    lat = np.asarray(cold_latencies + warm_latencies) if total_arr else np.array([0.0])
    p95 = float(np.percentile(lat, 95)); p99 = float(np.percentile(lat, 99))
    return {
        "cold_rate": float(colds.sum() / total_arr) if total_arr else 0.0,
        "cold_count": int(colds.sum()),
        "requests": int(total_arr),
        "p95_s": p95, "p99_s": p99,
        "instance_seconds": float(inst_seconds),
        "gb_seconds": float(gb_seconds),
        "cost_usd": float(gb_seconds * GB_S_PRICE_USD),
        "utilisation": float(arrs.sum() / max(held_N, 1)),
        "reward_sum": float(rew_sum),
    }
