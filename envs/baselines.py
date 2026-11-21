"""The three baseline policies, using the environment's memory-aware actions."""
from __future__ import annotations


class NoPreWarm:
    """On-demand instantiation only; platform keep-alive still applies."""
    name = "no_prewarm"

    def act(self, env):
        return 0


class StaticWarm:
    """Fixed number K of warm instances kept topped up (Lin and Glikson, 2019)."""
    name = "static_warm"

    def __init__(self, K: int = 1):
        self.K = K

    def act(self, env):
        n = len(env.ready) + len(env.starting)
        if n < self.K:
            return env.action_for_add(1)
        if len(env.ready) > self.K:
            return 1
        return 0


class ThresholdPolicy:
    """Reactive target-tracking on request rate versus warm capacity, with a
    scale-down stabilisation window of 5 minutes (Shahrad et al., 2020 style)."""
    name = "threshold"

    def __init__(self, target_util: float = 0.8, scale_down_after_s: int = 300):
        self.target = target_util
        self.down_after = scale_down_after_s
        self._over_since = None

    def act(self, env):
        hist = env._hist[-60:]
        forecast = max(sum(hist) / max(len(hist), 1), 0.0)
        capacity = env.ready_capacity
        n = len(env.ready) + len(env.starting)
        # scale-up: cover the forecast with headroom
        if forecast > capacity * self.target:
            deficit = forecast - capacity
            unit_capacity = max(1, int(env.cap /
                                       env.profile.constants_for(env.default_memory_mb)[0]))
            return env.action_for_add(3 if deficit > unit_capacity else 1)
        # scale-down: only after the stabilisation window of sustained slack
        if n > 0 and forecast < capacity * self.target * 0.5:
            if self._over_since is None:
                self._over_since = env._t
            elif env._t - self._over_since >= self.down_after:
                self._over_since = None
                return 1
        else:
            self._over_since = None
        return 0
