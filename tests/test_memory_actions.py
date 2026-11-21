import os
import unittest

import numpy as np

from envs.serverless_env import PreWarmEnv
from envs.workloads import FunctionProfile, load_function_profile


class MemoryActionTests(unittest.TestCase):
    def setUp(self):
        self.profile = FunctionProfile(
            name="test",
            mem_gb=0.5,
            exec_s=0.05,
            cold_s=0.08,
            runtime="python3.12",
            workload="test",
            memory_profiles={
                128: (0.2, 0.1),
                512: (0.05, 0.08),
            },
        )

    def test_action_and_observation_dimensions_follow_memory_grid(self):
        env = PreWarmEnv(self.profile, lam=0.02, T=2)
        observation, _ = env.reset(options={"arrivals": np.array([0, 1])})

        self.assertEqual(env.action_space.n, 6)
        self.assertEqual(observation.shape, (15,))
        self.assertEqual(env.action_for_add(1, 128), 2)
        self.assertEqual(env.action_for_add(1, 512), 3)
        self.assertEqual(env.action_for_add(3, 512), 5)

    def test_memory_actions_change_instance_latency_and_cost(self):
        env = PreWarmEnv(self.profile, lam=0.02, T=2)
        env.reset(options={"arrivals": np.array([0, 1])})
        env.step(env.action_for_add(1, 128))
        _, _, _, _, info = env.step(0)
        self.assertEqual(info["warm_latencies"], [0.2])

        env = PreWarmEnv(self.profile, lam=0.02, T=2)
        env.reset(options={"arrivals": np.array([0, 1])})
        env.step(env.action_for_add(1, 512))
        _, _, _, _, info = env.step(0)
        self.assertEqual(info["warm_latencies"], [0.05])
        self.assertGreater(info["gb_seconds"], 0.125)

    def test_calibration_loader_rejects_missing_memory_measurements(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "aws_calibration.csv")
        with self.assertRaisesRegex(ValueError, "Missing cold and warm calibration"):
            load_function_profile(path, memory_grid_mb=[128, 256, 512])

    def test_calibration_loader_reports_unknown_workload(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "aws_calibration.csv")
        with self.assertRaisesRegex(ValueError, "No calibration samples found"):
            load_function_profile(path, workload="image_resize")


if __name__ == "__main__":
    unittest.main()
