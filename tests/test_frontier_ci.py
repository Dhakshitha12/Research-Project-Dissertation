import os
import tempfile
import unittest

import pandas as pd

from evaluate import frontier


class FrontierConfidenceIntervalTests(unittest.TestCase):
    def test_frontier_exports_bootstrap_intervals_for_each_policy(self):
        rows = []
        for seed in range(5):
            for family in ("poisson", "bursty"):
                for episode in range(2):
                    rows.append({
                        "kind": "rl",
                        "algo": "ppo",
                        "lam": 0.02,
                        "seed": seed,
                        "family": family,
                        "episode": episode,
                        "cold_rate": 0.1 + seed * 0.01,
                        "cost_usd": 0.01 + seed * 0.001,
                        "p99_s": 0.2,
                        "instance_seconds": 100 + seed,
                        "gb_seconds": 50 + seed,
                    })
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as output:
            result = frontier(frame, output)

            self.assertEqual(len(result), 1)
            for metric in ("cold_rate", "cost_usd", "p99_s", "gb_seconds"):
                self.assertIn(f"{metric}_ci_low", result.columns)
                self.assertIn(f"{metric}_ci_high", result.columns)
                self.assertLessEqual(result.loc[0, f"{metric}_ci_low"],
                                     result.loc[0, f"{metric}_ci_high"])
            self.assertTrue(os.path.isfile(os.path.join(output, "frontier.png")))
            self.assertTrue(os.path.isfile(
                os.path.join(output, "frontier_points.csv")
            ))


if __name__ == "__main__":
    unittest.main()
