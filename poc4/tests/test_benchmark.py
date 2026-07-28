from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coloriage_lot3.benchmark import run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_keeps_full_label_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark"
            report = run_benchmark(output, assert_quality=True)
            self.assertTrue((output / "report.json").is_file())
            self.assertFalse(report["violations"])
            for fixture in report["fixtures"].values():
                self.assertEqual(fixture["v1"]["skipped_count"], 0)
                self.assertEqual(fixture["v1"]["coverage_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
