import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_cider_calibration import build_package, read_standard_queries


class CiderCalibrationChecks(unittest.TestCase):
    def make_repo(self, root: Path):
        (root / "data/img/clean").mkdir(parents=True)
        (root / "data/img/valset").mkdir(parents=True)
        (root / "data/img/testset").mkdir(parents=True)
        (root / "data/text").mkdir(parents=True)
        (root / "LICENSE").write_text("test license", encoding="utf-8")
        for relative, color in (
            ("data/img/clean/0.jpg", (1, 2, 3)),
            ("data/img/valset/clean_0.jpg", (4, 5, 6)),
            ("data/img/valset/prompt_constrained_0.jpg", (7, 8, 9)),
            ("data/img/testset/clean_1.jpg", (10, 11, 12)),
            ("data/img/testset/prompt_constrained_1.jpg", (13, 14, 15)),
        ):
            Image.new("RGB", (8, 8), color).save(root / relative)
        for name, prefix in (("valset.csv", "v"), ("testset.csv", "t")):
            with (root / "data/text" / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([f"{prefix} standard", "standard", "harmful", "", "", "", f"{prefix}-id"])
                writer.writerow([f"{prefix} ignored", "copyright", "copyright", "", "", "", f"{prefix}-ignored"])

    def test_query_filter_and_package(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            repo, output = base / "repo", base / "out"
            self.make_repo(repo)
            output.mkdir()
            self.assertEqual(len(read_standard_queries(repo / "data/text/valset.csv")), 1)
            result = build_package(repo, output, "fixture")
            self.assertEqual(result["sample_count"], 5)
            self.assertEqual(result["unique_image_count"], 5)
            self.assertEqual(result["label_counts"]["validation:0"], 1)
            self.assertEqual(result["label_counts"]["validation:1"], 1)
            rows = [json.loads(line) for line in (output / "fixture/manifest.inputs.jsonl").read_text().splitlines()]
            self.assertEqual({row["split"] for row in rows}, {"threshold_clean", "validation", "test"})
            with (output / "fixture/labels_for_B.csv").open(encoding="utf-8-sig") as stream:
                labels = list(csv.DictReader(stream))
            self.assertEqual({row["image_adv_label"] for row in labels}, {"0", "1"})
            self.assertTrue((output / "fixture.zip").is_file())


if __name__ == "__main__":
    unittest.main()
