import copy
import unittest

from p3_listen.dataset import connected_groups, split_groups, windows, validate
from p3_listen.prompts import contract


class DataTests(unittest.TestCase):
    def test_shared_participant_merges_sessions_transitively(self):
        def pair(i, session, people):
            return dict(pair=str(i), sources=[dict(id=f"naturalistic_V00_S{session:04d}_I00001_P{p:04d}") for p in people],
                        clips=[dict(duration=5, kind="quiet")])
        groups = connected_groups([pair(1, 1, [1, 2]), pair(2, 2, [2, 3]), pair(3, 2, [4, 5]), pair(4, 3, [6, 7])])
        self.assertEqual(sorted(len(g["pairs"]) for g in groups), [1, 3])

    def test_windows_cover_tail_without_crossing_floor(self):
        for duration in (1.021, 4, 4.01, 8.04, 8.05, 9.006, 118.847):
            rows = windows(1.123, 1.123 + duration)
            self.assertTrue(rows)
            self.assertEqual(rows[0]["start"], 1.123)
            self.assertLess(1.123 + duration - rows[-1]["end"], .04 + 1e-7)
            for r in rows:
                self.assertEqual((r["frames"] - 1) % 8, 0)
                self.assertLessEqual(r["end"], 1.123 + duration + 1e-7)
            for a, b in zip(rows, rows[1:]):
                self.assertLessEqual(b["start"], a["end"])

    def test_split_reproducible_and_isolated(self):
        groups = [dict(group_id=str(i), sessions=[str(i)], participants=[str(i)], pairs=[str(i)],
                       seconds=i + 20, candidate_counts={"quiet": 1, "feedback": i % 3}) for i in range(14)]
        a = split_groups(groups, 20261021)
        self.assertEqual(a, split_groups(groups, 20261021))
        self.assertEqual([g["split"] for g in a].count("train"), 10)
        validate(dict(groups=a, windows=[]))
        broken = copy.deepcopy(a)
        dev = next(g for g in broken if g["split"] == "dev")
        train = next(g for g in broken if g["split"] == "train")
        dev["participants"].append(train["participants"][0])
        with self.assertRaises(ValueError):
            validate(dict(groups=broken, windows=[]))

    def test_fixed_prompt_has_no_example_argument(self):
        self.assertEqual(contract(), contract())
        self.assertNotIn("Reply:", contract()["omni"])


if __name__ == "__main__":
    unittest.main()
