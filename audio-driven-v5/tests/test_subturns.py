import unittest
from p3_listen.subturns import cut_points


class SubturnTests(unittest.TestCase):
    def test_short_unchanged(self):
        self.assertEqual([(x["start"], x["end"]) for x in cut_points(1.1,11.1,[],[])], [(1.1,11.1)])

    def test_hard_cut_without_words(self):
        rows=cut_points(0,25,[],[])
        self.assertEqual([r["end"] for r in rows],[10,20,25])
        self.assertEqual(rows[0]["end_reason"],"hard_10s_no_word_gap")

    def test_pause_avoids_listener_event(self):
        words=[dict(start=0,end=6),dict(start=7,end=8),dict(start=9,end=16)]
        rows=cut_points(0,16,words,[dict(start=8,end=9)])
        self.assertEqual(rows[0]["end"],6.5)
        self.assertFalse(rows[0]["listener_event_cut"])

    def test_long_partition_preserves_all_time(self):
        words=[dict(start=i*.7,end=i*.7+.4) for i in range(180)]
        for duration in (10.001,19.999,20.,20.001,118.847):
            rows=cut_points(.123,.123+duration,words,[])
            self.assertAlmostEqual(sum(r["end"]-r["start"] for r in rows),duration)
            self.assertTrue(all(0<r["end"]-r["start"]<=10+1e-7 for r in rows))
            self.assertTrue(all(a["end"]==b["start"] for a,b in zip(rows,rows[1:])))


if __name__=="__main__": unittest.main()
