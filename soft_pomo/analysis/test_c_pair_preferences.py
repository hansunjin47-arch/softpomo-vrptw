import unittest
from audit_c_pair_preferences import reversed_routes


class PairTests(unittest.TestCase):
    def test_adjacent_has_one_candidate(self):
        self.assertEqual(reversed_routes([1, 2, 3, 4], 1, 2), [(1, 3, 2, 4)])

    def test_candidates_reverse_pair_and_preserve_coverage(self):
        route = [1, 2, 3, 4, 5]
        for i in range(len(route)):
            for j in range(i+1, len(route)):
                for candidate in reversed_routes(route, i, j):
                    self.assertEqual(sorted(candidate), route)
                    self.assertLess(candidate.index(route[j]), candidate.index(route[i]))


if __name__ == '__main__':
    unittest.main()
