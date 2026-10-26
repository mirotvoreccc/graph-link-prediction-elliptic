import unittest

import networkx as nx

from src.experiment import pair_features, sample_masked_edges


class SamplingTests(unittest.TestCase):
    def test_hidden_edges_and_nonedges_are_disjoint(self):
        graph = nx.cycle_graph(12)
        context, positives, negatives = sample_masked_edges(graph, seed=42)
        self.assertEqual(len(positives), len(negatives))
        self.assertTrue(all(not context.has_edge(*edge) for edge in positives))
        self.assertTrue(all(not graph.has_edge(*edge) for edge in negatives))
        self.assertEqual(len(set(negatives)), len(negatives))

    def test_features_use_context_graph(self):
        graph = nx.Graph([(0, 1), (1, 2), (0, 2)])
        self.assertEqual(pair_features(graph, [(0, 2)])[0, 0], 1)
        graph.remove_edge(0, 1)
        self.assertEqual(pair_features(graph, [(0, 2)])[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
