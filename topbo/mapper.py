"""Mapper graph over the surrogate-predicted feasible set.

The filter function is the feasibility score h(x). Its range is covered by
overlapping intervals; points in each interval are clustered with DBSCAN; one
graph node per cluster, with an edge wherever two clusters share a point. The
connected components of that graph are the regions the acquisition reasons
about, and their count is the estimate of beta0.
"""

import numpy as np
import kmapper as km
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


class MapperResult:
    """Regions extracted from one Mapper graph.

    components hold indices into the near-feasible set, node_positions are in
    the normalized space used for distance comparisons, and centroids are in
    the original parameter space so local candidates can be drawn around them.
    """

    def __init__(self, graph, components, centroids, beta0,
                 node_members, node_positions, nx_graph):
        self.graph = graph
        self.components = components
        self.centroids = centroids
        self.beta0 = beta0
        self.node_members = node_members
        self.node_positions = node_positions
        self.nx_graph = nx_graph


class MapperAnalyzer:

    def __init__(self, n_cubes=10, overlap=0.3, min_cluster_size=3,
                 dbscan_eps=0.3, eps_multiplier=1.0):
        """
        Args:
            n_cubes: Number of intervals covering the range of the filter.
            overlap: Overlap fraction between adjacent intervals.
            min_cluster_size: DBSCAN min_samples within one interval.
            dbscan_eps: DBSCAN eps within one interval.
            eps_multiplier: Scales the bridge/frontier threshold epsilon*.
        """
        self.n_cubes = n_cubes
        self.overlap = overlap
        self.min_cluster_size = min_cluster_size
        self.dbscan_eps = dbscan_eps
        self.eps_multiplier = eps_multiplier
        self.mapper = km.KeplerMapper(verbose=0)

    def analyze(self, X_near_feasible, h_values, X_original=None):
        """Build the Mapper graph and extract its connected components.

        Args:
            X_near_feasible: (m, d) near-feasible points, normalized to [0,1]^d.
            h_values: (m,) filter values for those points.
            X_original: (m, d) same points in original units, used for centroids.

        Returns:
            MapperResult, or None when the graph cannot be built.
        """
        if X_near_feasible.shape[0] < self.min_cluster_size * 2:
            return None

        X_orig = X_original if X_original is not None else X_near_feasible

        graph = self.mapper.map(
            h_values.reshape(-1, 1),
            X_near_feasible,
            cover=km.Cover(n_cubes=self.n_cubes, perc_overlap=self.overlap),
            clusterer=DBSCAN(eps=self.dbscan_eps, min_samples=self.min_cluster_size),
        )
        if not graph['nodes']:
            return None

        G = nx.Graph()
        node_members = {}
        node_positions = {}
        for node_id, members in graph['nodes'].items():
            G.add_node(node_id)
            node_members[node_id] = np.array(members)
            node_positions[node_id] = X_near_feasible[members].mean(axis=0)

        for src, targets in graph['links'].items():
            for tgt in targets:
                G.add_edge(src, tgt)

        components, centroids = [], []
        for cc in nx.connected_components(G):
            idx = set()
            for node_id in cc:
                idx.update(node_members[node_id].tolist())
            idx = np.array(sorted(idx))
            components.append(idx)
            centroids.append(X_orig[idx].mean(axis=0) if len(idx)
                             else np.zeros(X_orig.shape[1]))

        return MapperResult(
            graph=graph,
            components=components,
            centroids=np.array(centroids) if centroids
            else np.zeros((0, X_orig.shape[1])),
            beta0=len(components),
            node_members=node_members,
            node_positions=node_positions,
            nx_graph=G,
        )

    def classify_candidates(self, X_cand, mapper_result, X_near_feasible):
        """Topological sensitivity of each candidate, Section 3.3.4.

        A candidate within epsilon* of two or more regions is a bridge and one
        further than epsilon* from every region is a frontier; both may change
        the region count, so both score 1. A candidate close to exactly one
        region is interior and scores 0. epsilon* is the median distance
        between Mapper nodes, which adapts to the current graph rather than
        being tuned per circuit.

        Args:
            X_cand: (n, d) candidates, normalized to [0,1]^d.
            mapper_result: MapperResult from analyze().
            X_near_feasible: (m, d) near-feasible points, normalized.

        Returns:
            (n,) array of 0/1 scores.
        """
        n_cand = X_cand.shape[0]
        delta_beta0 = np.zeros(n_cand)
        K = mapper_result.beta0
        if K < 1:
            return delta_beta0

        positions = np.array(list(mapper_result.node_positions.values()))
        if len(positions) >= 2:
            node_dists = cdist(positions, positions)
            np.fill_diagonal(node_dists, np.inf)
            eps = np.median(node_dists[np.isfinite(node_dists)])
        else:
            eps = 0.5
        eps *= self.eps_multiplier

        comp_dists = np.full((n_cand, K), np.inf)
        for k, idx in enumerate(mapper_result.components):
            if len(idx) == 0:
                continue
            points = X_near_feasible[idx]
            # Large components are represented by their centroid; the exact
            # minimum distance would cost O(n_cand * |component|).
            if len(points) > 100:
                comp_dists[:, k] = cdist(X_cand, points.mean(axis=0,
                                                             keepdims=True)).ravel()
            else:
                comp_dists[:, k] = cdist(X_cand, points).min(axis=1)

        nearby = (comp_dists < eps).sum(axis=1)
        delta_beta0[(nearby >= 2) | (nearby == 0)] = 1.0
        return delta_beta0
