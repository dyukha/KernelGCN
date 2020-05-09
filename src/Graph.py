from __future__ import annotations

import math
from typing import List, Dict, Optional, Callable

import mxnet as mx
from mxnet import nd, Context
from mxnet.ndarray import NDArray

from common import data_ctx

class Vertex:
    def __init__(self, features: Optional[NDArray], neighbors: List[Vertex], clazz: int, id: int):
        self.features = features
        self.neighbors = neighbors
        self.clazz = clazz
        self.id: int = id
        """ Vertex id. It must hold that graph.vertices[i] = v <=> v.id = i """
        self.degree: int = -1
        """ Vertex degree. Must be initialized manually """

    def __str__(self):
        return f"Vertex(class={self.clazz}, #neighbors={len(self.neighbors)}, features={self.features})"

class Graph:
    def __init__(self, vertices: List[Vertex], num_features: int, num_classes: int):
        self.vertices = vertices
        self.num_features = num_features
        self.num_classes = num_classes
        self.n = len(vertices)

    def adj_matrix(self, context: Context, multiplier: Callable[[Vertex, Vertex], float]) -> NDArray:
        """
        Returns adjacency matrix as a sparse array, where (i,j)-th entry is also multiplied by multiplier(i,j)

        :param context: context of the created matrix
        :param multiplier: multiplier of (i,j)-th entry
        :return: adjacency matrix
        """
        adj_shape = (self.n, self.n)
        adj_data = [multiplier(u, v) for u in self.vertices for v in u.neighbors]
        adj_indices = [v.id for u in self.vertices for v in u.neighbors]
        adj_indptr = [0] + [u.degree for u in self.vertices]
        for i in range(1, len(adj_indptr)):
            adj_indptr[i] += adj_indptr[i - 1]
        return mx.ndarray.sparse.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape, ctx=context)

    def linear_convolution(self, features: NDArray, depth: int, neighbor_coef: float, concatenate: bool):
        assert features.shape[0] == self.n

        if depth == 0:
            return features

        # adj = self.adj_matrix(features.context, lambda u, v: 1 / math.sqrt(u.degree * v.degree))
        adj = self.adj_matrix(features.context, lambda u, v: 1 / u.degree)

        feature_layers = [features]

        for i in range(depth):
            feature_layers.append(nd.dot(adj, feature_layers[-1]))
            if not concatenate:
                feature_layers = [feature_layers[0] + neighbor_coef * feature_layers[1]]

        return nd.concat(*feature_layers, dim=1) if concatenate else feature_layers[-1]

    def copy(self) -> Graph:
        new_vertices = [Vertex(v.features[:], [], v.clazz, v.id) for v in self.vertices]
        for i in range(self.n):
            new_vertices[i].neighbors = [new_vertices[v.id] for v in self.vertices[i].neighbors]
        return Graph(new_vertices, self.num_features, self.num_features)

def read_data(graph_path: str, features_path: str, apply_kernel: bool) -> Graph:
    vertex_map: Dict[str, Vertex] = {}
    class_map: Dict[str, int] = {}
    vertices: List[Vertex] = []
    features: List[List[float]] = []

    # Read features
    with open(features_path) as fin:
        for line in fin:
            # The format is "id feature* class"
            data = line.rstrip().split()
            clazz = class_map.setdefault(data[-1], len(class_map))
            features_raw = [float(x) for x in data[1: -1]]
            features.append(features_raw)
            vertex = Vertex(None, [], clazz, len(vertices))
            vertices.append(vertex)
            vertex_map[data[0]] = vertex

    if apply_kernel:
        def sqr(x):
            return x * x

        n = len(features)
        kernels = [[] for _ in range(n)]
        # feature_indices = random.choices(range(len(features[0])), k=100)
        nd_features = [nd.array(arr, ctx=data_ctx) for arr in features]
        # indices = random.choices(range(n), k=100)
        indices = range(n)
        for u in range(n):
            for v in indices:
                dif = nd_features[u] - nd_features[v]
                norm = float(nd.norm(dif).asscalar())
                res = math.exp(-0.5 * sqr(norm))
                # res = math.exp(-0.5 * norm)
                # print(norm, )
                kernels[u].append(100 * res)
            # print(kernels[u])
        features = kernels

    for v in vertices:
        v.features = nd.array(features[v.id], ctx=data_ctx).reshape(-1, 1)

    # Read graph
    with open(graph_path) as fin:
        for line in fin:
            (u, v) = [vertex_map[x] for x in line.rstrip().split()]
            u.neighbors.append(v)
            v.neighbors.append(u)

    num_features = len(vertices[0].features)
    for v in vertices:
        v.neighbors.append(v)
        v.degree = len(v.neighbors)
        assert len(v.features) == num_features

    return Graph(vertices, num_features, len(class_map))

class Subgraph:
    """
    Stores graph expansion starting from some set of vertices: next layer includes their neighbors and so on.
    Depending on sampling method we get either the full neighbor-expansion-tree, or some part of it.
    """
    def __init__(self, vertex: int, degree: int, neighbors: List[Subgraph]):
        self.neighbors = neighbors
        """ Next vertices in the expansion """
        self.degree = degree
        """ Actual vertex degree. Note that it might be different from len(neighbors) """
        self.vertex = vertex
        """ The vertex """
