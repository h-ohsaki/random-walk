#!/usr/bin/env python3
#
#
# Copyright (c) 2023, Hiroyuki Ohsaki.
# All rights reserved.
#
# $Id: run.py,v 1.6 2023/03/20 08:44:56 ohsaki Exp ohsaki $
#

import collections
import math
import random

import numpy

EPS = 1e-4

# ----------------------------------------------------------------
# NOTE: The follwing code is essentially based on
# https://stackoverflow.com/questions/9026519/bloomfilter-python .
class BloomFilter:
    def __init__(self, size):
        if size is None:
            size = 1000
        self.size = size  # Size of the Bloom filter in bits.
        self.bitarray = [0] * self.size

    def hashes(self, key):
        """Return three independent hashes in [0 : self.size] for KEY."""
        digest = hash(str(key))
        hash1 = digest % self.size
        hash2 = (digest // self.size) % self.size
        hash3 = (digest // self.size // self.size) % self.size
        return hash1, hash2, hash3

    def add(self, key):
        """Register KEY to the Bloom filter."""
        for n in self.hashes(key):
            self.bitarray[n] = 1

    def query(self, key):
        """Check if KEY already exists in the Bloom filter.  Note that the
        Bloom filter may result in false positive but not in false
        negative."""
        return all(self.bitarray[n] == 1 for n in self.hashes(key))

# ----------------------------------------------------------------
class SRW:
    """Implementation of the simple random walk (SRW) agent."""
    def __init__(self, graph=None, current=None, *kargs, **kwargs):
        self.graph = graph
        self.n_nodes = len(self.graph.vertices())
        self.path = []  # List of visited vertiecs.
        self.step = 0  # Global clock.
        self.nvisits = collections.defaultdict(
            int)  # Records the number of vists.
        self.ncovered = 0  # The number of uniquely visisted vertices.
        self.hitting = collections.defaultdict(
            int)  # Records the first visiting time.
        if current:
            self.move_to(current)

    def __repr__(self):
        return f'{self.name()}(step={self.step}, current={self.current}, ncovered={self.ncovered})'

    def name(self):
        return type(self).__name__

    def weight(self, u, v):
        """Transistion weight form vertex U to vertex V."""
        # Every neighbor is chosen with the same probability.
        return 1

    def pick_next(self, u=None):
        """Randomly choose one of neighbors of vetex U with the probabiity
        proportional to its weight."""
        if u is None:
            u = self.current
        neighbors = self.graph.neighbors(u)
        # Vertex U must not be isolated.
        assert neighbors
        # Save all weights for transistion from vertex U.
        weights = {v: self.weight(u, v) for v in neighbors}
        total = sum(weights.values())
        chosen = random.uniform(0, total)
        accum = 0
        for v in neighbors:
            accum += weights[v]
            if chosen < accum:
                return v
        assert False  # Must not reach here.
        return None

    def move_to(self, v):
        """Move the random walker to vertex V."""
        self.current = v
        self.path.append(v)
        if not self.nvisits[v]:  # is this the first time?
            self.hitting[v] = self.step
            self.ncovered += 1
        self.nvisits[v] += 1

    def advance(self):
        """Advance the random walker one step forward."""
        v = self.pick_next()
        self.move_to(v)
        self.step += 1

    def prev_vertex(self, n=1):
        try:
            return self.path[-(n + 1)]
        except IndexError:
            return None

    def dump(self):
        v = self.current
        d = self.graph.degree(v)
        print(f'{self.step}\tvisit\t{v}\t{self.nvisits[v]}\t{d}')
        print(f'{self.step}\tstatus\t{self.ncovered}\t{self.n_nodes}')

# ----------------------------------------------------------------
class BiasedRW(SRW):
    """Implementation of the biased random walk (Biased-RW) agent."""
    def __init__(self, alpha=-.5, *kargs, **kwargs):
        self.alpha = alpha
        super().__init__(*kargs, **kwargs)

    def weight(self, u, v):
        """Biased RW randomlyh chooses one of its neighbor with the
        probability proportional to d_v^alpha where d_v is the degree of
        vertex V and alpha is a control parameter."""
        if u is None:
            u = self.current
        w = super().weight(u, v)
        dv = self.graph.degree(v)
        return w * dv**self.alpha

class NBRW(BiasedRW):
    """Implementation of the non-backtracking random walk (NBRW) agent."""
    def weight(self, u, v):
        if u is None:
            u = self.current
        # This code assumes that vertex U is the current vetex.
        assert u == self.current
        if v == self.prev_vertex():
            return EPS
        else:
            return super().weight(u, v)

class SARW(BiasedRW):
    """Implementation of the self-avoiding random walk (SARW) agent."""
    def weight(self, u, v):
        """SARW is equivalent to SRW except that the agent tries to avoid to
        re-visit vertices that the agent has already visited."""
        if u is None:
            u = self.current
        if self.nvisits[v]:
            return EPS
        else:
            return super().weight(u, v)

class BloomRW(BiasedRW):
    """Implementation of the random walk with the Bloom filter (Bloom-RW)
    agent."""
    def __init__(self, bf_size=None, *kargs, **kwargs):
        self.bf = BloomFilter(size=bf_size)
        super().__init__(*kargs, **kwargs)

    def weight(self, u, v):
        if u is None:
            u = self.current
        if self.bf.query(v):
            return EPS
        else:
            return super().weight(u, v)

    def move_to(self, v):
        super().move_to(v)
        self.bf.add(v)

class VARW(NBRW):
    """Implementation of the random walk with vicinity avoidance (VARW)
    agent."""
    def weight(self, u, v):
        """VARW tries to avoid vicinity (i.e., neighbor vertices of the
        previously-visited vertices)."""
        if u is None:
            u = self.current
        # This code assumes that vertex U is the current vetex.
        assert u == self.current
        # NOTE: the original VA-RW avoids neighbors of the last K vertices, rather
        # than those of the previous one.
        t = self.prev_vertex()
        if t and self.graph.has_edge(t, v):
            return EPS
        else:
            return super().weight(u, v)

# ----------------------------------------------------------------
class LZRW(SRW):
    """Implementation of the lazy random walk (LZRW) agent."""
    def __init__(self, laziness=.2, *kargs, **kwargs):
        self.laziness = laziness
        super().__init__(*kargs, **kwargs)

    def pick_next(self, u=None):
        """LZRW probabilistically stays at the current vertex."""
        if u is None:
            u = self.current
        if random.random() <= self.laziness:
            return u
        else:
            return super().pick_next(u)

class MixedRW(BloomRW):
    def weight(self, u, v):
        if u is None:
            u = self.current
        # This code assumes that vertex U is the current vetex.
        assert u == self.current
        # NBRW-like behavior.
        if v == self.prev_vertex():
            return EPS
        # VARW-like behavior.
        # NOTE: the original VA-RW avoids neighbors of the last K vertices, rather
        # than those of the previous one.
        t = self.prev_vertex()
        if t and self.graph.has_edge(t, v):
            return EPS
        # BloomRW-like behavior.
        if self.bf.query(v):
            return EPS
        # BiasedRW-like behavior.
        dv = self.graph.degree(v)
        return dv**-.5

class kSARW(BiasedRW):
    def __init__(self, hist_size=3, *kargs, **kwargs):
        self.hist_size = hist_size
        self.history = collections.deque(maxlen=hist_size)
        super().__init__(*kargs, **kwargs)

    def weight(self, u, v):
        if v in self.history:
            return EPS
        else:
            return super().weight(u, v)

    def move_to(self, v):
        super().move_to(v)
        # Always place the recent entry at the top.
        # NOTE: The history might have duplicates.
        self.history.append(v)

class kSARW_FIFO(kSARW):
    def move_to(self, v):
        super().move_to(v)
        if v not in self.history:
            # The oldest entry is flushed automatically.
            self.history.append(v)

class kSARW_LRU(kSARW):
    def move_to(self, v):
        super().move_to(v)
        # Always place the recent entry at the top.
        if v in self.history:
            self.history.remove(v)
        self.history.append(v)

class MERW(SRW):
    """Implementation of the maximal-entropy random walk (MERW)."""
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        adj = self.graph.adjacency_matrix()
        eigvals, eigvecs = numpy.linalg.eig(adj)
        # Find the index of the largest eigenvalue.
        i = max(enumerate(eigvals), key=lambda x: x[1])[0]
        self.eigval1 = eigvals[i]
        self.eigvec1 = eigvecs[:, i]
        # Alaways use positive eigenvector.  The signs of all elements must
        # be the same.
        if self.eigvec1[0] < 0:
            self.eigvec1 = -self.eigvec1

    def weight(self, u, v):
        if u is None:
            u = self.current
        return (1 / self.eigval1) * (self.eigvec1[v - 1] / self.eigvec1[u - 1])
