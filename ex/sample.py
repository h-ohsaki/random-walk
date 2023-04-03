#!/usr/bin/env python3
#
# Compare performance of different RW stragtegies.
# Copyright (c) 2023, Hiroyuki Ohsaki.
# All rights reserved.
#
# $Id: run.py,v 1.6 2023/03/20 08:44:56 ohsaki Exp ohsaki $
#

import sys
import math
import statistics
import os

from perlcompat import die, warn, getopts
import randwalk
import graph_tools
import tbdump

def usage():
    die(f"""\
usage: {sys.argv[0]} [-v] [file...]
  -v    verbose mode
""")

def create_graph(name, n_vertices=100, kavg=3.):
    n_edges = int(n_vertices * kavg / 2)
    g = graph_tools.Graph(directed=False)
    if name == 'random':
        return g.create_random_graph(n_vertices, n_edges)
    if name == 'ba':
        return g.create_graph('ba', n_vertices, 10, int(kavg))
    if name == 'barandom':
        return g.create_graph('barandom', n_vertices, n_edges)
    if name == 'ring':
        return g.create_graph('ring', n_vertices)
    if name == 'tree':
        return g.create_graph('tree', n_vertices)
    if name == 'btree':
        return g.create_graph('btree', n_vertices)
    if name == 'lattice':
        return g.create_graph('lattice', 2, int(math.sqrt(n_vertices)))
    if name == 'voronoi':
        return g.create_graph('voronoi', n_vertices // 2)
    if name == 'db':
        # NOTE: average degree must be divisable by 2.
        return g.create_graph('db', n_vertices, n_vertices * 2)
    if name == '3-regular':
        return g.create_random_regular_graph(n_vertices, 3)
    if name == '4-regular':
        return g.create_random_regular_graph(n_vertices, 4)
    if name == 'li_maini':
        # NOTE: 5 clusters, 5% of vertices in each cluster, other vertices are
        # added with preferential attachment.
        return g.create_graph('li_maini', int(n_vertices * .75), 5,
                              int(n_vertices * .25 / 5))
    # FIXMME: support treeba, general_ba, and latent.
    assert False

def simulate(ntrials, label, agent_name, g, start_node):
    n_vertices = len(g.vertices())
    covert_times = []
    hitting_times = []
    for n in range(1, ntrials + 1):
        # Create an agent of a given agent class.
        cls = eval('randwalk.' + agent_name)
        agent = cls(graph=g, current=start_node, bf_size=10000)
        # Perform an instance of simulation.
        while agent.ncovered < n_vertices:
            agent.advance()
            # agent.dump()
        # Collect statistics.
        covert_time = agent.step
        covert_times.append(covert_time)
        samples = [agent.hitting[v] for v in g.vertices()]
        hitting_time = statistics.mean(samples)
        hitting_times.append(hitting_time)
        # Calcurate averages of cover and hitting times.
        avg_cover = statistics.mean(covert_times)
        avg_hitting = statistics.mean(hitting_times)
        print(f'{label} {n:6} {avg_cover:8.2f} {avg_hitting:8.2f}\r',
              file=sys.stderr,
              end='')
    print(f'{label} {n:6} {avg_cover:8.2f} {avg_hitting:8.2f}')

def main():
    ntrials = 100
    n_vertices = 100
    kavg = 3.
    start_vertex = 1
    print('# agent    graph       |V|    |E|  trial       C      E[H]')
    for name in 'random ba barandom ring tree btree lattice voronoi db 3-regular 4-regular li_maini'.split(
    ):
        g = create_graph(name, n_vertices, kavg)
        n = g.nvertices()
        m = g.nedges()
        graph_info = f'{name:8} {n:6} {m:6}'
        for agent in 'SRW BiasedRW SARW MixedRW BloomRW kSARW_LRU kSARW_FIFO kSARW VARW NBRW'.split(
        ):
            label = f'{agent:10} {graph_info}'
            simulate(ntrials, label, agent, g, start_vertex)

if __name__ == "__main__":
    main()
