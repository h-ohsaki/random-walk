#!/usr/bin/env python3
#
# Compare performance of different RW stragtegies.
# Copyright (c) 2023, Hiroyuki Ohsaki.
# All rights reserved.
#
# $Id: run.py,v 1.6 2023/03/20 08:44:56 ohsaki Exp ohsaki $
#

import collections
import math
import random
import statistics
import sys

from perlcompat import die, warn, getopts
import randwalk
import graph_tools
import tbdump

MAX_STEPS = 10000
GRAPH_TYPES = 'random,ba,barandom,ring,tree,btree,lattice,voronoi,db,3-regular,4-regular,li_maini'
AGENT_NAMES = 'EmbedRW,SRW,SARW,HybridRW,BloomRW,kHistory_LRU,kHistory_FIFO,kHistory,VARW,NBRW,BiasedRW,EigenvecRW,ClosenessRW,BetweennessRW,EccentricityRW,LZRW,MaxDegreeRW,MERW'

def usage():
    die(f"""\
usage: {sys.argv[0]} [-s #] [-N #] [-n #] [-k #] [-A #]
  -s seed            specifiy the seed of random number generator
  -N                 the number of simulation runs (default: 100)
  -n                 the desired number of vertices (default: 100)
  -k                 the desired average degree (default: 2.5)
  -a name[,name...]  list of agent names (default: {AGENT_NAMES})
  -g name[,name...]  list of graph types (default: {GRAPH_TYPES})
  -A step            perform adversary attack every STEP
""")

def conf95(vals):
    """Return 95% confidence interval for measurements VALS."""
    zval = 1.960
    if len(vals) <= 1:
        return 0
    return zval * statistics.stdev(vals) / math.sqrt(len(vals))

def mean_and_conf95(vals):
    """Return the mean and the 95% confience interval for measurements
    VALS."""
    return statistics.mean(vals), conf95(vals)

def create_graph(type_, n=100, k=3.):
    """Randomly generate a graph instance using network generation model TYPE_.
    If possible, a graph with N vertices and the average degree of K is
    generated."""
    m = int(n * k / 2)
    g = graph_tools.Graph(directed=False)
    g.set_graph_attribute('name', type_)
    if type_ == 'random':
        return g.create_random_graph(n, m)
    if type_ == 'ba':
        return g.create_graph('ba', n, 10, int(k))
    if type_ == 'barandom':
        return g.create_graph('barandom', n, m)
    if type_ == 'ring':
        return g.create_graph('ring', n)
    if type_ == 'tree':
        return g.create_graph('tree', n)
    if type_ == 'btree':
        return g.create_graph('btree', n)
    if type_ == 'lattice':
        return g.create_graph('lattice', 2, int(math.sqrt(n)))
    if type_ == 'voronoi':
        return g.create_graph('voronoi', n // 2)
    if type_ == 'db':
        # NOTE: average degree must be divisable by 2.
        return g.create_graph('db', n, n * 2)
    if type_ == '3-regular':
        return g.create_random_regular_graph(n, 3)
    if type_ == '4-regular':
        return g.create_random_regular_graph(n, 4)
    if type_ == 'li_maini':
        # NOTE: 5 clusters, 5% of vertices in each cluster, other vertices are
        # added with preferential attachment.
        return g.create_graph('li_maini', int(n * .75), 5, int(n * .25 / 5))
    # FIXMME: support treeba, general_ba, and latent.
    assert False

def header_str():
    return '# agent    \tN\tM\ttype\tcount\tabort\tC\t95%\tH\t95%\tE[H]\t95%'

def status_str(agent, g, count, naborts, covers, hittings, mean_hittings):
    name = agent.name()
    try:
        alpha = agent.alpha
    except AttributeError:
        alpha = None
    n = g.nvertices()
    m = g.nedges()
    type_ = g.get_graph_attribute('name')
    # Collect statistics.
    cover = agent.step
    covers.append(cover)
    # NOTE: hiting[v] records the hitting time at vertex V.
    hitting = agent.hitting[agent.target]
    hittings.append(hitting)
    mean_hitting = statistics.mean(agent.hitting.values())
    mean_hittings.append(mean_hitting)
    c_avg, c_conf = mean_and_conf95(covers)
    h_avg, h_conf = mean_and_conf95(hittings)
    mh_avg, mh_conf = mean_and_conf95(mean_hittings)
    label = f"{name} {alpha or ''}"
    return f'{label:12}\t{n}\t{m}\t{type_}\t{count}\t{naborts}\t{c_avg:.0f}\t{c_conf:.0f}\t{h_avg:.0f}\t{h_conf:.0f}\t{mh_avg:.0f}\t{mh_conf:.0f}'

def attack_agent(agent):
    # Count the number of past visits.
    counter = collections.Counter(agent.path)
    g = agent.graph
    u = agent.current
    # Try to rewire to frequently-visisted vertex.
    for v, count in counter.most_common():
        # u: current vertex
        # v: one of frequently-visisted vertices
        if g.has_edge(u, v):
            continue
        # Try to find link (u, w) which can be safely deleted without isolating 
        # the graph into multiple components.
        neighbors = list(g.neighbors(u))
        random.shuffle(neighbors)
        for w in neighbors:
            h = g.copy_graph()
            h.delete_edge(u, w)
            # The graph still remains connected after link deletion?
            if h.is_connected():
                # Rewrire the link to the (possible) central vertex.
                g.delete_edge(u, w)
                g.add_edge(u, v)
            return True
    return False

def simulate(agent_name,
             g,
             start_vertex=1,
             alpha=0,
             ntrials=100,
             attack_interval=None):
    covers = []
    hittings = []
    mean_hittings = []
    naborts = 0
    for count in range(1, ntrials + 1):
        # Create an agent of a given agent name.
        cls = eval('randwalk.' + agent_name)
        agent = cls(graph=g,
                    current=start_vertex,
                    target=g.nvertices(),
                    alpha=alpha)
        # Perform an instance of simulation.
        while agent.ncovered < g.nvertices():
            agent.advance()
            if agent.step > MAX_STEPS:
                naborts += 1
                break
            if attack_interval is not None and agent.step % attack_interval == 0:
                attack_agent(agent)
        stat = status_str(agent, g, count, naborts, covers, hittings,
                          mean_hittings)
        print(stat + '\r', file=sys.stderr, end='')
        # Abort the experiment if it takes too long.
        if naborts >= 10:
            break
    # FIXME: workaround when the stdout is redirected.
    if not sys.stdout.isatty():
        print('', file=sys.stderr)
    print(stat + ' ')

def main():
    opt = getopts('s:N:n:k:a:g:A:') or usage()
    seed = int(opt.s) if opt.s else None
    random.seed(seed)
    ntrials = int(opt.N) if opt.N else 100
    n_desired = int(opt.n) if opt.n else 100
    k_desired = float(opt.k) if opt.k else 2.5
    agent_names = opt.a if opt.a else AGENT_NAMES
    graph_types = opt.g if opt.g else GRAPH_TYPES
    attack_interval = int(opt.A) if opt.A else None
    start_vertex = 1
    print(header_str())
    for type_ in graph_types.split(','):
        g = create_graph(type_, n_desired, k_desired)
        for agent in agent_names.split(','):
            alphas = [0]
            if agent in 'NBRW BiasedRW EigenvecRW ClosenessRW BetweennessRW EccentricityRW':
                alphas = [-.4, -.2, 0, .2, .4]
            for alpha in alphas:
                simulate(agent, g, start_vertex, alpha, ntrials,
                         attack_interval)

if __name__ == "__main__":
    main()
