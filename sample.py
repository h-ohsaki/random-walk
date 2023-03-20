#!/usr/bin/env python3
#
#
# Copyright (c) 2023, Hiroyuki Ohsaki.
# All rights reserved.
#
# $Id: run.py,v 1.6 2023/03/20 08:44:56 ohsaki Exp ohsaki $
#

import sys

from perlcompat import die, warn, getopts
import randwalk
import graph_tools
import tbdump

def usage():
    die(f"""\
usage: {sys.argv[0]} [-v] [file...]
  -v    verbose mode
""")

def simulate(g, start_node, agent_cls, ntrials=500):
    n_nodes = len(g.vertices())
    dst_node = 2
    covert_times = []
    hitting_times = []
    for n in range(1, ntrials + 1):
        # Create an agent of a given agent class.
        cls = eval('randwalk.' + agent_cls)
        agent = cls(graph=g, current=start_node)
        # Perform an instance of simulation.
        while agent.ncovered() < n_nodes:
            agent.advance()
        covert_times.append(agent.step)
        hitting_times.append(agent.hitting[dst_node])
        # Calcurate averages of cover and hitting times.
        cover_time = sum(covert_times) / n
        hitting_time = sum(hitting_times) / n
        print(f'{agent_cls:8} {n:4} {cover_time:8.2f} {hitting_time:8.2f}\r',
              end='')
    print()

def main():
    opt = getopts('v') or usage()
    verbose = opt.v

    n_nodes = 100
    g = graph_tools.Graph(directed=False)
    g = g.create_graph('random', n_nodes, int(n_nodes * 2))
    start_node = 1

    simulate(g, start_node, 'SRW')
    simulate(g, start_node, 'BiasedRW')
    simulate(g, start_node, 'NBRW')
    simulate(g, start_node, 'SARW')
    simulate(g, start_node, 'BloomRW')

if __name__ == "__main__":
    main()
