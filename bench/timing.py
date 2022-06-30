import sys
from time import perf_counter
import random
from os import path
from collections import defaultdict
from itertools import chain
from scipy.stats import kruskal, mannwhitneyu

here_dir = path.dirname(path.abspath(__file__))
sys.path.append(path.join(here_dir, '..'))

from neat import Genome, GeneSpec, Gene, ConnectionGene, Mutator, default_gene_factory

def generate_hmarks(num_hmarks, proba):
    n = 0
    hmark = 0
    while n < num_hmarks:
        if random.random() < proba:
            yield hmark
            n += 1
        hmark += 1


def make_genome(hmarks):
    # we add a few neurons, then a few connections, and repeat until we are out of hmarks
    a_few_neurons = 8
    a_few_conns = 8

    hmarks = tuple(hmarks)
    if a_few_neurons + a_few_conns > len(hmarks):
        a_few_neurons = len(hmarks) // 2
        a_few_conns = len(hmarks) - a_few_neurons

    layers = (
        GeneSpec("layer0"),
        GeneSpec("layer1"),
        GeneSpec("layer2"),
        GeneSpec("layer3"),
        GeneSpec("layer4"),
    )
    genes_in_layers = {layer: [] for layer in layers}
    genes_in_channels = defaultdict(list)
    conn_spec = GeneSpec("connection")

    neuron_genes = []
    connections = set()

    num_genes_to_go = len(hmarks)
    hmarks = iter(hmarks)
    while num_genes_to_go:
        for _ in range(min(a_few_neurons, num_genes_to_go)):
            layer = random.choice(layers)
            ng = Gene(layer, [], next(hmarks))
            genes_in_layers[layer].append(ng)
            neuron_genes.append(ng)
            num_genes_to_go -= 1

        for _ in range(min(a_few_conns, num_genes_to_go)):
            g1 = random.choice(neuron_genes)
            g2 = random.choice(neuron_genes)
            while (g1.historical_mark, g2.historical_mark) in connections:
                g1 = random.choice(neuron_genes)
                g2 = random.choice(neuron_genes)

            cg = ConnectionGene(
                conn_spec, [], g1.historical_mark, g2.historical_mark, next(hmarks))
            genes_in_channels[(g1.spec, g2.spec)].append(cg)
            connections.add((g1.historical_mark, g2.historical_mark))
            num_genes_to_go -= 1

    genome = Genome()
    for layer, gs in genes_in_layers.items():
        if len(gs) > 0:
            genome.add_layer(layer, gs)

    for channel, gs in genes_in_channels.items():
        if len(gs) > 0:
            genome.add_channel(channel, gs)

    return genome


def find_max_hmark(genome):
    return max(g.historical_mark
        for gs in chain(genome.layers().values(), genome.channels().values())
        for g in gs.iter_non_empty())


def make_mutator(genome):
    for conn_genes in genome.channels().values():
        for conn_gene in conn_genes:
            return Mutator(
                innovation_number=find_max_hmark(genome) + 1,
                neuron_factory=default_gene_factory(*genome.layers()),
                connection_factory=default_gene_factory(conn_gene.spec),
                channels=genome.channels(),
            )
    return None


def copy_genes_new(genome):
    for g in genome.neuron_genes():
        g_ = g.copy_new()
    for g in genome.connection_genes():
        g_ = g.copy_new()
    return g_


def copy_genes_old(genome):
    for g in genome.neuron_genes():
        g_ = g.copy()
    for g in genome.connection_genes():
        g_ = g.copy()
    return g_


if __name__ == "__main__":
    REPS_MAJOR = 10*100
    REPS_MINOR = 10
    DATA_SIZE = 500


    # genome = make_genome(generate_hmarks(DATA_SIZE, .02))
    # print(genome)
    # print(tuple(genome.channels()))
    # print(f"max mark = {find_max_hmark(genome)}")
    # mutator = make_mutator(genome)
    # print(mutator)

    # for _ in range(20):
    #     mutator.remove_random_neuron(genome)

    # print(f"after removal:\n{genome}")
    # sys.exit(0)

    times_old = []
    times_new = []

    print("gathering timing data...")
    for _ in range(REPS_MAJOR):
        g1 = make_genome(generate_hmarks(DATA_SIZE, .02))
        mutator = make_mutator(g1)

        ### OLD ###
        timer0 = perf_counter()
        for _ in range(REPS_MINOR):
            mutator.add_random_neuron(g1)
            # mut.add_random_connection(g1)
            # g2 = g1.copy()
            # copy_genes_old(g1)
            # for _ in Genome.align_genes(g1, g2):
            #     pass

        timer1 = perf_counter()
        times_old.append(timer1 - timer0)
        # print(f"OLD: {g2.connections_index}")

        ### NEW ###
        timer0 = perf_counter()
        for _ in range(REPS_MINOR):
            mutator.add_random_neuron_new(g1)
            # mut.add_random_connection_new(g1)
            # g2 = g1.copy_new()
            # copy_genes_new(g1)
            # g2 = Genome([*g1.neuron_genes()], [*g1.connection_genes()])
            # for _ in Genome.align_genes_2(g1, g2):
            #     pass

        timer1 = perf_counter()
        times_new.append(timer1 - timer0)
        # print(f"NEW: {g2.connections_index}")


        g1 = make_genome(generate_hmarks(DATA_SIZE, .02))
        mutator = make_mutator(g1)

        ### NEW ###
        timer0 = perf_counter()
        for _ in range(REPS_MINOR):
            mutator.add_random_neuron_new(g1)
            # mut.add_random_connection_new(g1)
            # g2 = g1.copy_new()
            # copy_genes_new(g1)
            # g2 = Genome([*g1.neuron_genes()], [*g1.connection_genes()])
            # for _ in Genome.align_genes_2(g1, g2):
            #     pass

        timer1 = perf_counter()
        times_new.append(timer1 - timer0)
        # print(f"NEW: {g2.connections_index}")

        ### OLD ###
        timer0 = perf_counter()
        for _ in range(REPS_MINOR):
            mutator.add_random_neuron(g1)
            # mut.add_random_connection(g1)
            # g2 = g1.copy()
            # copy_genes_old(g1)
            # for _ in Genome.align_genes(g1, g2):
            #     pass

        timer1 = perf_counter()
        times_old.append(timer1 - timer0)
        # print(f"OLD: {g2.connections_index}")


    print("running stats...")

    total_old = sum(times_old)
    total_new = sum(times_new)
    print(f"total time: old={total_old}, new={total_new}, diff={total_new - total_old}")

    h_val, p_val = kruskal(times_old, times_new)
    print(f"Kruskal H={h_val}, p={p_val}")


    u_val, p_val = mannwhitneyu(times_old, times_new)
    print(f"MW      U={h_val}, p={p_val}")
