import sys
import logging
import os
from os.path import join as pjoin

import numpy as np
import networkx as nx

import utility

sys.path.insert(1, 'gensim')
from gensim.models import Word2Vec

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

__version__ = '0.1'


class OhmNet():
    def __init__(self, net_input, weighted, directed,
        hierarchy_input, p, q, num_walks, walk_length,
        dimension, window_size, n_workers, n_iter, out_dir, seed=0):
        self.net_input = net_input
        self.weighted = weighted
        self.directed = directed
        self.hierarchy_input = hierarchy_input
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.dimension = dimension
        self.window_size = window_size
        self.n_workers = n_workers
        self.n_iter = n_iter
        self.out_dir = out_dir
        self.rng = np.random.RandomState(seed)

        self.log = logging.getLogger('OHMNET')

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.nets = utility.read_nets(
            self.net_input, self.weighted, self.directed, self.log)
        self.hierarchy = utility.read_hierarchy(self.hierarchy_input, self.log)

    def simulate_walks(self):
        all_walks = []
        for net_name, net in self.nets.items():
            self.log.info('Walk simulation: %s' % net_name)
            walks = utility.Walks(net, self.directed, self.p, self.q, self.log)
            sim_walks = walks.simulate_walks(self.num_walks, self.walk_length)
            all_walks.extend(sim_walks)
        return all_walks

    def relabel_nodes(self):
        new_nets = {}
        for net_name, net in self.nets.items():
            def mapping(x):
                return '%s__%d' % (net_name, x)
            new_nets[net_name] = nx.relabel_nodes(net, mapping, copy=False)
        return new_nets

    def update_internal_vectors(
            self, all_nodes, leaf_vectors, internal_vectors):
        new_internal_vectors = {}
        for hnode in self.hierarchy.nodes():
            if hnode in self.nets:
                # leaf vectors are optimized separately
                continue
            self.log.info('Updating internal hierarchy node: %s' % hnode)
            new_internal_vectors[hnode] = {}
            children = self.hierarchy.successors(hnode)
            parents = self.hierarchy.predecessors(hnode)
            for node in all_nodes:
                # update internal vectors (see Eq. 4 in Gopal et al.)
                if parents:
                    t1 = 1. / (len(children) + 1.)
                else:
                    t1 = 1. / len(children)
                t2 = [np.zeros(leaf_vectors.values()[0].shape)]
                for child in children:
                    if child in self.nets:
                        node_name = '%s__%s' % (child, node)
                        # node can be missing in certain networks
                        if node_name in leaf_vectors:
                            t2.append(leaf_vectors[node_name])
                    else:
                        t2.append(internal_vectors[child][node])
                if parents:
                    parent = self.hierarchy.predecessors(hnode)[0]
                    assert len(self.hierarchy.predecessors(hnode)) == 1, 'Problem'
                    parent_vector = internal_vectors[parent][node]
                else:
                    # root of the hierarchy
                    parent_vector = 0
                new_internal_vector = t1 * (parent_vector + sum(t2))
                new_internal_vectors[hnode][node] = new_internal_vector
        return new_internal_vectors

    def init_internal_vectors(self, all_nodes):
        internal_vectors = {}
        for hnode in self.hierarchy.nodes():
            if hnode in self.nets:
                # leaf vectors are optimized separately
                continue
            internal_vectors[hnode] = {}
            for node in all_nodes:
                vector = (self.rng.rand(self.dimension) - 0.5) / self.dimension
                internal_vectors[hnode][node] = vector
            n_vectors = len(internal_vectors[hnode])
            self.log.info('Hierarchy node: %s -- %d' % (hnode, n_vectors))
        return internal_vectors

    def save_parent_word2vec_format(self, all_nodes, internal_vectors, fname):
        node2internal_vector = {}
        for net_name, net in self.nets.items():
            for node in all_nodes:
                parent = self.hierarchy.predecessors(net_name)[0]
                assert len(self.hierarchy.predecessors(net_name)) == 1, 'Problems'
                parent_vector = internal_vectors[parent][node]
                node_name = '%s__%s' % (net_name, node)
                node2internal_vector[node_name] = parent_vector

        with open(fname, 'w') as fout:
            self.log.info('Writing: %s' % fname)
            n = sum([net.number_of_nodes() for net in self.nets.values()])
            d = len(node2internal_vector.values()[0])
            fout.write('%d %d\n' % (n, d))
            for _, net in self.nets.items():
                for node in net.nodes():
                    internal_vector = node2internal_vector[node]
                    profile = ' '.join(map(str, internal_vector))
                    fout.write('%s %s\n' % (node, profile))

    def save_internal_word2vec_format(
            self, all_nodes, internal_vectors, fname):
        self.log.info('Writing: %s' % fname)
        with open(fname, 'w') as fout:
            n = (self.hierarchy.number_of_nodes() - len(self.nets)) * len(all_nodes)
            d = len(internal_vectors.values()[0].values()[0])
            fout.write('%d %d\n' % (n, d))
            for hnode in self.hierarchy.nodes():
                if hnode in self.nets:
                    # leaf vectors are saved separately
                    continue
                for node in all_nodes:
                    internal_vector = internal_vectors[hnode][node]
                    profile = ' '.join(map(str, internal_vector))
                    node_name = '%s__%s' % (hnode, node)
                    fout.write('%s %s\n' % (node_name, profile))

    def get_all_nodes(self):
        all_nodes = set()
        for _, net in self.nets.items():
            nodes = [node.split('__')[1] for node in net.nodes()]
            all_nodes.update(nodes)
        self.log.info('All nodes: %d' % len(all_nodes))
        return list(all_nodes)

    def get_leaf_vectors(self, model):
        leaf_vectors = {}
        for word, val in model.vocab.items():
            leaf_vector = model.syn0[val.index]
            assert type(word) == str, 'Problems with vocabulary'
            leaf_vectors[word] = leaf_vector
        return leaf_vectors

    def embed_multilayer(self):
        """Neural embedding of a multilayer network"""

        self.nets = self.relabel_nodes()

        # Return parameter p
        # It controls the likelihood of immediately revisiting a node in the walk

        # In-out parameter q
        # If q > 1: the random walk is biased towards nodes close to node t
        # If q < 1: the random walk is more inclined to visit nodes which
        #           are further away from node t
        all_walks = self.simulate_walks()

        all_nodes = self.get_all_nodes()
        internal_vectors = self.init_internal_vectors(all_nodes)

        tmp_fname = pjoin(self.out_dir, 'tmp.emb')
        total_examples = len(all_walks) * self.n_iter
        pushed_examples = 1000

        for itr in range(self.n_iter):
            # update leaf layers
            self.log.info('Iteration: %d' % itr)

            if itr == 0:
                self.model = Word2Vec(
                    sentences=all_walks, size=self.dimension,
                    window=self.window_size, min_count=0, sg=1,
                    workers=self.n_workers, iter=1, batch_words=pushed_examples)
            else:
                self.model.current_iteration = itr
                self.model.load_parent_word2vec_format(fname=tmp_fname)
                delta = (self.model.alpha - self.model.min_alpha) *\
                        pushed_examples / total_examples
                next_alpha = self.model.alpha - delta
                next_alpha = max(self.model.min_alpha, next_alpha)
                self.model.alpha = next_alpha
                self.log.info('Next alpha = %8.6f' % self.model.alpha)

                self.model.train(all_walks)

            leaf_vectors = self.get_leaf_vectors(self.model)
            internal_vectors = self.update_internal_vectors(
                all_nodes, leaf_vectors, internal_vectors)
            self.save_parent_word2vec_format(
                all_nodes, internal_vectors, tmp_fname)

        self.log.info('Done!')

        fname = pjoin(self.out_dir, 'leaf_vectors.emb')
        self.log.info('Saving leaf vectors: %s' % fname)
        self.model.save_word2vec_format(fname)

        fname = pjoin(self.out_dir, 'internal_vectors.emb')
        self.log.info('Saving internal vectors: %s' % fname)
        self.save_internal_word2vec_format(
            all_nodes, internal_vectors, fname)
        return self.model
