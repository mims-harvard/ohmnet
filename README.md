# OhmNet

The *OhmNet* algorithm learns feature representations 
for nodes in any (un)directed, (un)weighted multi-layer network. Please 
check the [project page](http://snap.stanford.edu/ohmnet) for more details. 

## Usage

To run *OhmNet* on [human brain multi-layer network with nine layers](http://snap.stanford.edu/ohmnet/), 
run the following command from the project home directory:

    python2.7 main.py --input "data/brain.list" --outdir "tmp" --hierarchy "data/brain.hierarchy" 

## Options

To check *OhmNet*'s running options, use:

    python2.7 main.py --help

## Output

Results are saved to output directory specified by the ``out_dir`` 
option. 

The output file ``leaf_vectors.emb`` contains feature representations 
for nodes at the level of leaves in the hierarchy (i.e., leaves in the
hierarchy correspond exactly to network layers).  

The first line has the following format:

	total_num_of_nodes_in_layers dim_of_representation

The next ``total_num_of_nodes_in_layers`` lines are as follows:
	
	node_id dim1 dim2 ... dimd

where node_id is formatted as network_layer_name__node_id, and dim1, ... , dimd is 
the *d*-dimensional representation learned by *OhmNet*.

The output file ``internal_vectors.emb`` contains feature representations
for nodes at higher levels in the hierarchy (i.e., internal levels in the 
hierarchy contain feature representations at intermediate/higher scales).

The first line has the following format:

	total_num_of_nodes_in_hierarchy dim_of_representation

where ``total_num_of_nodes_in_hierarchy`` is equal to (size_hierarchy - 
num_layers) * total_num_nodes.

The next ``total_num_of_nodes_in_hierarchy`` lines are as follows:
	
	node_id dim1 dim2 ... dimd

where node_id is formatted as hierarchy_element_name__node_id, and dim1, ... , dimd is 
the *d*-dimensional representation learned by *OhmNet*.

## Citing

If you find *OhmNet* useful for your research, please consider citing 
the following [paper presented at ISMB/ECCB 2017](https://academic.oup.com/bioinformatics/article/33/14/i190/3953967/Predicting-multicellular-function-through-multi):

    @article{Zitnik2017,
      title     = {Predicting multicellular function through multi-layer tissue networks},
      author    = {Zitnik, Marinka and Leskovec, Jure},
      journal   = {Bioinformatics},
      volume    = {33},
      number    = {14},
      pages     = {190-198},
      year      = {2017},
      publisher = {Oxford Journals}
    }

## Miscellaneous

Please send any questions you might have about the code and/or the 
algorithm to <marinka@cs.stanford.edu>.

Note: This is a full Python implementation of *OhmNet* 
algorithm. A C++ implementation will be released as part of SNAP software.

## Dependencies

OhmNet is tested to work under Python 2.7.

The required dependencies for OhmNet are [NumPy](http://www.numpy.org) >= 1.12, and [NetworkX](https://networkx.github.io/) >= 1.11.

## License

OhmNet is licensed under the MIT License.
