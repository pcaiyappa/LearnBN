#DEPENDENCIES
1. pip install the python package library networkx:
2. local installation of graphviz

#USAGE
1. Sampler: For 1000 samples from  a randomly generated Bayes Net with max. number of parents of any node restricted to 3:
usage: bayes_net_sampler.py [-h] [-nnodes NNODES] [-nsamples NSAMPLES]
                            [-out OUT] [-maxp MAXP] [-verbose VERBOSE]

optional arguments:
  -h, --help          show this help message and exit
  -nnodes NNODES      Number of nodes. max=26!
  -nsamples NSAMPLES  Number of samples required!
  -out OUT            Output Filename
  -maxp MAXP          Max number of parents of any node.
  -verbose VERBOSE    Print attributes of the random bayes net used for
                      sampling.

Example arguments:
python bayes_net_sampler.py -nnodes 6 -nsamples 1000 -out trial -maxp 3 -verbose 1


2. Learner: source switches between running the learner on a randomly generated network and the de genes mouse neuron scRNAseq data
usage: bayes_net_learner.py [-h] [-source SOURCE] [-nnodes NNODES]
                            [-nsamples NSAMPLES] [-maxp MAXP] [-niter NITER]
                            [-out OUT] [-verbose VERBOSE]

optional arguments:
  -h, --help          show this help message and exit
  -source SOURCE      Enter 'random' to run on random sample or 'de_gene_expr'
                      to run on mouse data.
  -nnodes NNODES      Number of samples required! Only req for 'random'.
  -nsamples NSAMPLES  Number of samples required! Only req for 'random'
  -maxp MAXP          Max number of parents of any node.
  -niter NITER        Number of iterations
  -out OUT            Output Filename
  -verbose VERBOSE    Print attributes of the random bayes net used for
                      sampling.
Example arguments:
python bayes_net_learner.py -source random -nnodes 5 -nsamples 1000 -maxp 3 -niter=10 -out random_test
python bayes_net_learner.py -source de_expr_genes -maxp 3 -niter=100 -out mm10_neuron_test

