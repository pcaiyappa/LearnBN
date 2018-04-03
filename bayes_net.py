#class for a bayesian network
#attributes are the structure(adjacency matrix) and the parameters quantifying the CPT's
class bayes_net(object):
    def __init__(self,g,adjacency_matrix,cpts,parents,discrete_values_sets):
        self.graph = g
        self.adj_mat = adjacency_matrix
        self.cpts = cpts
        self.pi = parents
        self.discrete_sets = discrete_values_sets#arbitrarily assume the realizations to range (0,cardinality of the node)

