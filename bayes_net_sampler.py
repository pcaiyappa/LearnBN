import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
from string import ascii_lowercase as al
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import shlex
import subprocess
from itertools import product
from argparse import ArgumentParser
import os
from bayes_net import bayes_net

#returns a dict of list of all parents for every node keyed by each node
def get_parents_from_adj_matrix(adj_mat):
    parents = dict()  # initialize empty dict
    # for each node
    for col in adj_mat.columns:
        curr_parents = [row for row in adj_mat.index if adj_mat.loc[row,col]==1]
        parents[col] = curr_parents
    return parents

#returns a random categorical distribution with n categories
def gen_rand_categorical(n):
    categorical_dist = np.random.rand(n)
    return categorical_dist/sum(categorical_dist)

#create a cpt where each column is categorical - name cols and rows appropriately
def create_cpt(nrows,ncols,rownames,colnames):
    cpt = pd.DataFrame()
    for i in range(ncols):
        cpt = pd.concat([cpt, pd.Series(gen_rand_categorical(nrows))], axis=1)
    cpt.index = rownames
    cpt.columns = colnames
    return cpt

# generate arbitrary categorical distributions for over each value of the parents
def create_rand_cpts(adj_mat,parents,discrete_value_sets):
    #initialize empty return variable
    cpts = dict()

    for node in adj_mat.index:#for each node
        #create combinations of parents' discrete values
        parents_discrete = []
        if len(parents[node]) > 1:
            for pi in parents[node]:
                curr_parent_named = [pi+"="+str(discrete_value) for discrete_value in discrete_value_sets[pi]]
                parents_discrete.append(curr_parent_named)
            parent_combos = list(product(*parents_discrete))

            rownames = [node+"="+ str(x) for x in discrete_value_sets[node]]#independent variable possible discrete values
            colnames = [str(x) for x in parent_combos]
            cpt = create_cpt(len(discrete_value_sets[node]),len(parent_combos),rownames,colnames)
            cpts[node] = cpt
        elif len(parents[node]) == 1:
            rownames = [node+"="+ str(x) for x in discrete_value_sets[node]]
            colnames = [parents[node][0]+"="+ str(x) for x in discrete_value_sets[parents[node][0]]]
            cpt = create_cpt(len(discrete_value_sets[node]),len(discrete_value_sets[parents[node][0]]),rownames,colnames)
            cpts[node] = cpt
    # print(cpts)
    return cpts

#for a fixed structure( given parents of each node) create randomized cpts
def initialize_cpts(adj_mat):
    #create a dictionary of parents
    parents = get_parents_from_adj_matrix(adj_mat)

    #let the number of the discrete values realized by each node be random(min=2, max=3) and possible discrete values lie btw (1,10)
    discrete_value_sets = {adj_mat.index[i]:np.random.choice(np.arange(1,10),np.random.randint(2,4),replace=False) for i in range(len(adj_mat))}

    #create categorical distributions for every node conditioned on each possible parent
    cpts = create_rand_cpts(adj_mat,parents,discrete_value_sets)

    return cpts,parents,discrete_value_sets

#routine to remove the self-edges from the random scale-free network( Bayes Nets are DAGs)
def remove_self_edges(g):
    #identify self-edges in directed graph if any and remove them
    for e in g.edges():
        if e[0] == e[1]:
            g.remove_edge(e[0],e[0])#this should generate an error is self-edge was incorrectly identified

#routine to change node names to alphabets
def alphabetize(g):
    #convert the nodes from numerics to alphabets
    num_2_alpha = {i:x for i,x in enumerate(al)}
    #new empty graph
    alphabetized_g = nx.DiGraph()
    #map the nodes and edges into alphabet space
    nodes_to_add = [num_2_alpha[node] for node in g.nodes()]
    #take the set as the randomized graph function call returns repeated edges
    edges_to_add = [(num_2_alpha[e[0]],num_2_alpha[e[1]]) for e in set(g.edges())]
    alphabetized_g.add_nodes_from(nodes_to_add)
    alphabetized_g.add_edges_from(edges_to_add)
    return alphabetized_g

#creates a local pdf of the bayesian network structure using the graphviz layout
def create_pdf_from_nxg(g,filename):
    #save all the images in 'transitions' folder
    if not os.path.exists('DAGs'):
        os.makedirs('DAGs')
    os.chdir('DAGs')
    #write the .dot file
    write_dot(g,f'{filename}.dot')
    #call the local dot binary to create the pdf
    cmd = f'dot -Tpng -Gsize=6,6\! -Gdpi=300 {filename}.dot -o {filename}.png'
    args = shlex.split(cmd)
    write_pdf_proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    write_pdf_proc.wait()
    # print ("Wrote pdf. Error:\t=",write_pdf_proc.communicate())
    # remove the intermediate dot file
    cmd = f'rm {filename}.dot'
    args = shlex.split(cmd)
    rm_dot_proc = subprocess.Popen(args,stdout=subprocess.PIPE)
    rm_dot_proc.wait()
    # print ("Removed dot file. Error:\t=",rm_dot_proc.communicate())
    os.chdir('..')#move out of 'DAGs' after saving all the jpgs for the metropolis method
    return

#remove a link from every existing cycle until no cycles exist
def remove_cycles(g):
    while True:
        try:
            cycle = nx.find_cycle(g,orientation='ignore')
            ran_int = cycle[np.random.randint(0,len(cycle))]
            g.remove_edge(ran_int[0],ran_int[1])#randomly select any edge in the cycle to remove
        except:
            break
    return

#removes edge pairs of the form (u,v) and (v,u) if such a pair exists
def retain_uniq_directions(g):
    for e_outer in g.edges():#check each edge against all other edges
        for e_inner in g.edges():
            if e_outer[::-1] == e_inner:#check if reverse edge exists
                g.remove_edge(e_inner[0],e_inner[1])
    return

#remove any nodes with in-degree = out-degree = 0
def remove_singletons(g):
    for node in g.nodes():
        if g.degree(node) == 0:
            g.remove_node(node)
    return

#check if indegree greater than a speficied int
def is_indegree_invalid(g,max_indegree):
    indegrees = g.in_degree(g.nodes())
    for i in indegrees.values():
        if i > max_indegree:
            return True
    return False

#creates a random scale free directed graph and modify it into a DAG
def create_rand_nx_graph(n_nodes,max_num_parents):
    if n_nodes > 13:
        g = nx.dense_gnm_random_graph(n_nodes,np.ceil(n_nodes/2)+5)
    else:
        g = nx.scale_free_graph(n_nodes)
    remove_self_edges(g)
    retain_uniq_directions(g)
    remove_singletons(g)
    g_alpha = alphabetize(g)
    remove_cycles(g_alpha)
    #if the number of indegree exceeds the maximum number of parents --> recursively call the function
    while is_indegree_invalid(g_alpha,max_num_parents):
        g_alpha = create_rand_nx_graph(n_nodes,max_num_parents)
    return g_alpha

#serializes a bayes net object and saves the pickle
def save_bn_as_pickle(bn,filename):
    with open(f'{filename}.pickle','wb') as file:
        pickle.dump(bn,file)
    return

#initialize a random bayes net
def create_random_bn(n_nodes,max_parents):
    g = create_rand_nx_graph(n_nodes,max_parents)

    #create corresponding adjacency matrix --> save as pandas object
    adjacency_matrix = pd.DataFrame(data=nx.adjacency_matrix(g).todense(),index=g.nodes(),columns=g.nodes())

    ##based on the adjacency matrix create cpt tables
    cpts, parents, discrete_value_sets = initialize_cpts(adjacency_matrix)

    #create categorical distribution for independent parent nodes
    independent_parents = [k for k,v in parents.items() if not v]
    for pi in independent_parents:
        cpts[pi] = np.random.rand(len(discrete_value_sets[pi]))
        cpts[pi] /= sum(cpts[pi])

    ##instantiate a bayes_net object with given structure & cpts
    bn = bayes_net(g,adjacency_matrix,cpts,parents,discrete_value_sets)

    return bn

#prints the bayes net attributes on prompt
def print_bn_attributes(bn):
    print("\nDiscrete value sets", bn.discrete_sets)
    print("\nAdjacency Matrix: \n", bn.adj_mat)
    print("\nParents", bn.pi)
    print("\nCPTS")
    for k, v in bn.cpts.items():
        print("\n",k,"\n",v)
    return

#generates random samples from a given bayes net
def bn_sampler(bn, n_samples):
    sampled_data = []
    #use the top sort to randomly samples parents and then subsequently the children
    top_sort = nx.topological_sort(bn.graph)
    for i in range(int(n_samples)):
        currently_sampled = OrderedDict()
        for node in top_sort:#for each node sample randomly from the categorical distributions
            if not bn.pi[node]:#if independent parent node --> probabilities aren't conditional
                node_val = bn.discrete_sets[node][np.argmax(np.random.multinomial(1,bn.cpts[node]))]
                currently_sampled[node] = node_val
            else:#get the conditions, i.e. the realizations of the parents
                if len(bn.pi[node]) == 1:#access column simply with the name of the parent
                    parent = bn.pi[node][0]
                    col_name = parent + "=" + str(currently_sampled[parent])
                    conditioned_probs = bn.cpts[node].loc[:,col_name]
                    node_val = bn.discrete_sets[node][np.argmax(np.random.multinomial(1,conditioned_probs))]
                    currently_sampled[node] = node_val
                elif len(bn.pi[node]) > 1:
                #create the column name to acccess the categorical probabilities from the cpt table for that node
                    col_name = list()
                    for parent in bn.pi[node]:
                        col_name.append(parent+"="+str(currently_sampled[parent]))
                    col_name = str(tuple(col_name))
                    conditioned_probs = list(bn.cpts[node].loc[:,col_name])
                    node_val = bn.discrete_sets[node][np.argmax(np.random.multinomial(1,conditioned_probs))]
                    currently_sampled[node] = node_val
        curr_sample_vector = [x for x in currently_sampled.values()]
        sampled_data.append(curr_sample_vector)

    return pd.DataFrame(np.matrix(sampled_data),columns=list(top_sort))

#initializes a random bayes net with n_nodes and generates n_samples by invoking bn_sampler on the bn object
def gen_sample(args):
    #random bayes net generator
    bn = create_random_bn(args.nnodes,args.maxp)
    print("Initialized the random bayes net.")
    if args.verbose:
        print_bn_attributes(bn)
    # sampler
    sampled_data = bn_sampler(bn, args.nsamples)
    #save the sampled bayes net object
    # save graphviz layout of top-sorted graph as pdf
    create_pdf_from_nxg(bn.graph, f'{args.out}_DAG')

    #save a pickle object of the bayes_net
    save_bn_as_pickle(bn,f'{args.out}')

    #save the sample as csv file
    sampled_data.to_csv(f'{args.out}_{int(args.nsamples)}_samples.csv',header=True)
    return bn,sampled_data

if __name__ == '__main__':
    # collect arguments from cmd line
    parser = ArgumentParser()
    parser.add_argument('-nnodes', type=int, help="Number of nodes. max=26!")
    parser.add_argument('-nsamples', type=float, help="Number of samples required!")
    parser.add_argument('-out',type=str,help="Output Filename")
    parser.add_argument('-maxp',type=int,help="Max number of parents of any node.")
    parser.add_argument('-verbose',type=bool,help="Print attributes of the random bayes net used for sampling.")
    args = parser.parse_args()
    print(args)
    print("Generated ",args.nsamples," samples from Random Bayes Net with", args.nnodes," nodes\n")

    #generate the sample according to user input
    bn,sampled_data = gen_sample(args)
    print(f'Saved sampled bayes net as {args.out}.pickle.\n A pdf of the DAG was created @{args.out}_DAG.pdf.')
    print(f'Saved Sample data as {args.out}_{args.nsamples}_samples.csv.\n\nCOMPLETE.')
