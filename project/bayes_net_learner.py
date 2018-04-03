from bayes_net_sampler import create_random_bn,bn_sampler,print_bn_attributes,create_pdf_from_nxg,\
    remove_cycles,bayes_net,get_parents_from_adj_matrix
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import product
import os,sys
import shlex
import subprocess
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from argparse import ArgumentParser
#TODO: CONVERT TO JSON FOR D3 --> last

# create a random sparse binary(0,1) adjaceny matrix - returns a nx graph object and the pandas adjacency matrix
def create_rand_adj_matrix(n_nodes):
    # matrix = np.random.choice([0,1],size=(n_nodes**2,),p=[2/3,1/3]).reshape(n_nodes,n_nodes)
    matrix = np.random.randint(low=0,high=2,size=n_nodes**2).reshape(n_nodes,n_nodes)
    # it is a connected component initially(include all nodes)
    for row in range(matrix.shape[0]):
        if sum(matrix[row,:]) == 0 and sum(matrix[:,row]) == 0:#if in-degree and out-degree equal to zero
            possible_edges = [zip([row]*n_nodes,np.arange(0,n_nodes))] + [zip(np.arange(0,n_nodes),[row]*n_nodes)]
    #remove self-edges, i.e diagonal entries - Bayes Net constraint
    np.fill_diagonal(matrix,0)
    #remove an edge from reverse edge-pairs, i.e. those implying bidirectionality
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row,col] == 1 and matrix[col,row] == 1:
                matrix[row,col] = 0#arbitrarily remove a reverse edge
    return matrix

#create an adjaceny matrix and a corresponding nx graph based on given number of nodes and node ids
def initialize_random_structure(n_nodes,node_ids):
    #create a random binary matrix
    adj_mat = create_rand_adj_matrix(n_nodes)
    df = pd.DataFrame(data=adj_mat,index=node_ids,columns=node_ids)
    #create edge list representation using the adjacency matrix
    edgelist = [(row,col) for row in list(df.index) for col in list(df.columns) if df.loc[row,col]==1]
    g = nx.DiGraph()
    g.add_edges_from(edgelist)
    #remove cycles
    remove_cycles(g)
    # #ensure all components occur
    # while not len(g.nodes()) == n_nodes:
    #     g = initialize_random_structure(n_nodes,node_ids)

    return g,pd.DataFrame(data=nx.adjacency_matrix(g).todense(),index=g.nodes(),columns=g.nodes())

#returns the unique discrete values/realizations of the nodes
def get_discrete_sets(data):
    discrete_sets = {}
    for i,node in enumerate(data.columns):
        discrete_sets[node] = set(data.loc[:,node].tolist())
    return discrete_sets

#returns a dictionary of the total number of occurences keyed by discrete realizations of all parent sets
def get_parents_total_counts(data,parents_dict):
    total_counts = defaultdict(int)
    #create a list of unique parent set to avoid double counting(if 2 nodes have the same set of parents)
    uniq_parents = set([",".join(x) for x in list(parents_dict.values()) if x])
    for parent_set in uniq_parents:#get the corresponding counts for each unique parent set
        for i in list(data.index):
            key = f'{parent_set}={",".join([str(data.loc[i,x]) for x in parent_set.split(",")])}'
            total_counts[key] += 1
    return total_counts

#perform mle for the parameters node-wise --> a parameter is equal to prob(node-value|parent_set-values)
def perform_mle(data,parents):
    #closed-form soln: fraction of common occurences of the parent_set values and the node_value over all parent
    conditional_counts = defaultdict(int)
    roots_counts = defaultdict(int)
    #maintain a dict of the total number of counts for an discrete assignment to the parent set
    parents_total_counts = get_parents_total_counts(data,parents)
    for node in data:#for each node
        if parents[node]:#node is conditional upon some parents
            for i in list(data.index):
                node_value = data.loc[i][node]
                parents_values = list(data.loc[i][parents[node]])
                key = f'{node}={node_value}|{",".join(parents[node])}={",".join([str(x) for x in parents_values])}'
                conditional_counts[key] += 1
        else:#independent node; calculate the probabilities in the same pass
            for i in list(data.index):
                roots_counts[f'{node}={data.loc[i][node]}'] += 1
    #normalize the probabilities
    conditional_probs = {k:v/(parents_total_counts[k.split("|")[1]]) for k,v in conditional_counts.items()}
    root_probs = {k:(v/data.shape[0]) for k,v in roots_counts.items()}
    return conditional_counts,conditional_probs,roots_counts,root_probs

#use an adjacency matrix and the training data to compute mle estimates of all conditional probabilities
def compute_cpts_mle(adjacency_matrix,data):
    #id the parents based on the structure encoded in the adjacency matrix
    pi = get_parents_from_adj_matrix(adjacency_matrix)
    #compute mle estimates of the cpts
    return perform_mle(data,pi)

#compute the penalty term of the bic score
def compute_penalty(data,adj_mat):
    #get the number of parameters
    discrete_sets = get_discrete_sets(data)
    #get the parents for each node
    pi = get_parents_from_adj_matrix(adj_mat)
    #get the number of parameters for the nodes with parents
    num_params = 0
    for node in adj_mat.index:
        #for node with no parents, the number of parameters is equal to the number of such nodes
        if len(pi[node]) == 0:
            num_params += len(discrete_sets[node])-1
        else:
            combos = 1
            for parent in pi[node]:#get possible combinations of the parents
                combos *= len(discrete_sets[parent])
            num_params += combos*(len(discrete_sets[node])-1)
    penalty = np.log(data.shape[0])/2 * num_params
    return penalty

#computes the likelihood over a single node, given the structure and the mle estimates
def compute_node_likelihood(node,counts,cpts,root_counts,root_probs):
    node_parent_cooccur_counts = [k for k in list(counts.keys()) if node == k[0]]
    if node_parent_cooccur_counts:
        return sum([counts[k] * np.log(cpts[k]) for k in node_parent_cooccur_counts])
    else:
        return sum(root_counts[k] * np.log(root_probs[k]) for k in root_counts if node in k)

#compute the log-likelihood of the data give the mle estimate and structures
def compute_likelihood(data,counts,cpts,root_counts,root_probs):
    #likelihood decomposes over the nodes conditioned upon their parents
    node_likelihoods = defaultdict(float)#save the likelihood to prevent duplicate computation in next state
    for node in data.columns:
        node_likelihoods[node] = compute_node_likelihood(node,counts,cpts,root_counts,root_probs)
    return node_likelihoods

#compute the BIC score as sum of likelihood of parameters given data
def compute_bic_score(cp_counts,cpts,root_counts,root_probs,adj_mat,data):
    model_complexity_penalty = compute_penalty(data,adj_mat)
    model_node_likelihoods = compute_likelihood(data,cp_counts,cpts,root_counts,root_probs)
    bic_score = sum(model_node_likelihoods.values()) - model_complexity_penalty
    return bic_score

##compute the log loss as a function of the learned model on test data
## this is the same as the log likelihood for density estimates
def compute_loss(cp_counts,cpts,root_counts,root_probs,test_data):
    model_node_likelihoods = compute_likelihood(test_data, cp_counts, cpts, root_counts, root_probs)
    return (1/test_data.shape[0])*sum(model_node_likelihoods.values())#return the average over the test data

#create a DAG by local transitions(arc addition,arc deletion, arc reversal) on the current structure
def get_neighbour_state(g):
    g_next = g.copy()
    #remove/reverse an existing edge OR add a new edge
    if not len(g.edges()) == 0:
        choice = np.random.choice([1,2,3],1)
    else:
        choice = 3
    if choice == 1:#remove an existing edge
        #check for completely disconnected set
            print("Removing Edge")
            rand_choice = np.random.randint(0,len(g_next.edges()),1)[0]
            g_next.remove_edges_from([g_next.edges()[rand_choice]])
    elif choice == 2:#reverse an existing edge
        print("Reversing Edge")
        existing_edges = g_next.edges()#get existing edges
        rand_choice = np.random.randint(0,len(existing_edges),1)[0]
        random_edge_choice = g_next.edges()[rand_choice]#choose uniformly at random
        g_next.remove_edges_from([random_edge_choice])#remove the edge
        g_next.add_edges_from([random_edge_choice[::-1]])#add its reverse
    elif choice == 3:#add an edge; prevent duplication and cycle creation
        print("Adding Edge")
        all_possible_edges = [x for x in list(product(g_next.nodes(),g_next.nodes())) if x[0]!=x[1]]#self-edges can't be added
        all_possible_edges = [x for x in all_possible_edges if x not in g_next.edges() and x[::-1] not in g_next.edges()]#reverse edge create degenerate cycle
        if all_possible_edges:
            rand_choice = np.random.randint(0,len(all_possible_edges),1)[0]
        else:
            pass
        random_edge_choice = all_possible_edges[rand_choice]#choose uniformly at random
        g_next.add_edges_from([random_edge_choice])
        #check for cycle and remove edge from set of possible neighbours
        while not nx.is_directed_acyclic_graph(g_next):
            g_next = g.copy()
            #remove the edge creating the cycle from future considerations instead of catching the exception with nx.find_cycle()
            all_possible_edges = [x for x in all_possible_edges if x!= all_possible_edges[rand_choice]]
            if len(all_possible_edges) == 0:#break if all edges exhausted
                break
            rand_choice = np.random.randint(0,len(all_possible_edges),1)[0]
            random_edge_choice = all_possible_edges[rand_choice]
            g_next.add_edges_from([random_edge_choice])
    return g_next,pd.DataFrame(data=nx.adjacency_matrix(g_next).todense(),index=g_next.nodes(),columns=g_next.nodes())

#save a dict of the multinomial conditional probabilities
def save_csv_report(dict,filename):
    df = pd.DataFrame.from_dict(dict,orient='index')
    df.to_csv(filename)
    return

#optimizes the BIC score using Metropolis Method with local transitions
def run_metropolis(g, train_data, test_data, n_steps, initial_bic_score):
    #container for losses
    log_loss = []
    bic0 = initial_bic_score.copy()
    bic_best = initial_bic_score.copy()
    g_best = g.copy()
    probs_best = []
    for i in range(n_steps):
        #create a random transition
        g_next,adj_mat_next = get_neighbour_state(g)
        create_pdf_from_nxg(g_next,f'dag_{i}')
        #get the parents
        pi_next = get_parents_from_adj_matrix(adj_mat_next)
        #get the mle estimates
        cp_counts, cpts, root_counts, root_probs = perform_mle(train_data, pi_next)
        probs_best = [cp_counts, cpts, root_counts, root_probs]
        #compute the bic score
        curr_bic = compute_bic_score(cp_counts, cpts, root_counts, root_probs, adj_mat_next, train_data)
        print("previous BIC Score:\t",bic0,"\nNext State BIC Score:\t",curr_bic,"\n")
        #maximize the BIC score
        if curr_bic >= bic0: #adopt the neighbour state if BIC score increases
            g,adj_mat = g_next,adj_mat_next
            bic0 = curr_bic
            log_loss.append(compute_loss(cp_counts,cpts,root_counts,root_probs,test_data))
            #store the best solution
            if curr_bic >= bic_best:
                bic_best = curr_bic
                g_best = g_next
                probs_best = [cp_counts, cpts, root_counts, root_probs]
        else:
            pass
            selection_prob = 1 - bic0/curr_bic
            if np.random.rand() < selection_prob:
                g,adj_mat = g_next,adj_mat_next
                bic0 = curr_bic
                log_loss.append(compute_loss(cp_counts,cpts,root_counts,root_probs,test_data))
            else:
                log_loss.append(compute_loss(cp_counts,cpts,root_counts,root_probs,test_data))
                pass
    #save the best structure found
    create_pdf_from_nxg(g_best,"best_structure")
    save_csv_report(probs_best[1],'CPTS_best')
    save_csv_report(probs_best[3],'root_probabilities_best')
    print("\nprobs bests:",probs_best,"\n")
    return g_next,cp_counts,cpts,root_counts,root_probs,log_loss

def fetch_data(source,n_nodes,n_samples,max_pi):
    if source == 'random':
        # create a random bayes net with given number of nodes
        bn = create_random_bn(n_nodes,max_pi)
        #fetch samples from it
        training_data = bn_sampler(bn,n_samples)
        test_data = bn_sampler(bn,n_samples/2)
        print("Fetched training data from a random bayes net!")

    elif source == 'de_gene_expr':
        ##read top 10 differentially expressed genes discretized expression values
        # gene_expr_data_filename = 'neuron_gene_expr_813_10_named.csv'
        # gene_expr_data_filename = 'neuron_813_10_named_2.csv'
        gene_expr_data_filename = 'pathway.csv'
        gene_expr_data = pd.read_csv(gene_expr_data_filename, index_col=0)
        ##randomly shuffle
        shuffle(gene_expr_data)
        ##split dataset into train and test
        split_idx = int(np.ceil(gene_expr_data.shape[0] / 2))
        training_data = gene_expr_data.iloc[:split_idx + 1, :]
        test_data = gene_expr_data.iloc[split_idx + 1:, :]
        print("Read Gene Expression Data.")

    return training_data,test_data

# # #plots the log-loss over the Model inference procedure
def plot_loss(log_loss,filename):
    plt.plot(log_loss)
    plt.ylabel('Log-Loss',fontsize=12)
    plt.xlabel('Iteration#',fontsize=12)
    plt.title("Learning Belief Networks",fontsize=20)
    plt.savefig(filename)
    return

##runs the bayes net learning routine based on the source
def run_learner(**kwargs):#de_gene_expr is top 10 differentially expressed genes in mouse neuron cell-type
    ##get the appropriate dataset
    training_data,test_data = fetch_data(kwargs['src'],kwargs['nnodes'],kwargs['nsamples'],kwargs['maxp'])

    # ###initialize random structure - using sparse adjacency matrix
    n_nodes = np.shape(training_data)[1]

    # ##create a random DAG
    node_ids = training_data.columns
    g, adj_mat = initialize_random_structure(n_nodes, node_ids)
    create_pdf_from_nxg(g,kwargs['out'])
    print("\nInitialized Random Adjacency matrix\n", adj_mat)
    print("\nGraph:\n",g.nodes(),g.edges())

    # ##compute mle estimates
    cp_counts, cpts, root_counts,root_probs = compute_cpts_mle(adj_mat,training_data)
    print("\nComputed MLE estimates For Initial Structure.\n")

    # ###compute BIC score
    initial_bic_score = compute_bic_score(cp_counts,cpts,root_counts,root_probs,adj_mat,training_data)

    ###metropolis if statement
    print("Running Metropolis Sampler for ",kwargs['niter']," steps.\n")
    (g_next, cp_counts, cpts, root_counts, root_probs, log_loss) = run_metropolis(g,training_data,test_data,kwargs['niter'],initial_bic_score)
    create_pdf_from_nxg(g_next, f'{kwargs["out"]}_final_DAG')

    ##plot the loss function
    plot_loss(log_loss,f'{kwargs["out"]}_log_loss.png')

    #save the cpts and root node probabilities
    save_csv_report(cpts,'CPTs_final.csv')
    save_csv_report(root_probs,'root_probabilies.csv')
    print("Parameters saved as csv files.")

    ##create a GIF of the bayes net DAG transitions
    create_gif(f'{kwargs["out"]}')
    return log_loss

#create a gif file from the metropolis run iterations
def create_gif(filename):
    os.chdir('DAGs')
    cmd = f'convert -delay 20 -loop 0 *.png {filename}.gif'
    args = shlex.split(cmd)
    create_gif_proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    create_gif_proc.wait()
    #remove all the png files
    # for file in glob.glob('*.png'):
    #     os.remove(file)
    os.chdir('..')
    return

if __name__ == '__main__':
    # collect arguments from cmd line
    parser = ArgumentParser()
    parser.add_argument('-source', type=str, help="Enter 'random' to run on random sample or 'de_gene_expr' to run on mouse data.")
    parser.add_argument('-nnodes', type=int, help="Number of samples required! Only req for 'random'.")
    parser.add_argument('-nsamples', type=int, help="Number of samples required! Only req for 'random'")
    parser.add_argument('-maxp', type=int, help="Max number of parents of any node.")
    parser.add_argument('-niter',type=int,help="Number of iterations")
    parser.add_argument('-out', type=str, help="Output Filename")
    parser.add_argument('-verbose', type=bool, help="Print attributes of the random bayes net used for sampling.")
    args = parser.parse_args()

    if not args:
        print('See python bayes_net_learner -h for usage.')
        sys.exit()

    #invoke the bayes net learner with user defined arguments
    args = {'src':args.source,
            'nnodes':args.nnodes,
            'nsamples':args.nsamples,
            'out':args.out,
            'niter':args.niter,
            'maxp':args.maxp}

    run_learner(**args)
    print("\nComplete.")