from bayes_net_learner import run_learner,plot_loss
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

### find performance as a function of nodes in the random case, average 5 runs of the sampler for each dataset
def run_tests():
    ### find the performance on 1000 iterations - single run
    args = {'src':'random',
            'nnodes':10,
            'nsamples':1000,
            'out':'random_10_nodes_1000_iterations',
            'niter':500,
            'maxp':3}
    loss_1000_iters = run_learner(**args)
    plot_loss(loss_1000_iters,args['out'])

    ##find performance on average of 5 runs for number of nodes ranging from 5 to 10 for 1000samples 100 iterations
    loss_per_node = defaultdict(list)
    n_repeats = 5
    for n_nodes in [5, 10, 15]:
        for duplicates in range(n_repeats):
            args['src'] = 'random'
            args['nsamples'] = 10
            args['niter'] = 10
            args['maxp'] = 3
            args['out'] = f'avg_loss_5runs_{n_nodes}_nodes'
            args['nnodes'] = n_nodes
            loss_per_node[n_nodes].append(run_learner(**args))
        print('loss for ', n_nodes, 'nodes:\n', loss_per_node[n_nodes])
        # loss_per_node[n_nodes] = [sum(x) for x in zip(*loss_per_node[n_nodes])]
        avg_loss = np.zeros((n_repeats, 10))
        for i, l in enumerate(loss_per_node[n_nodes]):
            avg_loss[i] = l
        print(avg_loss)
        print(np.sum(avg_loss, axis=0) / len(avg_loss))
        loss_per_node[n_nodes] = avg_loss
    print("Loss per node\n\n", loss_per_node)
    # pd.DataFrame.from_dict(loss_per_node).to_csv('loss_per_node')
    plt.plot(loss_per_node[5], 'r--')
    plt.ylabel('Log-Loss', fontsize=10)
    plt.xlabel('Iteration#', fontsize=10)
    plt.title(f'Log-Loss for 5,10 and 15 nodes', fontsize=18)
    plt.savefig('loss_curves_for_num_nodes')
    plt.show()

if __name__ == '__main__':
    run_tests()
