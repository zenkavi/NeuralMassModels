import numpy as np

def actflow(task_betas, fc):
    
    num_nodes = task_betas.shape[0]
    actflow_pred = np.zeros(num_nodes)
    
    for cur_node in range(num_nodes):
        other_nodes=list(range(num_nodes))
        other_nodes.remove(cur_node)
        actflow_pred[cur_node]=np.sum(task_betas[other_nodes]*fc[cur_node,other_nodes])
    
    return actflow_pred
