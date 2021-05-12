import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import scale

def run_glm(all_nodes_ts, task_reg, standardize=False, intercept=True):
    
    """
    Runs classical GLM looping through each node of a network
    Parameters:
        all_nodes_ts = time series of all nodes in the network (DVs for GLM). 2D array with nodes for rows and time points for columns
        task_reg = task regressor (IV for GLM)
    Returns: 
        Np array with betas
    """
    
    nregions = all_nodes_ts.shape[0]
    betas = np.zeros((nregions))
    
    for region in range(0, nregions):
        y = all_nodes_ts[region,:]
        
        if standardize:
            y = scale(y)
            X = scale(task_reg)
        else:
            X = task_reg
            
        if intercept:
            X = sm.add_constant(X)
        
        mod = sm.OLS(y, X)
        res = mod.fit()
        
        if intercept: 
            betas[region] = res.params[1]
        else:
            betas[region] = res.params[0]
        
    return betas