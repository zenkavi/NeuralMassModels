import numpy as np
import statsmodels.api as sm

def get_multreg_fc_wci(activity_matrix, network):
    
    nnodes = activity_matrix.shape[0]
    
    multreg_fc = np.zeros((nnodes,nnodes))
    multreg_fc_lb = np.zeros((nnodes,nnodes))
    multreg_fc_ub = np.zeros((nnodes,nnodes))
    
    for targetnode in range(nnodes):
        othernodes = list(range(nnodes))
        othernodes.remove(targetnode) # Remove target node from 'other nodes'
        X = activity_matrix[othernodes,:].T
        X = sm.add_constant(X)
        y = activity_matrix[targetnode,:]
        mod = sm.OLS(y, X)
        res = mod.fit()
        
        # Select starting from 1 to exclude intercept
        multreg_fc[targetnode,othernodes]=res.params[1:]
        multreg_fc_lb[targetnode,othernodes] = [x[0] for x in res.conf_int()[1:]]
        multreg_fc_ub[targetnode,othernodes] = [x[1] for x in res.conf_int()[1:]]
    
    # Compute RMSE
    multreg_fc_rmse = np.sqrt(np.mean(np.square(multreg_fc - network['W'])))
    
    return multreg_fc, multreg_fc_lb, multreg_fc_ub, multreg_fc_rmse