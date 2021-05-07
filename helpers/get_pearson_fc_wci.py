import numpy as np
from scipy import stats

def get_pearson_fc_wci(activity_matrix, network, alpha = 0.05):
    
    pearson_fc = np.corrcoef(activity_matrix)
    
    # Set diagonals to 0 temporarily to avoid error during Fisher conversion
    np.fill_diagonal(pearson_fc, 0)
    
    # Compute lower and upper bounds for CI
    fisher_z = np.arctanh(pearson_fc)
    se = 1/np.sqrt(activity_matrix.shape[1]-3)
    z = stats.norm.ppf(1-alpha/2)
    lb_z, ub_z = fisher_z-z*se, fisher_z+z*se
    pearson_fc_lb = np.tanh(lb_z)
    pearson_fc_ub = np.tanh(ub_z)    
    
    # Set diagonals back to 1
    np.fill_diagonal(pearson_fc, 1)
    np.fill_diagonal(pearson_fc_lb, 1)
    np.fill_diagonal(pearson_fc_ub, 1)
    
    # Compute RMSE
    pearson_fc_rmse = np.sqrt(np.mean(np.square(pearson_fc - network['W'])))
    
    return pearson_fc, pearson_fc_lb, pearson_fc_ub, pearson_fc_rmse