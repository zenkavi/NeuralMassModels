import numpy as np
from scipy import stats

def get_multreg_fc_wci(activity_matrix, network):
    
    # Compute RMSE
    multreg_fc_rmse = np.sqrt(np.mean(np.square(multreg_fc - network['W'])))
    
    return multreg_fc, multreg_fc_lb, multreg_fc_ub, multreg_fc_rmse