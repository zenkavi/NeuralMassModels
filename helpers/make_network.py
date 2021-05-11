from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set default arguments that appear repeatadly in functions below and downstream functions that import this script
default_args = {'alternate_stim_nodes': 0,
                'bottomup': False, 
                'dt':.5,  
                'g':1, 
                'hubnetwork_dsity': .25,
                'I': None,
                'innetwork_dsity': .60,
                'local_com': 1, 
                'ncommunities': 3,
                'noise': None,
                'noise_loc': 0, 
                'noise_scale': 0,
                'nodespercommunity': 1,
                'outnetwork_dsity':.08,
                'plot_network': False,
                'plot_task': False, 
                's':.8,
                'selfcon_loc': -.5, 
                'selfcon_scale': .1,
                'showplot':False,
                'standardize':False,
                'stim_mag':.5,
                'stimsize': 1, 
                'taskdata':None,
                'tasktiming':None,
                'tau':1, 
                'Tmax':1000,
                'topdown':True,
                'W': None,
                'weight_loc': 1.0, 
                'weight_scale': .2}


def phi(x):
    return(np.tanh(x))

def generateStructuralNetwork(args_dict = default_args):
    """
    Randomly generates a structural network with a single hub network

    Parameters:
        ncommunities = number of communities within the network (one will automatically be a hub-network
        innetwork_dsity = connectivity density of within-network connections
        outnetwork_dsity = connectivity density of out-of-network connections
        hubnetwork_dsity = out-of-network connectivity density for the hub-network
        showplot = if set to True, will automatically display the structural matrix using matplotlib.pyplot

    Returns: 
        Unweighted structural connectivity matrix (with 1s indicating edges and 0s otherwise)
    """
    
    # Initialize parameters
    ncommunities=args_dict['ncommunities']
    innetwork_dsity=args_dict['innetwork_dsity']
    outnetwork_dsity=args_dict['outnetwork_dsity']
    hubnetwork_dsity=args_dict['hubnetwork_dsity']
    nodespercommunity=args_dict['nodespercommunity']
    showplot=args_dict['showplot']
    
    totalnodes = nodespercommunity * ncommunities

    S = np.zeros((totalnodes,totalnodes))
    # Construct structural matrix
    nodecount = 0
    for i in range(ncommunities):
        for j in range(ncommunities):
            #for node in range(nodespercommunity):
            # Set within network community connections
            if i==j:
                tmp_a = np.random.rand(nodespercommunity,nodespercommunity)<innetwork_dsity
                indstart = i*nodespercommunity
                indend = i*nodespercommunity+nodespercommunity
                S[indstart:indend,indstart:indend] = tmp_a
            else:
                tmp_b = np.random.rand(nodespercommunity,nodespercommunity)<outnetwork_dsity
                indstart_i = i*nodespercommunity
                indend_i = i*nodespercommunity + nodespercommunity
                indstart_j = j*nodespercommunity
                indend_j = j*nodespercommunity + nodespercommunity
                S[indstart_i:indend_i, indstart_j:indend_j] = tmp_b

    # Redo a community as a hub-network
    hubnetwork = 0
    if hubnetwork_dsity>0: # Only do if we want a hub network
        for i in range(ncommunities):
            for j in range(ncommunities):
                if (i==hubnetwork or j==hubnetwork) and i!=j:
                    tmp_b = np.random.rand(nodespercommunity,nodespercommunity)<hubnetwork_dsity
                    indstart_i = i*nodespercommunity
                    indend_i = i*nodespercommunity + nodespercommunity
                    indstart_j = j*nodespercommunity
                    indend_j = j*nodespercommunity + nodespercommunity
                    S[indstart_i:indend_i, indstart_j:indend_j] = tmp_b

    # Make sure self-connections are 0 as they are added separately in the next step
    np.fill_diagonal(S, 0)

    if showplot:
        S_plot = copy(S)
        plt.figure()
        plt.imshow(W_plot, origin='lower',cmap='bwr')
        plt.title('Structural Matrix', y=1.08)
        plt.xlabel('From')
        plt.ylabel('To')
        plt.colorbar()
        plt.tight_layout()
    
    return S

def generateSynapticNetwork(S, args_dict):
    """
    Generate synaptic matrix over structural matrix with randomized gaussian weighs with
    mean = 1.0 and standard deviation of 0.2 (so all weights are positive)
    
    Parameters:
        W = structural connectivity matrix
        showplot = if set to True, will automatically display the structural matrix using matplotlib.pyplot

    Returns:
        Synaptic matrix with Gaussian weights on top of structural matrix
    """
    showplot = args_dict['showplot']
    weight_loc = args_dict['weight_loc']
    weight_scale = args_dict['weight_scale']
    selfcon_loc = args_dict['selfcon_loc']
    selfcon_scale = args_dict['selfcon_scale']
    
    # Find non-zero connections
    W = np.zeros((S.shape))
    totalnodes = W.shape[0]
    connect_ind = np.where(S!=0)
    nconnects = len(connect_ind[0])
    weights = np.random.normal(loc=weight_loc,scale=weight_scale, size=(nconnects,))
    W[connect_ind] = weights
    
    # Find num connections per node
    # Add self-connection indicators to avoid 0's in the denominator of degree calculation
    np.fill_diagonal(S, 1)
    nodeDeg = np.sum(S,axis=1)

    # Synaptic scaling according to number of incoming connections
    for col in range(W.shape[1]):
        W[:,col] = np.divide(W[:,col],np.sqrt(nodeDeg))
        
    # Add inhibitory self-connections
    self_cons = np.random.normal(loc=selfcon_loc, scale=selfcon_scale, size=(totalnodes,))
    np.fill_diagonal(W, self_cons)

    if showplot:
        plt.figure()
        plt.imshow(W, origin='lower')
        plt.colorbar()
        plt.title('Synaptic Weight Matrix -- Coupling Matrix', y=1.08)
        plt.xlabel('Regions')
        plt.ylabel('Regions')
        plt.tight_layout()
        
    return W

def networkModel(W, args_dict = default_args):
    """
    G = Synaptic Weight Matrix    
    
    """
    # Initialize parameters
    Tmax=args_dict['Tmax']
    dt=args_dict['dt']
    g=args_dict['g']
    s=args_dict['s']
    tau=args_dict['tau']
    I=args_dict['I']
    noise=args_dict['noise']
    noise_loc = args_dict['noise_loc']
    noise_scale = args_dict['noise_scale']
       
    T = np.arange(0, Tmax, dt)
    totalnodes = W.shape[0]
  
    # External input (or task-evoked input) && noise input
    if I is None: I = np.zeros((totalnodes,len(T)))
    # Noise parameter
    if noise is None: noise = np.zeros((totalnodes,I.shape[1]))
    elif noise == 1: noise = np.random.normal(size=(totalnodes,I.shape[1]), loc = noise_loc, scale = noise_scale)

    # Initial conditions and empty arrays
    Enodes = np.zeros((totalnodes,I.shape[1]))
    # Initial conditions
        # Initial condition 0 if there is a task simulation
    if I is not None:
        Einit = np.zeros((totalnodes,))
    else:
        Einit = np.random.rand(totalnodes,)
    
    Enodes[:,0] = Einit

    spont_act = np.zeros((totalnodes,))
    
    for t in range(I.shape[1]-1):

        ## Solve using Runge-Kutta Order 2 Method
        spont_act = (noise[:,t] + I[:,t])
        k1e = -Enodes[:,t] + g*np.dot(W,phi(Enodes[:,t]))
        k1e += s*phi(Enodes[:,t]) + spont_act
        k1e = k1e/tau
        # 
        ave = Enodes[:,t] + k1e*dt
        #
        spont_act = (noise[:,t+1] + I[:,t+1])
        k2e = -ave + g*np.dot(W,phi(ave))
        k2e += s*phi(ave) + spont_act
        k2e = k2e/tau

        Enodes[:,t+1] = Enodes[:,t] + (.5*(k1e+k2e))*dt

    return (Enodes, noise)
