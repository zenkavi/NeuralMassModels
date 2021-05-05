print("Importing eGLM helpers...")
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale

from eGLM_model import default_args, generateStructuralNetwork, generateSynapticNetwork, networkModel

def phi(x): 
    return(np.tanh(x))

def run_ucr_glm(all_nodes_ts, task_reg, standardize=False):
    
    """
    Runs classical GLM looping through each node of a network

    Parameters:
        all_nodes_ts = time series of all nodes in the network (DVs for GLM). 2D array with nodes for rows and time points for columns
        task_reg = task regressor (IV for GLM)

    Returns: 
        Dictionary with two items
        ucr_task_betas = uncorrected task parameter estimates
        ucr_mods = sm.OLS objects from which the task parameters come from
    """
    
    nregions = all_nodes_ts.shape[0]
    ucr_task_betas = np.zeros((nregions))
    ucr_mods = []
    
    for region in range(0, nregions):
        cur_y = all_nodes_ts[region,:]
        if standardize:
            ucr_mod = sm.OLS(scale(cur_y), scale(task_reg)) 
        else:
            ucr_mod = sm.OLS(cur_y, task_reg)
        ucr_res = ucr_mod.fit()
        ucr_task_betas[region] = ucr_res.params[0]
        ucr_mods.append(ucr_mod)
    
    return ({"ucr_task_betas":ucr_task_betas,
             "ucr_mods": ucr_mods})

def run_ext_glm(all_nodes_ts, args_dict, task_reg, inc_net_act=True, inc_self_stim=True): 
    
    """
    Runs extended GLM looping through each node of a network

    Parameters:
        all_nodes_ts = time series of all nodes in the network (DVs for GLM). 2D array with nodes for rows and time points for columns
        args_dict = dictionary with parameters and adjacency matrix used for the simulation of the timeseries
        task_reg = task regressor (IV for GLM)

    Returns: 
        Dictionary with two items
        ext_task_betas = corrected task parameter estimates
        ext_mods = sm.OLS objects from which the task parameters come from
    """
    
    s = args_dict['s']
    dt = args_dict['dt']
    tau = args_dict['tau']
    weight_matrix = args_dict['W']
    
    nregions = all_nodes_ts.shape[0]
    ext_task_betas = np.zeros((nregions))
    ext_mods = []
    
    # This is external; don't know how the task affects a node
    i_t = task_reg[:-1]
       
#     for node in range(0, nregions):
        #Drop the first time point
        
    return ({"ext_task_betas": ext_task_betas,
                 "ext_mods": ext_mods})
        
def make_stimtimes(stim_nodes, args_dict = default_args):
    
    """
    Creates task timing and timeseries for all nodes in network

    Parameters:
        Tmax = task length
        dt = sampling rate
        stim_nodes = nodes that are stimulated by the task
        tasktiming = block task array is created if not specified. If specified must be of length Tmax/dt
        ncommunities = number of communities in network
        nodespercommunity = number of nodes per community in network
        on_len = duration of stimulus
        off_len = duration of off period

    Returns: 
        2D array with nodes in rows and time points in columns
    """
    
    # Initialize parameters
    Tmax=args_dict['Tmax']
    dt=args_dict['dt']
    stim_mag=args_dict['stim_mag']
    tasktiming=args_dict['tasktiming']
    ncommunities = args_dict['ncommunities']
    nodespercommunity = args_dict['nodespercommunity']
    on_len = int(np.ceil(args_dict['on_len']/dt))
    off_len = int(np.ceil(args_dict['off_len']/dt))
    alternate_stim_nodes = args_dict['alternate_stim_nodes']
    
    totalnodes = nodespercommunity*ncommunities
    
    T = np.arange(0,Tmax,dt)
    num_blocks = int(np.ceil(len(T)/(on_len + 2*off_len)))
    
    # Construct timing 
    if tasktiming is None:
        tasktiming = np.tile(np.concatenate([np.zeros(off_len), np.ones(on_len), np.zeros(off_len)]), num_blocks)
    
    stimtimes = np.zeros((totalnodes,len(T)))
    
    # When task is ON the activity for a stim_node at that time point changes the size of stim_mag
    for t in range(len(T)):
        if tasktiming[t] == 1:
            stimtimes[stim_nodes,t] = stim_mag
            
    # Calculate number of blocks for each node ff you have alternating stim nodes
    num_node_blocks = np.floor(len(T)/len(stim_nodes))
    node_stim_dur = num_node_blocks*(on_len + 2*off_len)
    
    # Make sure the tasktiming and stimtime length are the same
    new_task_len = int((node_stim_dur)*len(stim_nodes))
    tasktiming = tasktiming[:new_task_len]
    stimtimes = [x[:new_task_len] for x in stimtimes]
    
    # Turn other nodes off when one node is on
    for i in range(totalnodes):
        off_nodes = [x for x in stim_nodes if x != i]
        stimtimes[off_nodes,i*node_stim_dur:node_stim_dur*(i+1)-1] = 0 
            
    return(tasktiming, stimtimes)

def sim_network_task_glm(args_dict = default_args):
    
    """
    Simulates task activity in network and runs both uncorrected and corrected GLMs to estimate task parameters

    Parameters:
        ncommunities = number of communities in network (needed even if W is provided)
        innetwork_dsity = probability of a node being connected to another in the same community
        outnetwork_dsity = probability of a node being connected to another in a different (non-hub) community
        hubnetwork_dsity = probability of a node being connected to another in a hub community
        nodespercommunity = number of nodes per community in network (needed even if W is provided)
        plot_network = make plot showing the network connectivity weight matrix
        dt = sampling rate
        tau = time constany
        g = global information transfer strength
        s = self information transfer strength
        topdown = Topdown task that stimulates hub network only
        bottomup = Bottom up task that stimulates local network only
        local_com = local community index
        Tmax = task length
        plot_task = Make plot of task timing
        stimsize = number of nodes that will be stimulated by the task
        noise = Will noise be added to the timeseries
        noise_loc = Mean of normal distribution noise will be drawn from
        noise_scale = SD of normal distribution noise will be drawn from
        stim_mag = magnitude of stimulation
        W = alternative pre-specified weight matrix
        tasktiming = block task array is created if not specified. If specified must be of length Tmax/dt
        sa = start point of stimulation
        ea = end point of stimulation
        iv = interstim interval
        standardize = Should the timeseries and the design matrix be standardized for the GLMs
        
    Returns: 
        Dictionary with weight matrix, corrected and uncorrected task parameters and model objects, stimulated nodes, timeseries for all nodes
    """
    
    
    # Initialize parameters
    ncommunities = args_dict['ncommunities']
    innetwork_dsity = args_dict['innetwork_dsity']
    outnetwork_dsity = args_dict['outnetwork_dsity']
    hubnetwork_dsity = args_dict['hubnetwork_dsity']
    nodespercommunity = args_dict['nodespercommunity']
    plot_network = args_dict['plot_network']
    dt = args_dict['dt']
    tau = args_dict['tau']
    g = args_dict['g']
    s = args_dict['s']
    topdown = args_dict['topdown']
    bottomup = args_dict['bottomup'] 
    local_com = args_dict['local_com']
    Tmax = args_dict['Tmax']
    plot_task = args_dict['plot_task']
    stimsize = args_dict['stimsize']
    noise = args_dict['noise']
    noise_loc = args_dict['noise_loc']
    noise_scale = args_dict['noise_scale']
    stim_mag = args_dict['stim_mag']
    W = args_dict['W']
    taskdata = args_dict['taskdata']
    tasktiming = args_dict['tasktiming']
    sa = args_dict['sa']
    ea = args_dict['ea']
    iv = args_dict['iv']
    standardize=args_dict['standardize']
    
    
    #############################################
    # Create network
    #############################################

    if W is None:        
        totalnodes = nodespercommunity*ncommunities
        # Construct structural matrix
        S = generateStructuralNetwork(args_dict = args_dict)
        # Construct synaptic matrix
        W = generateSynapticNetwork(S, showplot=plot_network)
        
    else:
        totalnodes = W.shape[0]

    if plot_network:
        plt.rcParams["figure.figsize"][0] = 5
        plt.rcParams["figure.figsize"][1] = 4
        sns.heatmap(W, xticklabels=False, yticklabels=False)
        plt.xlabel('Regions')
        plt.ylabel('Regions')
        plt.title("Synaptic Weight Matrix")

    #############################################
    # Get stim nodes
    #############################################
    
    # Construct a community affiliation vector
    Ci = np.repeat(np.arange(ncommunities),nodespercommunity) 
    # Identify the regions associated with the hub network (hub network is by default the 0th network)
    hub_ind = np.where(Ci==0)[0]
    
    if topdown:
        stim_nodes_td = np.arange(0, stimsize, dtype=int)
    else:
        stim_nodes_td = None
    
    if bottomup:
        # Identify indices for one of the local communities
        local_ind = np.where(Ci==local_com)[0] 
        # Identify efferent connections from local network to hub network
        W_mask = np.zeros((W.shape))
        W_mask[local_ind,hub_ind] = 1.0
        local2hub_connects = np.multiply(W,W_mask)
        local_regions_wcon = np.where(local2hub_connects!=0)[0]
        local_regions_ncon = np.setdiff1d(local_ind,local_regions_wcon)
        #If there are enough nodes in the local community with hub connections:
        if len(local_regions_wcon)>= np.floor(stimsize/2):
            #Half of the stimulated local community nodes have hub connections while the other does not
            stim_nodes_bu = np.hstack((np.random.choice(local_regions_ncon, int(np.floor(stimsize/2)), replace=False),
                                np.random.choice(local_regions_wcon, int(stimsize-np.floor(stimsize/2)), replace=False)))
        else:
            stim_nodes_bu = np.hstack((np.random.choice(local_regions_wcon, len(local_regions_wcon), replace=False),
                                np.random.choice(local_regions_ncon, int(stimsize-len(local_regions_wcon)), replace=False)))
    else:
        stim_nodes_bu = None
    
    if stim_nodes_td is not None and stim_nodes_bu is not None:
        stim_nodes = np.hstack((stim_nodes_td, stim_nodes_bu))
    elif stim_nodes_td is not None and stim_nodes_bu is None:
        stim_nodes = stim_nodes_td
    else:
        stim_nodes = stim_nodes_bu
    
    #############################################
    # Make task and node stimulus timing
    #############################################
   
    tasktiming, I = make_stimtimes(stim_nodes=stim_nodes, args_dict = args_dict)
    net_args = copy(args_dict)
    net_args.update({'I': I})

    if plot_task:
        if len(T)>9999:
            plt.plot(T[:10000], tasktiming[0,:10000])
            plt.ylim(top = 1.2, bottom = -0.1)
        else:
            plt.plot(T, tasktiming[0,:])
            plt.ylim(top = 1.2, bottom = -0.1)

    #############################################
    # Make task data (timeseries for all nodes)
    #############################################
    
    if taskdata is None:
        taskdata, error = networkModel(W, args_dict=net_args)
    
    #Use only a subset of data for GLM's if it's too long
    if taskdata.shape[1]>44999:
        short_lim = int(np.floor(taskdata.shape[1]/3))
        y = copy(taskdata[:,:short_lim])
        task_reg = copy(tasktiming[:short_lim])
    else:
        y = copy(taskdata)
        task_reg = copy(tasktiming)
        
    #############################################
    # Run uncorrected and extended GLM to compare task regressor
    #############################################
    
    ucr_model = run_ucr_glm(all_nodes_ts = y, task_reg = task_reg, standardize=standardize)
    ext_model = run_ext_glm(all_nodes_ts = y, task_reg = task_reg, weight_matrix = W, g = g, s = s, standardize=standardize)
    
    ucr_betas = ucr_model["ucr_task_betas"]
    ext_betas = ext_model["ext_task_betas"]
    
    ucr_glms = ucr_model["ucr_mods"]
    ext_glms = ext_model["ext_mods"]
        
    out = copy(net_args)
    out.update({"W":W, "ucr_betas": ucr_betas, "ucr_glms": ucr_glms, "ext_betas": ext_betas, "ext_glms": ext_glms, "stim_nodes": stim_nodes, "taskdata": taskdata, 'tasktiming': tasktiming})
    
    return(out)

def get_true_baseline(sim, stim_nodes_only=True):    
    
    """
    Get baselines for stimulated and non stimulated nodes against which GLM results will be compared

    Parameters:
        sim = simulation dictionary output from sim_network_task_glm
        
    Returns: 
        baseline_vec = baseline for all nodes; task regression weights of stimulated nodes for noiseless data
        
    """
    
    baseline_vec = np.zeros(sim['W'].shape[0])
    
    # To get the baselines run the task in the network without noise
    nonoise_sim = copy(sim)
    nonoise_sim.update({'noise': None})
    taskdata, _ = networkModel(sim['W'], args_dict = nonoise_sim)
    
    # Run the extended glm on this noiseless stimulation
    # Initialize parameters
    tasktiming = sim['tasktiming']
    if taskdata.shape[1]>44999:
        short_lim = int(np.floor(taskdata.shape[1]/3))
        y = copy(taskdata[:,:short_lim])
        task_reg = copy(tasktiming[:short_lim])
    else:
        y = copy(taskdata)
        task_reg = copy(tasktiming)
        
    ext_model = run_ext_glm(all_nodes_ts = y, task_reg = task_reg, 
                            weight_matrix = sim['W'], g = sim['g'], s = sim['s'])
    
    # Extract the task regression weights from these extended GLMs for stimulated nodes
    ext_betas = ext_model["ext_task_betas"]
    
    # Replace the value of these stimulated nodes in the baseline_vec
    if stim_nodes_only:
        stim_nodes = sim['stim_nodes']
        baseline_vec[stim_nodes] = ext_betas[stim_nodes]
    else: 
        baseline_vec = ext_betas
    
    return(baseline_vec)

def plot_sim_network_glm(sim,
                         ax = None,
                         width = 8,
                         height = 6,
                         ncoms = 3,
                         nnods = 35,
                         task_type = "td",
                         ucr_label = "cGLM (baseline)",
                         ext_label = "eGLM (baseline)",
                         base_label = None,
                         alp = 1, 
                         stim_nodes_only=True):
    
    """
    Plotting wrapper comparing cGLM to eGLM results

    Parameters:
        sim = simulation dictionary output from sim_network_task_glm
        width = width of figure
        height = height of figure
        ncoms = number of communities
        nnodes = number of nodes per community
        task_type = "td" for topdown (not extended to include "bu" yet)
        ucr_label = label for cGLM results
        ext_label = label for eGLM results
        base_label = label for baseline
        alp = opacity level 
        stim_nodes = nodes that are stimulated by the task
        tasktiming = block task array is created if not specified. If specified must be of length Tmax/dt
        sa = start point of stimulation
        ea = end point of stimulation
        iv = interstim interval

    Returns: 
        inline plot with nodes on the x axis and task parameter estimates on the y axis colored by GLM type
    """
    
    totalnodes = ncoms*nnods
    stim_nodes = sim['stim_nodes']
    
    if ax is None:
        ax = plt.gca()
        plt.rcParams["figure.figsize"][0] = width
        plt.rcParams["figure.figsize"][1] = height
    
    ax.plot(sim['ucr_betas'], alpha = alp, color = "C0", label = ucr_label)
    ax.plot(sim['ext_betas'], alpha = alp, color = "C1", label = ext_label)
    
    baseline_vec = get_true_baseline(sim, stim_nodes_only = stim_nodes_only)
  
    ax.plot(baseline_vec, color = "black", linestyle = '--', label = base_label, alpha = alp)
    
    ax.set_ylabel('Beta',fontsize=14)
    ax.set_xlabel('Node',fontsize=14)
    
    for n in range(1,ncoms):
        ax.axvline(x=nnods*n,linewidth=2, color='gray', ls = "--")
    
    ax.legend(loc="best")

print("All eGLM helpers imported!")
