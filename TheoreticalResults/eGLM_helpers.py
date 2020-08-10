print("Importing eGLM helpers...")
from collections import OrderedDict
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("white")
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale

sys.path.append('../')
sys.path.append('../../utils/')
# Primary module with most model functions
import model

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

def run_ext_glm(all_nodes_ts, task_reg, weight_matrix, g, s, standardize=False): 
    
    """
    Runs extended GLM looping through each node of a network

    Parameters:
        all_nodes_ts = time series of all nodes in the network (DVs for GLM). 2D array with nodes for rows and time points for columns
        task_reg = task regressor (IV for GLM)
        weight_matrix = weight matrix containing the connectivity weights for the network
        g = global information transfer strength
        s = local/self information transfer strength

    Returns: 
        Dictionary with two items
        ext_task_betas = corrected task parameter estimates
        ext_mods = sm.OLS objects from which the task parameters come from
    """
    
    nregions = all_nodes_ts.shape[0]
    ext_task_betas = np.zeros((nregions))
    ext_mods = []
    
    # This is external; don't know how the task affects a node
    i_t = task_reg[:-1]
       
    for node in range(0, nregions):
        #Drop the first time point
        y = all_nodes_ts[node,1:]

        #Drop last time point
        s_phi_x = s*phi(all_nodes_ts[node,:-1])

        g_w_phi_x = np.delete(all_nodes_ts, node, axis=0)[:,:-1]
        g_w_phi_x = np.apply_along_axis(phi, 0, g_w_phi_x)
        cur_w = np.delete(weight_matrix[node,:], node, axis=0)
        cur_w = cur_w.reshape(-1,1)
        g_w_phi_x = cur_w * g_w_phi_x
        g_w_phi_x = np.sum(g_w_phi_x, axis=0)
        g_w_phi_x = g*g_w_phi_x      
        
        #All IVs in design matrix
        mod_df = pd.DataFrame(data = {"y": y, "s_phi_x":s_phi_x, "g_w_phi_x":g_w_phi_x, "i_t":i_t})
        
        s_df = pd.DataFrame(scale(mod_df))
        s_df.rename(columns={i:j for i,j in zip(s_df.columns,mod_df.columns)}, inplace=True)
        
        if standardize:
            ext_mod = smf.ols(formula = 'y ~ s_phi_x + g_w_phi_x + i_t', data = s_df)
        else:
            ext_mod = smf.ols(formula = 'y ~ s_phi_x + g_w_phi_x + i_t', data = mod_df)
        
        ext_res = ext_mod.fit()
        ext_params = ext_res.params

        ext_task_betas[node] = ext_params["i_t"]
        ext_mods.append(ext_mod)
        
    return ({"ext_task_betas": ext_task_betas,
                 "ext_mods": ext_mods})
        
def make_stimtimes(stim_nodes, Tmax=1000, dt=1, stim_mag=.5, tasktiming=None, ncommunities = 3, nodespercommunity = 35,  sa = 50, ea = 100, iv = 200):
    
    """
    Creates task timing and timeseries for all nodes in network

    Parameters:
        Tmax = task length
        dt = sampling rate
        stim_nodes = nodes that are stimulated by the task
        tasktiming = block task array is created if not specified. If specified must be of length Tmax/dt
        ncommunities = number of communities in network
        nodespercommunity = number of nodes per community in network
        sa = start point of stimulation
        ea = end point of stimulation
        iv = interstim interval

    Returns: 
        2D array with nodes in rows and time points in columns
    """
    
    totalnodes = nodespercommunity*ncommunities
    T = np.arange(0,Tmax,dt)
    # Construct timing array for convolution 
    # This timing is irrespective of the task being performed
    # Tasks are only determined by which nodes are stimulated
    if tasktiming is None:
        tasktiming = np.zeros((len(T)))
        for t in range(len(T)):
            if t%iv>sa and t%iv<ea:
                tasktiming[t] = 1.0
    
    stimtimes = np.zeros((totalnodes,len(T)))
    
    # When task is ON the activity for a stim_node at that time point changes the size of stim_mag
    for t in range(len(T)):
        if tasktiming[t] == 1:
            stimtimes[stim_nodes,t] = stim_mag
            
    return(tasktiming, stimtimes)

def sim_network_task_glm(ncommunities = 3, 
                         innetwork_dsity = .60, 
                         outnetwork_dsity = .08, 
                         hubnetwork_dsity = .25, 
                         nodespercommunity = 35, 
                         plot_network = False,
                         dt = 1, tau = 1, g = 1, s = 1, 
                         topdown = True, bottomup = False, 
                         local_com = 1, 
                         Tmax = 1000, 
                         plot_task = False, 
                         stimsize = np.floor(35/3.0), 
                         noise = None,
                         noise_loc = 0, 
                         noise_scale = 0,
                         stim_mag = .5,
                         W = None,
                         taskdata = None,
                         tasktiming = None,
                         sa = 50,
                         ea = 100,
                         iv = 200,
                         standardize=False):
    
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
    
    #############################################
    # Create network
    #############################################

    if W is None:        
        totalnodes = nodespercommunity*ncommunities
        # Construct structural matrix
        S = model.generateStructuralNetwork(ncommunities=ncommunities,
                                            innetwork_dsity=innetwork_dsity,
                                            outnetwork_dsity=outnetwork_dsity,
                                            hubnetwork_dsity=hubnetwork_dsity,
                                            nodespercommunity=nodespercommunity,
                                            showplot=plot_network)
        # Construct synaptic matrix
        W = model.generateSynapticNetwork(S, showplot=plot_network)
        
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
   
    tasktiming, stimtimes = make_stimtimes(Tmax=Tmax, dt=dt, stim_nodes=stim_nodes, stim_mag=stim_mag, 
                   tasktiming=tasktiming, ncommunities = ncommunities, nodespercommunity = nodespercommunity,  
                   sa = sa, ea = ea, iv = iv)

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
        taskdata, error = model.networkModel(W,Tmax=Tmax,dt=dt,g=g,s=s,tau=tau, I=stimtimes, noise=noise, noise_loc = noise_loc, noise_scale = noise_scale)
    
    #Use only a subset of data for GLM's if it's too long
    if taskdata.shape[1]>44999:
        short_lim = int(np.floor(taskdata.shape[1]/3))
        y = copy.copy(taskdata[:,:short_lim])
        task_reg = copy.copy(tasktiming[:short_lim])
    else:
        y = copy.copy(taskdata)
        task_reg = copy.copy(tasktiming)
        
    #############################################
    # Run uncorrected and extended GLM to compare task regressor
    #############################################
    
    ucr_model = run_ucr_glm(all_nodes_ts = y, task_reg = task_reg, standardize=standardize)
    ext_model = run_ext_glm(all_nodes_ts = y, task_reg = task_reg, 
                            weight_matrix = W, g = g, s = s,
                            standardize=standardize)
    
    ucr_betas = ucr_model["ucr_task_betas"]
    ext_betas = ext_model["ext_task_betas"]
    
    ucr_glms = ucr_model["ucr_mods"]
    ext_glms = ext_model["ext_mods"]
    
    return({"W":W, "ucr_betas": ucr_betas, "ucr_glms": ucr_glms, "ext_betas": ext_betas, "ext_glms": ext_glms,
            "stim_nodes": stim_nodes, "taskdata": taskdata, 'tasktiming': tasktiming, 
            'Tmax':Tmax, 'dt':dt, 'g':g, "s":s, 'tau': tau, 'stimtimes': stimtimes})


def get_true_baseline(sim, stim_nodes_only=True):    
    
    """
    Get baselines for stimulated and non stimulated nodes against which GLM results will be compared

    Parameters:
        sim = simulation dictionary output from sim_network_task_glm
        
    Returns: 
        baseline_vec = baseline for all nodes; task regression weights of stimulated nodes for noiseless data
        
    """
    
    baseline_vec = np.zeros(sim['W'].shape[0])
    
    # Initialize parameters
    W = sim['W']
    Tmax= sim['Tmax']
    dt=sim['dt']
    g=sim['g']
    s=sim['s']
    tau=sim['tau'] 
    I=sim['stimtimes']
    stim_nodes = sim['stim_nodes']
    tasktiming = sim['tasktiming']
    
    # To get the baselines I need to run the task in the network without noise
    # To run the task in the network without noise I need all of these parameters: 
    taskdata, error = model.networkModel(W, Tmax=Tmax,dt=dt,g=g,s=s,tau=tau, I=I, noise=None)
    
    # Run the extended glm on this noiseless stimulation
    # Use only a subset of data for GLM's if it's too long
    if taskdata.shape[1]>44999:
        short_lim = int(np.floor(taskdata.shape[1]/3))
        y = copy.copy(taskdata[:,:short_lim])
        task_reg = copy.copy(tasktiming[:short_lim])
    else:
        y = copy.copy(taskdata)
        task_reg = copy.copy(tasktiming)
        
    
    ext_model = run_ext_glm(all_nodes_ts = y, task_reg = task_reg, 
                            weight_matrix = W, g = g, s = s)
    
    # Extract the task regression weights from these extended GLMs for stimulated nodes
    ext_betas = ext_model["ext_task_betas"]
    
    # Replace the value of these stimulated nodes in the baseline_vec
    if stim_nodes_only:
        baseline_vec[stim_nodes] = ext_betas[stim_nodes]
    else: 
        baseline_vec = ext_betas
    
    return(baseline_vec)

def plot_sim_network_glm(data,
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
        data = simulation dictionary output from sim_network_task_glm
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
    stim_nodes = data['stim_nodes']
    
    plt.rcParams["figure.figsize"][0] = width
    plt.rcParams["figure.figsize"][1] = height
    
    plt.plot(data['ucr_betas'], alpha = alp, color = "C0", label = ucr_label)
    plt.plot(data['ext_betas'], alpha = alp, color = "C1", label = ext_label)
    
    baseline_vec = get_true_baseline(data, stim_nodes_only = stim_nodes_only)
  
    plt.plot(baseline_vec, 
     color = "black", linestyle = '--', label = base_label, alpha = alp)
    
    plt.ylabel('Beta',fontsize=14)
    plt.xlabel('Node',fontsize=14)
    
    for n in range(1,ncoms):
        plt.axvline(x=nnods*n,linewidth=2, color='gray', ls = "--")
    
    plt.legend(loc="best")

print("All eGLM helpers imported!")
