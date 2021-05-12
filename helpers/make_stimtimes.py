import numpy as np
import pandas as pd

from make_network import default_args
        
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
    
    # If tasktiming ended up being longer than T
    if len(tasktiming) > len(T):
        T = np.arange(0,len(tasktiming)*dt,dt)
        args_dict.update({'Tmax': len(T)})
    
    stimtimes = np.zeros((totalnodes,len(T)))
    
    # When task is ON the activity for a stim_node at that time point changes the size of stim_mag
    for t in range(len(T)):
        if tasktiming[t] == 1:
            stimtimes[stim_nodes,t] = stim_mag
               
    if alternate_stim_nodes:
        # Calculate number of blocks for each node if you have alternating stim nodes
        num_node_blocks = np.floor(len(T)/(on_len + 2*off_len)/len(stim_nodes))
        node_stim_dur = int(num_node_blocks*(on_len + 2*off_len))

        # Make sure the tasktiming and stimtime length are the same
        new_task_len = int((node_stim_dur)*len(stim_nodes))
        tasktiming = tasktiming[:new_task_len]
        stimtimes = np.array([x[:new_task_len] for x in stimtimes])

        # Turn other nodes off when one node is on
        for i in stim_nodes:
            off_nodes = [x for x in stim_nodes if x != i]
            off_range_start = int(i*node_stim_dur)
            off_range_end = int(node_stim_dur*(i+1)-1)
            stimtimes[off_nodes, off_range_start:off_range_end] = 0 
            
    return(tasktiming, stimtimes, args_dict)