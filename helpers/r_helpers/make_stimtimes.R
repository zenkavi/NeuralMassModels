make_stimtimes = function(stim_nodes, args_dict){
  
  # """
  #   Creates task timing and timeseries for all nodes in network
  #   Parameters specified in args_dict:
  #       Tmax = task length
  #       dt = sampling rate
  #       stim_nodes = nodes that are stimulated by the task
  #       tasktiming = block task array is created if not specified. 
  #       ncommunities = number of communities in network
  #       nodespercommunity = number of nodes per community in network
  #       sa = start point of stimulation
  #       ea = end point of stimulation
  #       iv = interstim interval
  #       W = adjacency matrix (prioritized over ncommunities and nodespercommunity)
  #   Returns: 
  #       tasktiming = 1D array of resampled tasktiming depending on dt 
  #       stimtimes = 2D array with nodes in rows and time points in columns
  #   """
  
  # Initialize parameters
  Tmax=args_dict$Tmax
  dt=args_dict$dt
  stim_mag=args_dict$stim_mag
  
  # Check if there is an adjacency matrix specified 
  if(is.null(args_dict$W)){
    ncommunities = args_dict$ncommunities
    nodespercommunity = args_dict$nodespercommunity
    totalnodes = nodespercommunity*ncommunities
  } else {
    totalnodes = dim(args_dict$W)[1]
  }
  
  # resampled task length depending on sampling rate dt
  # TT = seq(1,Tmax,dt)
  TT = seq(0,Tmax,dt)
  
  
  # Check if there is a taasktiming specified in the inputs 
  tasktiming=args_dict$tasktiming
  if (is.null(tasktiming)){    
    # If not task specified create task timing
    tasktiming = rep(0, length(TT))
    sa = args_dict$sa
    ea = args_dict$ea
    iv = args_dict$iv
    for (t in 1:(length(TT))){
      if (t%%iv>sa & t%%iv<ea){
        tasktiming[t] = 1.0
      }
    }
  }else if(length(tasktiming) != length(TT)){
    short_tasktiming = tasktiming
    tasktiming = rep(NA, length(TT))
    cur_t = 1
    
    # resample input tasktiming to Tmax/dt
    for(i in 1:length(short_tasktiming)){
      while(floor(cur_t)<length(short_tasktiming)){
        cur_stim = short_tasktiming[floor(cur_t)]
        first_na_index = which(is.na(tasktiming))[1]
        tasktiming[first_na_index] = cur_stim
        cur_t = cur_t + dt
      }
    }
    tasktiming[which(is.na(tasktiming))] = 0
  }
  
  stimtimes = matrix(0, totalnodes, length(TT))
  # When task is ON the activity for a stim_node at that time point changes the size of stim_mag
  for (t in 1:length(TT)){
    if (tasktiming[t] == 1){
      stimtimes[stim_nodes,t] = stim_mag
    }
  }
  
  return(list(tasktiming = tasktiming, stimtimes = stimtimes))
}
