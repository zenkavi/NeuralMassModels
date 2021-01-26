run_ext_glm_node = function(node, all_nodes_ts, args_dict, task_reg=NULL, inc_net_act=TRUE, inc_self_stim=TRUE){
  
  s = args_dict$s
  dt = args_dict$dt
  tau = args_dict$tau
  
  #if only timeseries data is provided
  x_t_dt = all_nodes_ts[node,-1]
  
  x_t = all_nodes_ts[node, -ncol(all_nodes_ts)]
  
  g = args_dict$g
  W = args_dict$W
  
  # 3x3 . 3x201 = 1x201 network activity to be added for all node at each time point
  net_act = g*(W %*% phi(all_nodes_ts)) 
  
  # Drop last time point for network activity that'll be added from given time point to current node only 
  g_N_t = net_act[node, -ncol(all_nodes_ts)]
  
  s_phi_x_t = s * phi(x_t)
  
  I_t = task_reg[-length(task_reg)]
  I_t_dt = task_reg[-1]
  
  # for ave to be for all nodes k1e needs to be for all nodes as well
  k1e = -all_nodes_ts + net_act + s * phi(all_nodes_ts) + args_dict$I
  k1e = k1e/tau
  # ave needs to be for all nodes
  ave = all_nodes_ts + k1e*dt
  g_N_t_dt = g*(W[node,] %*% phi(ave))
  g_N_t_dt = g_N_t_dt[-ncol(all_nodes_ts)]
  
  s_phi_ave = s * phi(((1 - (dt/tau))*x_t)+((dt/tau)*(g_N_t+s_phi_x_t+I_t)))
  
  
  if (inc_net_act[1]==TRUE){
    if (inc_self_stim[1]==TRUE){
      mod = lm(x_t_dt ~ -1 + x_t + g_N_t + s_phi_x_t + I_t + g_N_t_dt + s_phi_ave + I_t_dt)
    } else {
      mod = lm(x_t_dt ~ -1 + x_t + g_N_t + I_t + g_N_t_dt + I_t_dt)
    }
  } else {
    mod = lm(x_t_dt ~ -1 + x_t + s_phi_x_t + I_t + I_t_dt)
  } #if both are false then it's uncorrected GLM (so not including that condition in this function)
  
  out = list(mod_df = mod$model, coef = coef(mod)['I_t_dt'])
  
  return(out)
}

run_ext_glm = function(all_nodes_ts, args_dict, task_reg=NULL, inc_net_act = TRUE, inc_self_stim = TRUE){
  
  num_nodes = dim(all_nodes_ts)[1]
  
  
  out = list(ext_task_betas = rep(NA, num_nodes),
             ext_mods = list())
  
  for(node in 1:num_nodes){
    node_out = run_ext_glm_node(node, all_nodes_ts, args_dict, task_reg, inc_net_act = inc_net_act, inc_self_stim=inc_self_stim)
    out$ext_mods[[node]] = node_out$mod_df
    out$ext_task_betas[node] = node_out$coef
  }
  
  return(out)
}