run_ext_glm_node = function(node, all_nodes_ts, args_dict, task_reg=NULL){
  
  s = args_dict$s
  dt = args_dict$dt
  tau = args_dict$tau
  
  #if debug output it provided
  if(length(all_nodes_ts) == 2){
    
    x_t_dt = all_nodes_ts$Enodes[node,-1]
    
    x_t = all_nodes_ts$Enodes[node, -ncol(all_nodes_ts$Enodes)]
    
    # Because the network activity regressor is extracted from the debug output 
    # changing the connectivity matrix W in args_dict does not affect the results
    g_N_t = all_nodes_ts$int_out$net_act1[node,]
    
    s_phi_x_t = s * phi(x_t)
    
    if(is.null(task_reg)){
      I_t = all_nodes_ts$int_out$spont_act1[node,]
      I_t_dt = all_nodes_ts$int_out$spont_act2[node,]
    } else {
      I_t = task_reg[-length(task_reg)]
      I_t_dt = task_reg[-1]
    }
    
    g_N_t_dt = all_nodes_ts$int_out$net_act2[node,]
    
    s_phi_ave = s * phi(((1 - (dt/tau))*x_t)+((dt/tau)*(g_N_t+s_phi_x_t+I_t))) 
    
  } else { #if only timeseries data is provided
    
    x_t_dt = all_nodes_ts[node,-1]
    
    x_t = all_nodes_ts[node, -ncol(all_nodes_ts)]
    
    g = args_dict$g
    W = args_dict$W
    
    net_act = g*(W[node,] %*% phi(all_nodes_ts))
    g_N_t = net_act[-ncol(all_nodes_ts)]
    
    s_phi_x_t = s * phi(x_t)
    
    I_t = task_reg[-length(task_reg)]
    I_t_dt = task_reg[-1]
    
    g_N_t_dt = net_act[-1]
  
    s_phi_ave = s * phi(((1 - (dt/tau))*x_t)+((dt/tau)*(g_N_t+s_phi_x_t+I_t)))
  }
  
  mod = lm(x_t_dt ~ -1 +x_t + g_N_t + s_phi_x_t + I_t + g_N_t_dt + s_phi_ave + I_t_dt)
  
  out = list(mod_df = mod$model, coef = coef(mod)['I_t_dt'])
  
  return(out)
}

run_ext_glm = function(all_nodes_ts, args_dict, task_reg=NULL){
  
  
  if(length(all_nodes_ts) == 2){
    num_nodes = dim(all_nodes_ts$Enodes)[1]
  } else {
    num_nodes = dim(all_nodes_ts)[1]
  }
  
  out = list(ext_task_betas = rep(NA, num_nodes),
             ext_mods = list())
  
  for(node in 1:num_nodes){
    node_out = run_ext_glm_node(node, all_nodes_ts, args_dict, task_reg)
    out$ext_mods[[node]] = node_out$mod_df
    out$ext_task_betas[node] = node_out$coef
  }
  
  return(out)
}