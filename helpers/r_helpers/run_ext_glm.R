run_ext_glm_node = function(node, all_nodes_ts, s, task_reg=NULL){
  
  x_t_dt = net_dat_debug$Enodes[node,-1]
  
  x_t = net_dat_debug$Enodes[node, -ncol(net_dat_debug$Enodes)]
  
  g_N_t = net_dat_debug$int_out$net_act1[node,]
  
  s_phi_x_t = s * phi(x_t)
  
  if(is.null(task_reg)){
    I_t = net_dat_debug$int_out$spont_act1[node,]
    I_t_dt = net_dat_debug$int_out$spont_act2[node,]
  } else {
    I_t = task_reg[-1]
    I_t_dt = task_reg[-length(task_reg)]
  }
  
  g_N_t_dt = net_dat_debug$int_out$net_act2[node,]
  
  s_phi_ave = s * phi((const_2_1*x_t)+((dt/tau)*(g_N_t+s_phi_x_t+I_t)))
  
  mod = lm(x_t_dt ~ -1 +x_t + g_N_t + s_phi_x_t + I_t + g_N_t_dt + s_phi_ave + I_t_dt)
  
  return(mod)
}

run_ext_glm = function(all_nodes_ts, task_reg=NULL){
  
  num_nodes = dim(all_nodes_ts$Enodes)[1]
  
  out = list(ext_task_betas = rep(NA, num_nodes),
             ext_mods = rep(NA, num_nodes))
  
  for(node in 1:num_nodes){
    mod = run_ext_glm_node(node, all_nodes_ts, task_reg=task_reg)
    out$ext_mods[node] = mod
    out$ext_task_betas[node] = coef(mod)$I_t_dt
  }
  
  return(out)
}