check_net_resids = function(node, all_nodes_ts, args_dict, excl_I = FALSE, verbose=FALSE){
  
  x_t_dt = all_nodes_ts$Enodes[node,-1]
  dt = args_dict$dt
  tau = args_dict$tau
  const1 = 1 - (dt/tau) +  ((dt^2)/(2*tau^2))
  x_t = all_nodes_ts$Enodes[node, -ncol(all_nodes_ts$Enodes)]
  const2 = dt/(2*tau)
  const2_1 = 1 - (dt/tau)
  g_N_t = all_nodes_ts$int_out$net_act1[node,]
  s = args_dict$s
  s_phi_x_t = s * phi(x_t)
  I_t = all_nodes_ts$int_out$spont_act1[node,]
  g_N_t_dt = all_nodes_ts$int_out$net_act2[node,]
  
  I_t_dt = all_nodes_ts$int_out$spont_act2[node,]
  
  if(excl_I){
    s_phi_ave = s * phi((const2_1*x_t)+((dt/tau)*(g_N_t+s_phi_x_t)))
    rhs = (const1 * x_t) + const2 * (const2_1 * (g_N_t + s_phi_x_t) + g_N_t_dt + s_phi_ave)
  } else{
    s_phi_ave = s * phi((const2_1*x_t)+((dt/tau)*(g_N_t+s_phi_x_t+I_t)))
    rhs = (const1 * x_t) + const2 * (const2_1 * (g_N_t + s_phi_x_t + I_t) + g_N_t_dt + s_phi_ave + I_t_dt) 
  }
  
  if(verbose){
    out = data.frame(const1, x_t, const2, const2_1, g_N_t, s_phi_x_t, I_t, g_N_t_dt, s_phi_ave, I_t_dt)
    return(out)
  } else {
    return(unique(round(x_t_dt - rhs, 10)))
  }
}