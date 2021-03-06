helpers_path = '~/Dropbox/RangelLab/NeuralMassModels/helpers/r_helpers/'
source(paste0(helpers_path,'extract_ts_matrix.R'))

run_ucr_glm_node = function(node, all_nodes_ts, task_reg=NULL){
  
  all_nodes_ts = extract_ts_matrix(all_nodes_ts)
  
  x_t = all_nodes_ts[node, -ncol(all_nodes_ts)]
  
  I_t = task_reg[-length(task_reg)]
  
  mod = lm(x_t ~ -1+I_t)
  
  out = list(mod_df = mod$model, coef = coef(mod))
  return(out)
}

run_ucr_glm = function(all_nodes_ts, task_reg=NULL){
  
  all_nodes_ts = extract_ts_matrix(all_nodes_ts)
  
  num_nodes = dim(all_nodes_ts)[1]
  
  out = list(ucr_task_betas = rep(NA, num_nodes),
             ucr_mods = list())
  
  for(node in 1:num_nodes){
    node_out = run_ucr_glm_node(node, all_nodes_ts, task_reg=task_reg)
    out$ucr_mods[[node]] = node_out$mod_df
    out$ucr_task_betas[node] = node_out$coef
  }
  
  return(out)
}