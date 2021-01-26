require(tidyverse)

extract_ts_matrix = function(all_nodes_ts){
  
  # Convert output of desolve ODE solver to matrix of timeseries to be used in other functions
  if ("time" %in% names(data.frame(all_nodes_ts))){
    out_mat = data.frame(all_nodes_ts) %>%
      select(-time)
    
    return(t(out_mat))
  } else {
    return(all_nodes_ts)
  }
  
}