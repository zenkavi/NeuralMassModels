generateSynapticNetwork = function(W, showplot=default_args['showplot'], weight_loc = 1.0, weight_scale = .2){
  # """
  #   Generate synaptic matrix over structural matrix with randomized gaussian weighs with
  #   mean = 1.0 and standard deviation of 0.2 (so all weights are positive)
  #   
  #   Parameters:
  #       W = structural connectivity matrix
  #       showplot = if set to True, will automatically display the structural matrix using matplotlib.pyplot
  # 
  #   Returns:
  #       Synaptic matrix with Gaussian weights on top of structural matrix
  #   """
  # Find non-zero connections
  
  G = matrix(0, nrow=dim(W)[1], ncol=dim(W)[2])
  totalnodes = dim(G)[1]
  connect_ind = (W != 0)
  nconnects = sum(connect_ind)
  weights = rnorm(nconnects, weight_loc, weight_scale)
  G[connect_ind] = weights
  
  # Find num connections per node
  nodeDeg = colSums(W)
  
  # Synaptic scaling according to number of incoming connections
  # np.fill_diagonal(G,0)
  G = G/nodeDeg

  if (showplot){
    data.frame(G) %>%
      mutate(to = row.names(.),
             from = names(.),
             from = gsub("X","",from)) %>%
      gather(key, weight, -to, -from) %>%
      mutate(from = sort(from)) %>%
      select(-key) %>%
      ggplot(aes(x=from, y=to, fill=weight))+
      geom_tile()
  }
  
  return(G) 
}
