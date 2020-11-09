phi = function(a){
  out = (exp(2*a)-1)/(exp(2*a)+1)
  return(out)
}

default_args_dict = list('bottomup'= FALSE, 
                 'dt'=.5,  
                 'ea'=200,
                 'g'=1, 
                 'hubnetwork_dsity'= .25,
                 'I'= NULL,
                 'innetwork_dsity'= .60,
                 'iv'= 400,
                 'local_com'= 1, 
                 'ncommunities'= 3,
                 'noise'= NULL,
                 'noise_loc'= 0, 
                 'noise_scale'= 0,
                 'nodespercommunity'= 35,
                 'outnetwork_dsity'=.08,
                 'plot_network'= FALSE,
                 'plot_task'= FALSE, 
                 's'=.8,
                 'sa'=100,
                 'showplot'=FALSE,
                 'standardize'=FALSE,
                 'stim_mag'=.5,
                 'stimsize'= 3, 
                 'taskdata'=NULL,
                 'tasktiming'=NULL,
                 'tau'=1, 
                 'Tmax'=1000,
                 'topdown'=TRUE,
                 'W'= NULL)

networkModel = function(W, args_dict, old=FALSE, debug = FALSE){
  
  Tmax=args_dict$Tmax
  dt=args_dict$dt
  g=args_dict$g
  s=args_dict$s
  tau=args_dict$tau
  I=args_dict$I
  noise=args_dict$noise
  noise_loc = args_dict$noise_loc
  noise_scale = args_dict$noise_scale
  
  TT = seq(1, Tmax, dt)
  totalnodes = dim(W)[1]
  
  # External input (or task-evoked input) && noise input
  if(is.null(I)){
    I = matrix(0, totalnodes, length(TT))
    } 
  
  # Noise parameter
  if (is.null(noise)){
    noise = matrix(0, totalnodes, length(TT))
  } else {
    noise = matrix(rnorm(totalnodes*length(TT), mean = noise_loc, sd = noise_scale), totalnodes, length(TT))
  }
  
  # Initial conditions and empty arrays
  Enodes = matrix(0, totalnodes, length(TT))
  # Initial conditions
  # Initial conditions to 0 if there is a task simulation and no noise
  if (!is.null(I)){
      Einit = matrix(0, totalnodes, 1)
  } else {
    Enit  = matrix(rnorm(totalnodes), totalnodes, 1)
  }
  
  # Assign initial values to first time point of all nodes
  Enodes[,1] = Einit
  
  # Initialize spont_act matrix for all nodes
  spont_act = matrix(0, totalnodes, 1)
  
  #Debugging
  if(debug){
    int_out = list(spont_act1 = matrix(NA, totalnodes, length(TT)-1),
                   net_act1 = matrix(NA, totalnodes, length(TT)-1),
                   k1e = matrix(NA, totalnodes, length(TT)-1),
                   ave =  matrix(NA, totalnodes, length(TT)-1),
                   spont_act2  = matrix(NA, totalnodes, length(TT)-1),
                   net_act2 = matrix(NA, totalnodes, length(TT)-1),
                   k2e = matrix(NA, totalnodes, length(TT)-1))
  }  
  
  for (t in 1:(length(TT)-1)){
    ## Solve using Runge-Kutta Order 2 Method
    ## End point form: https://lpsa.swarthmore.edu/NumInt/NumIntSecond.html##section17

    spont_act = noise[,t] + I[,t]
    net_act = g*(W %*% phi(Enodes[,t]))
    k1e = -Enodes[,t] + net_act + s*phi(Enodes[,t]) + spont_act
    k1e = k1e/tau
    
    # Debugging
    if(debug){
      int_out$spont_act1[,t] = spont_act
      int_out$net_act1[,t] = net_act
      int_out$k1e[,t] = k1e
    }
    
    ave = Enodes[,t] + k1e*dt
    spont_act = noise[,t+1] + I[,t+1]
    net_act = g*(W %*% phi(ave))
    k2e = -ave + net_act + s*phi(ave) + spont_act 
    k2e = k2e/tau
    
    # Debugging    
    if(debug){
      int_out$ave[,t] = ave
      int_out$net_act2[,t] = net_act
      int_out$spont_act2[,t] = spont_act
      int_out$k2e[,t] = k2e
    }
    
    Enodes[,t+1] = Enodes[,t] + (.5*(k1e+k2e))*dt
    
  }
  
  if(debug){
    return(list(Enodes = Enodes, int_out = int_out))
  } else{
    return(Enodes)
  }

}