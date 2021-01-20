require(deSolve)

phi = function(a){
  out = (exp(2*a)-1)/(exp(2*a)+1)
  return(out)
}

dx_dt = function(t, state, params){
  
  num_nodes = length(state)
  
  dxdt = rep(NA, num_nodes)
  
  s = params$s
  g = params$g
  tau = params$tau
  noise_scale = params$noise_scale
  W = params$W
  
  if("I" %in% names(params)){
    I = params$I
  } 
  
  for(i in 1:num_nodes){
    noise = rnorm(1, mean = 0, sd = noise_scale)
    if("I" %in% names(params)){
      spont_act = I[i, t] + noise
    } else{
      spont_act = noise
    }
    print(spont_act)
    # dxdt[i] = (-state[i] + s*phi(state[i]) + g * (W[i,] %*% state) + spont_act)/tau
    dxdt[i] = 0
  }
  
  return(list(dxdt))
}

# Usage
# times = seq(0, cur_args_dict$Tmax, by = cur_args_dict$dt)
# state0 = rep(0, n_nodes)
# net_dat_ode =  ode(y = state0, times = times, func = dx_dt, parms = cur_args_dict, method="rk4")
