require(deSolve)

dx_dt = function(t, state, params){
  
  num_nodes = length(state)
  
  dxdt = rep(NA, num_nodes)
  
  s = params['s']
  g = params['g']
  tau = params['tau']
  sigma = params['sigma']
  
  for(i in 1:num_nodes){
    spont_act = rnorm(1, mean=0, sd= sigma)
    dxdt[i] = (-state[i] + s*phi(state[i]) + g * ... + spont_act)/tau 
  }
  
  return(list(dxdt))
}


times = seq(0, Tmax, by = dt)

parameters = c(s = ..., 
                g = ...,
                tau = ...,
                noise_sd = ...)

state = rep(0, n_nodes)

networkModelOde =  ode(y = state, times = times, func = dx_dt, parms = parameters, method="rk4")
