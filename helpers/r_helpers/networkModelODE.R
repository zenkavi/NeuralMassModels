require(deSolve)

dx_dt = function(t, state, params){
  
}


times = seq(0, Tmax, by = dt)

parameters = c(s = ..., 
                g = ...,
                tau = ...,
                noise_sd = ...)

state = rep(0, n_nodes)

networkModelOde =  ode(y = state, times = times, func = dx_dt, parms = parameters, method="rk4")
