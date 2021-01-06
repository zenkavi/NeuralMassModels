plot_adj_mat = function(W, border_to = NA, border_from = NA){
  dat = data.frame(W) %>%
    mutate(to = row.names(.),
           from = names(.),
           from = gsub("X","",from),
           to = as.numeric(to),
           from = as.numeric(to)) %>%
    gather(key, weight, -to, -from) %>%
    mutate(from = sort(from)) %>%
    select(-key) %>%
    mutate(border = ifelse(to == border_to & from == border_from, T, NA))
  
  p = dat %>%
    ggplot(aes(x=from, y=to, fill=weight))+
    geom_tile(size = 2, aes(color=border)) +
    scale_color_manual(guide=F, values = c('TRUE' = "red"))
  return(list(p=p, dat = dat))
}
