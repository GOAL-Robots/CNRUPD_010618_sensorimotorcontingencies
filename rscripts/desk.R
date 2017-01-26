# dp = data_predictions
# dp$x_y = paste(as.character( dp$x),as.character(dp$y), sep='_')
# 
# #dp = subset(dp, x_y == "0_0")
# 
# #dp = subset(dp, idx> 1 & idx<3500)
# 
# p = ggplot(dp, aes(x=idx, y=prediction, group=x_y, color=x_y))
# p = p + geom_line()
# print(p)
# 
require(gtable)


grad_legend_grob = function(n_cols = 10)
{
    p = ggplot(data.frame(list(X=seq(n_cols),Y=seq(n_cols))),aes(x=X,y=Y,color=X)) + geom_blank()
    p = p + scale_colour_gradientn(colours=rainbow(length(df$X)))
    p = p + theme_bw()
    p = p + theme(
                  axis.ticks = element_blank(),
                  axis.title.x = element_blank(),
                  axis.title.y = element_blank(), 
                  axis.text.x = element_blank(),
                  axis.text.y = element_blank(),
                  panel.border = element_blank(),
                  legend.title=element_blank(), 
                  legend.key.width=unit(.1,'npc'), 
                  legend.key.height=unit(.1,'npc'),
                  legend.position=c(.5,.5)
                  )
    return(ggplotGrob(p))
}

g = grad_legend_grob(30)
