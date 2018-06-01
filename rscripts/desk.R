##
## Copyright (c) 2018 Francesco Mannella. 
## 
## This file is part of sensorimotor-contingencies
## (see https://github.com/GOAL-Robots/CNRUPD_010618_sensorimotorcontingencies).
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
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


l=2
