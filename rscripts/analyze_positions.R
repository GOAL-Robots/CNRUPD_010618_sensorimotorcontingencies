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
toInstall <- c("extrafont", "ggplot2", "data.table", "cowplot")
for(pkg in toInstall)
{
    if(!require(pkg, character.only=TRUE) )
    {
        install.packages(pkg, repos = "http://cran.us.r-project.org")
    }
}

require(data.table)
require(ggplot2)
require(cowplot)
library(extrafont)

if (!("Verdana" %in% fonts()) )
{
    font_import()
    loadfonts()
}


###############################################################################################################################

sem<-function(x) sd(x)/sqrt(length(x))

###############################################################################################################################

positions = fread('all_positions')
N_GOALS = 9

pos_labels = c("xl4","yl4","xl3","yl3","xl2","yl2","xl1","yl1",
                      "xr1","yr1","xr2","yr2","xr3","yr3", "xr4","yr4")

names(positions) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", pos_labels, "CURR_GOAL")


positions = melt(positions, 
             id.vars = c("LEARNING_TYPE", "INDEX", "TIMESTEPS", "CURR_GOAL"), 
             measure.vars = pos_labels, 
             variable.name="pos", 
             value.name="angle" )


xpositions = subset(positions, grepl("x", pos))
ypositions = subset(positions, grepl("y", pos))

positions = xpositions[,.(LEARNING_TYPE, INDEX,TIMESTEPS, CURR_GOAL)]

positions$xpos = xpositions$pos
positions$ypos = ypositions$pos

positions$x = xpositions$angle
positions$y = ypositions$angle

means = positions[,.(x = mean(x), y = mean(y), 
                         x_sd = sd(x), y_sd = sd(y), 
                         x_err = sem(x), y_err = sem(y),
                         x_min = min(x), y_min = min(y)), by=.(LEARNING_TYPE, xpos, ypos)]

