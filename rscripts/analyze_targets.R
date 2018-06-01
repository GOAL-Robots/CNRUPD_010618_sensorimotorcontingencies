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

N_JOINTS = 6
targets = fread('all_targets')
N_GOALS = (dim(targets)[2] - 4)/N_JOINTS 

goal_labels = floor((0:(N_GOALS*N_JOINTS-1))/N_JOINTS)
pos_labels = floor((0:(N_GOALS*N_JOINTS-1))%%N_JOINTS)
goal_pos_labels = paste("G", goal_labels, "_P", pos_labels, sep="")
names(targets) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", goal_pos_labels, "CURR_GOAL")
targets = melt(targets, 
             id.vars = c("LEARNING_TYPE", "TIMESTEPS", "INDEX"), 
             measure.vars = goal_pos_labels, 
             variable.name="GOAL_POS", 
             value.name="angle" )
targets$GOAL = gsub("_.*","",targets$GOAL_POS)
targets$POS = gsub(".*_","",targets$GOAL_POS)



