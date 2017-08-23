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



