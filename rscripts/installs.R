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

