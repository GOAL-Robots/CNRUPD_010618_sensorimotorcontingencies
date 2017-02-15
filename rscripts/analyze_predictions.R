require(data.table)
require(ggplot2)

toInstall <- c("extrafont")
for(pkg in toInstall)
    if(!require(pkg, character.only=TRUE) )
    {
        install.packages(pkg, repos = "http://cran.us.r-project.org")
        library(extrafont)
        font_import()
    }

library(extrafont)

###############################################################################################################################

sem<-function(x) sd(x)/sqrt(length(x))

###############################################################################################################################


predictions <- fread("all_predictions")
N_GOALS = dim(predictions)[2] - 4 
names(predictions) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", paste("G", 1:N_GOALS, sep=""),"CURR_GOAL")


predictions = melt(predictions, 
             id.vars = c("LEARNING_TYPE", "TIMESTEPS", "INDEX"), 
             measure.vars = paste("G", 1:N_GOALS, sep = ""), 
             variable.name="GOAL", 
             value.name="prediction" )

means = predictions[, 
              .(p_mean = mean(prediction), 
                p_sd = sd(prediction), 
                p_err = sem(prediction),
                p_min = min(prediction),
                p_max = max(prediction) ), 
              by = .(LEARNING_TYPE, 
                     TIMESTEPS, INDEX)]

means$TIMESTEPS=floor(means$TIMESTEPS/1000)*1000
means = means[, 
              .(p_mean = mean(p_mean), 
                p_sd = mean(p_sd), 
                p_err = mean(p_err),
                p_min = mean(p_min),
                p_max = mean(p_max) ), 
              by = .(LEARNING_TYPE, 
                     TIMESTEPS )]

means$th = 1

pdf("means.pdf")
gp = ggplot(means, aes(x = TIMESTEPS, y = p_mean, group = LEARNING_TYPE))
gp = gp + geom_ribbon(aes(ymin = p_min, ymax = p_max), colour = "#666666", fill = "#dddddd")
gp = gp + geom_line(size = 1.5, colour = "#000000")
gp = gp + geom_line(aes(x = TIMESTEPS, y = th), inherit.aes = FALSE, show.legend = F )
gp = gp + theme_bw() 
gp = gp + facet_grid(LEARNING_TYPE~.)

gp = gp + theme( 
                text=element_text(size=14, family="Verdana"), 
                panel.border=element_blank(),
                legend.title = element_blank(),
                legend.background = element_blank(),
                panel.grid.major = element_blank(),
                panel.grid.minor = element_blank()
                )
print(gp)
dev.off()
