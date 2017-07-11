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


all_predictions <- fread("all_predictions")
N_GOALS = dim(all_predictions)[2] - 4 
names(all_predictions) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", paste("G", 1:N_GOALS, sep=""),"CURR_GOAL")


predictions = melt(all_predictions, 
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
TYPES=length(unique(sort(means$LEARNING_TYPE)))

pdf("means.pdf")
gp = ggplot(means, aes(x = TIMESTEPS, y = p_mean, group = LEARNING_TYPE))
gp = gp + geom_ribbon(aes(ymin = p_min, ymax = p_max), 
                      colour = "#666666", fill = "#dddddd")
gp = gp + geom_ribbon(aes(ymin = pmax(0, p_mean - p_sd), ymax = pmin(1,p_mean + p_sd)),
                      colour = "#666666", fill = "#bbbbbb")
gp = gp + geom_line(size = 1.5, colour = "#000000")
gp = gp + geom_line(aes(x = TIMESTEPS, y = th), inherit.aes = FALSE, show.legend = F )
gp = gp + xlab("Timesteps") 
gp = gp + ylab("Means of goal predictions") 
gp = gp + theme_bw() 
if(TYPES>1) gp = gp + facet_grid(LEARNING_TYPE~.)

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

g_means = predictions[, 
              .(p_mean = mean(prediction), 
                p_sd = sd(prediction), 
                p_err = sem(prediction),
                p_min = min(prediction),
                p_max = max(prediction) ), 
              by = .(LEARNING_TYPE,GOAL, 
                     TIMESTEPS, INDEX)]

g_means$TIMESTEPS=floor(g_means$TIMESTEPS/1000)*1000
g_means = g_means[, 
              .(p_mean = mean(p_mean), 
                p_sd = mean(p_sd), 
                p_err = mean(p_err),
                p_min = mean(p_min),
                p_max = mean(p_max) ), 
              by = .(LEARNING_TYPE,GOAL,
                     TIMESTEPS )]

g_means$th = 1
TS = max(g_means$TIMESTEPS)

pdf("g_means.pdf")
gp = ggplot(g_means, aes(x = TIMESTEPS, y = p_mean, group = GOAL, colour = GOAL))
gp = gp + geom_point(data=all_predictions, 
                     aes(x = TIMESTEPS, y = 1.05 + 0.4*(CURR_GOAL)/max(N_GOALS)), 
                     size=0.4,
                     inherit.aes=FALSE)
gp = gp + geom_line(size = 1, show.legend = F)
gp = gp + geom_line(aes(x = TIMESTEPS, y = th), inherit.aes = FALSE, show.legend = F )
gp = gp + scale_y_continuous(limits=c(0, 1.5), breaks= c(0,.5, 1))
gp = gp + scale_x_continuous(limits=c(0, TS))
gp = gp + xlab("Timesteps") 
gp = gp + ylab("Means of goal predictions                    ") 
gp = gp + theme_bw() 
if(TYPES>1) gp = gp + facet_grid(LEARNING_TYPE~.)
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

pdf("means_all.pdf", width=7, height=3)
gp = ggplot(g_means, aes(x = TIMESTEPS, y = p_mean))
gp = gp + geom_point(data=all_predictions, 
                     aes(x = TIMESTEPS, y = 1.2 + 0.4*(CURR_GOAL)/max(N_GOALS)), 
                     size=0.1, stroke = 0,
                     inherit.aes=FALSE)
gp = gp + geom_ribbon(data=means, aes(ymin = p_min, ymax = p_max), 
                      colour = NA, fill = "#dddddd", size=0.0)
gp = gp + geom_ribbon(data=means, aes(ymin = pmax(0, p_mean - p_sd), 
                                      ymax = pmin(1,p_mean + p_sd)),
                      colour = NA, fill = "#aaaaaa", size=0.0)
gp = gp + geom_line(data=means, size = .5, colour = "#000000")
gp = gp + geom_line(aes(x = TIMESTEPS, y = th), size=0.1, 
                    inherit.aes = FALSE, show.legend = F )
gp = gp + scale_y_continuous(limits=c(0, 1.5), breaks= c(0,.5, 1), 
                             labels=c("0.0","0.5","1.5"))
gp = gp + scale_x_continuous(limits=c(0, TS))
gp = gp + xlab("Timesteps") 
gp = gp + ylab("Means of goal predictions") 
gp = gp + theme_bw() 

if(TYPES>1) gp = gp + facet_grid(LEARNING_TYPE~.)

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
