require(data.table)
require(ggplot2)
require(cowplot)

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

TS = TS*(4/6)
pdf("g_means.pdf")
gp0 = ggplot(g_means, aes(x = TIMESTEPS, y = p_mean, group = GOAL, colour = GOAL))
gp0 = gp0 + geom_point(data=all_predictions, 
                     aes(x = TIMESTEPS, y = 1.05 + 0.4*(CURR_GOAL)/max(N_GOALS)), 
                     size=0.4,
                     inherit.aes=FALSE)
gp0 = gp0 + geom_line(size = 1, show.legend = F)
gp0 = gp0 + geom_line(aes(x = TIMESTEPS, y = th), inherit.aes = FALSE, show.legend = F )
gp0 = gp0 + scale_y_continuous(limits=c(0, 1.5), breaks= c(0,.5, 1))
gp0 = gp0 + scale_x_continuous(limits=c(0, TS))
gp0 = gp0 + xlab("Timesteps") 
gp0 = gp0 + ylab("Means of goal predictions                    ") 
gp0 = gp0 + theme_bw() 
if(TYPES>1) gp0 = gp0 + facet_grid(LEARNING_TYPE~.)
gp0 = gp0 + theme( 
                text=element_text(size=14, family="Verdana"), 
                panel.border=element_blank(),
                legend.title = element_blank(),
                legend.background = element_blank(),
                panel.grid.major = element_blank(),
                panel.grid.minor = element_blank()
                )
print(gp0)
dev.off()

gp1 = ggplot(g_means, aes(x = TIMESTEPS, y = p_mean))
gp1 = gp1 + geom_point(data=all_predictions, 
                     aes(x = TIMESTEPS, y = 1.2 + 0.4*(CURR_GOAL)/max(N_GOALS)), 
                     size=0.3, stroke = 0,
                     inherit.aes=FALSE)
gp1 = gp1 + geom_ribbon(data=means, aes(ymin = p_min, ymax = p_max), 
                      colour = NA, fill = "#dddddd", size=0.0)
gp1 = gp1 + geom_ribbon(data=means, aes(ymin = pmax(0, p_mean - p_sd), 
                                      ymax = pmin(1,p_mean + p_sd)),
                      colour = NA, fill = "#aaaaaa", size=0.0)
gp1 = gp1 + geom_line(data=means, size = .5, colour = "#000000")
gp1 = gp1 + geom_line(aes(x = TIMESTEPS, y = th), size=0.1, 
                    inherit.aes = FALSE, show.legend = F )
gp1 = gp1 + scale_y_continuous(limits=c(0, 1.5), breaks= c(0,.5, 1), 
                             labels=c("0.0","0.5","1.5"))
gp1 = gp1 + scale_x_continuous(limits=c(0, TS))
gp1 = gp1 + xlab("Timesteps") 
gp1 = gp1 + ylab("") 
gp1 = gp1 + theme_bw() 

if(TYPES>1) gp1 = gp1 + facet_grid(LEARNING_TYPE~.)

gp1 = gp1 + theme( 
    text=element_text(size=11, family="Verdana"), 
    panel.border=element_blank(),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)
png("means_all.png", width=700, height=300)
print(gp1)
dev.off()
pdf("means_all.pdf", width=7, height=3)
print(gp1)
dev.off()
svg("means_all.svg", width=7, height=3)
print(gp1)
dev.off()

