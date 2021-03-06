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

TS_SEC = 60000
first_g_means = subset(g_means, TIMESTEPS < TS_SEC)
first_predictions = subset(all_predictions, TIMESTEPS < TS_SEC)
first_means = subset(means, TIMESTEPS < TS_SEC)
TS_SEC = max(first_means$TIMESTEPS)
gp2 = ggplot(first_g_means, aes(x = TIMESTEPS, y = p_mean))
gp2 = gp2 + geom_point(data=first_predictions, 
                     aes(x = TIMESTEPS, y = 1.2 + 0.4*(CURR_GOAL)/max(N_GOALS)), 
                     size=0.5, stroke = 0,
                     inherit.aes=FALSE)
gp2 = gp2 + geom_ribbon(data=first_means, aes(ymin = p_min, ymax = p_max), 
                      colour = NA, fill = "#dddddd", size=0.0)
gp2 = gp2 + geom_ribbon(data=first_means, aes(ymin = pmax(0, p_mean - p_sd), 
                                      ymax = pmin(1,p_mean + p_sd)),
                      colour = NA, fill = "#aaaaaa", size=0.0)
gp2 = gp2 + geom_line(data=first_means, size = .5, colour = "#000000")
gp2 = gp2 + geom_line(aes(x = TIMESTEPS, y = th), size=0.1, 
                    inherit.aes = FALSE, show.legend = F )
gp2 = gp2 + scale_y_continuous(limits=c(0, 1.5), breaks= c(0,.5, 1), 
                             labels=c("0.0","0.5","1.5"))
gp2 = gp2 + scale_x_continuous(limits=c(0, TS_SEC))
gp2 = gp2 + xlab("Timesteps") 
gp2 = gp2 + ylab("") 
#gp2 = gp2 + ylab("Means of goal predictions") 
gp2 = gp2 + theme_bw() 

if(TYPES>1) gp2 = gp2 + facet_grid(LEARNING_TYPE~.)

gp2 = gp2 + theme( 
    text=element_text(size=11, family="Verdana"), 
    panel.border=element_blank(),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)
pdf("means_firsts.pdf", width=7, height=3)
print(gp2)
dev.off()
png("means_firsts.png", width=700, height=300)
print(gp2)
dev.off()
svg("means_firsts.svg", width=7, height=3)
print(gp2)
dev.off()



TS_SEC2 = 0.3e6
START = 1.5e6
first_g_means = subset(g_means, TIMESTEPS > START  & TIMESTEPS < START + TS_SEC2  )
first_predictions = subset(all_predictions, TIMESTEPS > START  & TIMESTEPS < START + TS_SEC2)
first_means = subset(means,  TIMESTEPS > START  & TIMESTEPS < START + TS_SEC2)
gp3 = ggplot(first_g_means, aes(x = TIMESTEPS, y = p_mean))
gp3 = gp3 + geom_point(data=first_predictions, 
                       aes(x = TIMESTEPS, y = 1.2 + 0.4*(CURR_GOAL)/max(N_GOALS)), 
                       size=.5, stroke = 0,
                       inherit.aes=FALSE)
gp3 = gp3 + geom_ribbon(data=first_means, aes(ymin = p_min, ymax = p_max), 
                        colour = NA, fill = "#dddddd", size=0.0)
gp3 = gp3 + geom_ribbon(data=first_means, aes(ymin = pmax(0, p_mean - p_sd), 
                                              ymax = pmin(1,p_mean + p_sd)),
                        colour = NA, fill = "#aaaaaa", size=0.0)
gp3 = gp3 + geom_line(data=first_means, size = .5, colour = "#000000")
gp3 = gp3 + geom_line(aes(x = TIMESTEPS, y = th), size=0.1, 
                      inherit.aes = FALSE, show.legend = F )
gp3 = gp3 + scale_y_continuous(limits=c(0, 1.5), breaks= c(0,.5, 1), 
                               labels=c("0.0","0.5","1.5"))
gp3 = gp3 + scale_x_continuous(limits=c(START, START+TS_SEC2))
gp3 = gp3 + xlab("Timesteps") 
gp3 = gp3 + ylab("") 
#gp3 = gp3 + ylab("Means of goal predictions") 
gp3 = gp3 + theme_bw() 

if(TYPES>1) gp3 = gp3 + facet_grid(LEARNING_TYPE~.)

gp3 = gp3 + theme( 
    text=element_text(size=11, family="Verdana"), 
    panel.border=element_blank(),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)
pdf("means_firsts1.pdf", width=7, height=3)
print(gp3)
dev.off()
png("means_firsts1.png", width=700, height=300)
print(gp3)
dev.off()
svg("means_firsts1.svg", width=7, height=3)
print(gp3)
dev.off()

pp = ggdraw() 
pp = pp + draw_plot(gp1, 0, 0.5, 1, 0.5)
pp = pp + draw_plot(gp2, 0, 0, 0.48, 0.5)
pp = pp + draw_plot(gp3, 0.5, 0, 0.48, 0.5)
print(pp)
pdf("means_comp.pdf", width=7, height=4)
print(pp)
dev.off()
png("means_comp.png", width=700, height=400)
print(pp)
dev.off()
svg("means_comp.svg", width=7, height=4)
print(pp)
dev.off()
