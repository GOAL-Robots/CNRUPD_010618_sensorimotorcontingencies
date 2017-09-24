# INTRO ------------------------------------------------------------------------

rm(list = ls())

# __ list of required packages ====
toInstall <- c("extrafont",
               "ggplot2",
               "data.table",
               "cowplot",
               "grid",
               "gridExtra")

# __ verify and install uninstalled packages ====
for (pkg in toInstall) {
    if (!require(pkg, character.only = TRUE)) {
        install.packages(pkg,
                         repos = "http://cran.us.r-project.org")
    }
}

# __ load Verdana font ====
if (!("Verdana" %in% fonts())) {
    font_import()
    loadfonts()
}

offline.plot <- TRUE

# UTILS ------------------------------------------------------------------------

# standard error
std.error <- function(x) {
    sd(x) / sqrt(length(x))
}

# finds the decimal scale of a time series
# returns one of 1e+01 1e+02 1e+03 1e+04
#                1e+05 1e+06 1e+07 1e+08
#                1e+09 1e+10
find.decimal.scale <- function(x)
{
    scales <- 10^seq(1, 10)
    x.scaled <- x/scales
    test.scales <- x.scaled < 10 & x.scaled > 1

    scales[which(test.scales == TRUE)]
}

# UTILS ------------------------------------------------------------------------

TS_ALL = 1000e+3

TS_LAST =  410e+3
TS_GAP =    20e+3




all_predictions <- fread("all_predictions")
N_GOALS = dim(all_predictions)[2] - 4
names(all_predictions) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", paste("G", 1:N_GOALS, sep=""),"CURR_GOAL")

all_weights <- fread("all_weights")
names(all_weights) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", "KOHONEN", "ECHO")
TS = max(all_weights$TIMESTEPS)
TS_ALL=TS
scale = find.decimal.scale(TS_GAP)
trials = 1:length(all_weights$TIMESTEPS)
tlbrk = trials[trials%%(scale/10) == 0]
tsbrk = all_weights$TIMESTEPS[tlbrk]
TSS = 1:TS
tslbrk = c(0,TSS[TSS%%(scale)==0])

scale_dense=find.decimal.scale(TS)
tslbrk_dense = TSS[TSS%%(scale_dense)==0]

tlbrk_medium = trials[trials%%500 == 0]
tsbrk_medium = all_weights$TIMESTEPS[tlbrk_medium]
tlbrk_dense = trials[trials%%100 == 0]
tsbrk_dense = all_weights$TIMESTEPS[tlbrk_dense]

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

gp = ggplot(means, aes(x = TIMESTEPS, y = p_mean, group = LEARNING_TYPE))
gp = gp + geom_ribbon(aes(ymin = p_min, ymax = p_max),
                      colour = "#666666", fill = "#dddddd")
gp = gp + geom_ribbon(aes(ymin = pmax(0, p_mean - p_sd), ymax = pmin(1,p_mean + p_sd)),
                      colour = "#666666", fill = "#bbbbbb")
gp = gp + geom_line(size = 1.5, colour = "#000000")
gzp = gp + geom_line(aes(x = TIMESTEPS, y = th), inherit.aes = FALSE, show.legend = F )
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

pdf("means.pdf")
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

TS = max(means$TIMESTEPS)
gp0 = ggplot(g_means, aes(x = TIMESTEPS, y = p_mean, group = GOAL, colour = GOAL))
gp0 = gp0 + geom_point(data=all_predictions,
                     aes(x = TIMESTEPS, y = 1.05 + 0.4*(CURR_GOAL)/max(N_GOALS)),
                     size=0.4,
                     inherit.aes=FALSE)
gp0 = gp0 + geom_line(size = 1, show.legend = F)
gp0 = gp0 + geom_line(aes(x = TIMESTEPS, y = th), inherit.aes = FALSE, show.legend = F )
gp0 = gp0 + scale_y_continuous(limits=c(0, 1.5), breaks= c(0,.5, 1))
gp0 = gp0 + scale_x_continuous(limits=c(0, TS),
                               breaks=tsbrk,
                               labels=tlbrk )
gp0 = gp0 + xlab("Trials")
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
pdf("g_means.pdf")
print(gp0)
dev.off()

TS = TS_ALL
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
                             labels=c("0.0","0.5","1.0"))
gp1 = gp1 + scale_x_continuous(limits=c(0, TS),
                               breaks=tsbrk,
                               labels=tlbrk,
                               sec.axis = sec_axis(~.,
                                                   name = "Timesteps",
                                                   breaks = tslbrk_dense,
                                                   labels = tslbrk_dense))
gp1 = gp1 + xlab("Trials")
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

pdf("means_all.pdf", width=7, height=3)
print(gp1)
dev.off()


TS_SEC = TS_GAP
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
                             labels=c("0.0","0.5","1.0"))
gp2 = gp2 + scale_x_continuous(limits=c(0, TS_SEC),
                               breaks=tsbrk_dense,
                               labels=tlbrk_dense,
                               sec.axis = sec_axis(~.,
                                                   name = "Timesteps",
                                                   breaks = tslbrk,
                                                   labels = tslbrk))
gp2 = gp2 + xlab("Trials")
gp2 = gp2 + ylab("")
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


TS_SEC2 = TS_GAP
START = TS_LAST -TS_GAP
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
                               labels=c("0.0","0.5","1.0"))

gp3 = gp3 + scale_x_continuous(limits=c(START, START+TS_SEC2),
                               breaks=tsbrk_dense,
                               labels=tlbrk_dense,
                               sec.axis = sec_axis(~.,
                                                   name = "Timesteps",
                                                   breaks = tslbrk,
                                                   labels = tslbrk))
gp3 = gp3 + xlab("Trials")
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

pp = ggdraw()
pp = pp + draw_plot(gp1, 0, 0.5, 1, 0.5)
pp = pp + draw_plot(gp2, 0, 0, 0.48, 0.5)
pp = pp + draw_plot(gp3, 0.5, 0, 0.48, 0.5)
pdf("prediction_history.pdf", width=7, height=4)
print(pp)
dev.off()

