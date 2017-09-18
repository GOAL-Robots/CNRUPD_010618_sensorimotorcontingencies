rm(list=ls())


toInstall <-
    c("extrafont",
      "ggplot2",
      "data.table",
      "cowplot",
      "grid",
      "gridExtra")
for (pkg in toInstall)
{
    if (!require(pkg, character.only = TRUE))
    {
        install.packages(pkg, repos = "http://cran.us.r-project.org")
    }
}

require(data.table)
require(ggplot2)
require(cowplot)
library(extrafont)

if (!("Verdana" %in% fonts()))
{
    font_import()
    loadfonts()
}
###############################################################################################################################

sem <- function(x)
    sd(x) / sqrt(length(x))

###############################################################################################################################

LASTIMESTEPS = 000
AMP_TH = 0.1

#---------------------------------------------------------------------------------------
predictions <- fread("all_predictions")
N_GOALS = dim(predictions)[2] - 4
names(predictions) <-
    c("LEARNING_TYPE",
      "INDEX",
      "TIMESTEPS",
      paste("G", 1:N_GOALS, sep = ""),
      "CURR_GOAL")

goal_labels = paste("G", 1:N_GOALS, sep = "")
gpredictions = melt(
    predictions,
    id.vars = c("LEARNING_TYPE",  "INDEX", "TIMESTEPS", "CURR_GOAL"),
    measure.vars = goal_labels,
    variable.name = "GOAL",
    value.name = "prediction"
)
gpredictions$CGOAL = paste("G", gpredictions$CURR_GOAL + 1, sep = "")
gpredictions$CCGOAL = gpredictions$CGOAL == gpredictions$GOAL
gpredictions = subset(gpredictions, CCGOAL == TRUE)
gpredictions = gpredictions[order(LEARNING_TYPE, INDEX, TIMESTEPS), ]

#---------------------------------------------------------------------------------------
sensors = fread('all_sensors')
N_SENSORS = (dim(sensors)[2] - 4)
sens_labels = paste("S", 1:N_SENSORS, sep = "")
names(sensors) <-
    c("LEARNING_TYPE",
      "INDEX",
      "TIMESTEPS",
      sens_labels,
      "CURR_GOAL")
sensors = sensors[with(sensors, order(LEARNING_TYPE, INDEX, TIMESTEPS)), ]
sensors$prediction =  gpredictions$prediction
sensors = melt(
    sensors,
    id.vars = c(
        "LEARNING_TYPE",
        "INDEX",
        "TIMESTEPS",
        "CURR_GOAL",
        "prediction"
    ),
    measure.vars = sens_labels,
    variable.name = "sensor",
    value.name = "amp"
)
TS = max(sensors$TIMESTEPS)
n_tics = 15
TS_TICS = as.integer(floor(seq(1, as.integer(TS), length.out = n_tics)))



cont_sensors = fread("log_cont_sensors")
n_sensors = dim(cont_sensors)[2] - 2
names(cont_sensors) = c('ts', paste("S", 1:n_sensors, sep = ""), "goal")
melted_cont_sensors = melt(
    cont_sensors,
    id.vars = c("ts", "goal"),
    variable.name = "sensor",
    value.name = "touch"
)
melted_cont_sensors = melted_cont_sensors[, .(touch = sum(touch > 0.1)), by = .(sensor)]


last_of <- function(x, a)
{
    idx = last(which( x == a))

    res = rep(0, length(x))

    if(length(idx)>0)
    {
        after = c(rep(0, idx - 1), rep(1, length(x) - (idx - 1)))
        res = c(0, diff(after)) * (x > 0)
    }

    res
}


sensors = sensors[,
         .(amp = max(amp)),
         by = .(CURR_GOAL, prediction, TIMESTEPS)]


# #sensors = subset(sensors, (last_of(prediction,1)==1) )
# #---------------------------------------------------------------------------------------
# gps = list()
# for (tic in 2:n_tics)
# {
#     curr_sensors = subset(sensors, TIMESTEPS >= TS_TICS[tic - 1]
#                           & TIMESTEPS < TS_TICS[tic] )
#
#     amps = curr_sensors[,
#                         .(camp = sum(amp > 0.0)),
#                         by = .(CURR_GOAL, sensor)]
#     amps = amps[,
#                 .(camp = max(camp),
#                   sensor = sensor[camp == max(camp)]),
#                 by = .(CURR_GOAL)]
#
#     amps = amps[order(CURR_GOAL)]
#
#     gp = ggplot(amps, aes(x = as.numeric(sensor), y = camp))
#     gp = gp + geom_bar(stat = "identity")
#     gp = gp + scale_y_continuous(limits = c(0, 250), breaks = c(1, 200))
#     gp = gp + scale_x_continuous(limits = c(0, 30), breaks = c(1, 10, 20, 30))
#     gp = gp + geom_line(
#         data = melted_cont_sensors,
#         aes(x = as.numeric(sensor), y = touch / 100),
#         inherit.aes = FALSE,
#         color = "#000000",
#         size = 1
#     )
#     gp = gp + geom_line(
#         data = melted_cont_sensors,
#         aes(x = as.numeric(sensor), y = touch / 100),
#         inherit.aes = FALSE,
#         color = "#ffffff",
#         size = .7
#     )
#     gp = gp + xlab("Sensors")
#     gp = gp + ylab("Number of touches")
#     gp = gp + theme(
#         text = element_text(size = 11, family = "Verdana"),
#         # axis.ticks = element_blank(),
#         #axis.title.x = element_blank(),
#         axis.title.x = if (tic != n_tics)
#             element_blank(),
#         axis.text.x = if (tic != n_tics)
#             element_blank(),
#         axis.title.y = if (tic != n_tics / 2)
#             element_blank()
#         else
#             element_text(angle = 90),
#         #axis.text.y = element_blank(),
#         legend.title = element_blank(),
#         legend.background = element_blank(),
#         panel.grid.major = element_blank(),
#         plot.margin = unit(c(.0, .0, .0, .0), "in"),
#         panel.grid.minor = element_blank()
#     )
#
#     gps[[tic - 1]] = ggplotGrob(gp)
# }
#
# #pdf("goal_acquisition_history.pdf", width=3, height=6)
# grid.newpage()
# grid.draw(do.call(rbind, gps))
# #dev.off()
