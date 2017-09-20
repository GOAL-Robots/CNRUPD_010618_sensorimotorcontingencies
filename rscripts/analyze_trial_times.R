rm(list = ls())

# listo fo required packages
toInstall <- c("extrafont",
               "ggplot2",
               "data.table",
               "cowplot",
               "grid",
               "gridExtra")

# verify and install uninstalled packages
for (pkg in toInstall)
    if (!require(pkg, character.only = TRUE))
        install.packages(pkg,
                         repos = "http://cran.us.r-project.org")

# load Verdana font
if (!("Verdana" %in% fonts()))
{
    font_import()
    loadfonts()
}

#--------------------------------------------------------------------

# - load data
trials <- fread("all_trials")
names(trials) <- c("LEARNING_TYPE", "INDEX", "TIMESTEPS", "GOAL")

trials = trials[]
TS = max(trials$TIMESTEPS)
N = length(trials$TIMESTEPS)
GOALS = unique(trials$GOAL)
N_GOALS = length(GOALS[GOALS>=0])

# - compute the duration of each trial in timesteps
attach(trials)
trials$TRIAL_DURATION = c(TIMESTEPS[1],
                          TIMESTEPS[2:N] - TIMESTEPS[1:(N - 1)])
detach(trials)
trials = trials[trials$TRIAL_DURATION<110]

predictions <- fread("all_predictions")
names(predictions) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", 
                        format(1:N_GOALS), "CURR_GOAL")

predictions = melt(predictions, 
                   id.vars = c("TIMESTEPS"), 
                   measure.vars = format(1:N_GOALS), 
                   variable.name="GOAL", 
                   value.name="prediction" )
predictions$GOAL = as.numeric(predictions$GOAL) - 1


setkey(trials, TIMESTEPS, GOAL)
setkey(predictions, TIMESTEPS, GOAL)

comp = trials[predictions, nomatch=0]

comp = comp[,.(TIMESTEPS, ord=1:length(TIMESTEPS), prediction, TRIAL_DURATION), by = .(GOAL)]

comp_mean = comp[,.(m = mean(TRIAL_DURATION), sd = sd(TRIAL_DURATION), 
                    mp = mean(prediction), sdp = sd(prediction)), by = .(ord) ]

gp = ggplot(subset(comp), 
            aes(y = TRIAL_DURATION, x = ord, group=GOAL, color=factor(GOAL)))
gp = gp + geom_line()
print(gp)

gp = ggplot(subset(comp_mean), 
            aes(y = m, x = ord))
gp = gp + geom_ribbon(aes(ymin = m - sd, ymax = m +sd ), fill="#888888")
gp = gp + geom_line()
print(gp)

gp = ggplot(subset(comp_mean), 
            aes(x = mp, y = m, color=ord))
gp = gp + geom_point(size=1.5)
gp = gp + scale_color_gradientn(colours = c("#000000","#ff0000","#ff4400", "#ffff00"))
print(gp)


# # - Create dataset with histories of trial
# #   durations for each goal
# dfs = list()
# for (g in 1:N_GOALS)
# {
#     # data for a single goal
#     g_trials = subset(trials, GOAL == g - 1)
#     # smoothing of the history of trial durations
#     f = filter(g_trials$TRIAL_DURATION, rep(1, 50) / 50)
#     f = f[!is.na(f)]
#     # create a data.table for the goal
#     df = data.table(
#         goal = rep(g, length(f)),
#         trial = 1:length(f),
#         TRIAL_DURATION = f)
#     dfs[[g]] = df
# }
# 
# 
# 
# # merge all goal data.tables into a single dataset
# df = rbindlist(dfs)
# 
# # - plot
# 
# pdf("trial_duration_per_goal.pdf", width=3.5, height=3)
# # plot trial duration histories for each trial
# gp = ggplot(df, aes(x = trial,
#                     y = TRIAL_DURATION,
#                     group = factor(goal),
#                     color = factor(goal) ) )
# gp = gp + geom_line(show.legend = FALSE)
# gp = gp + xlab("Trials")
# gp = gp + ylab("Trial duration\n(timesteps)")
# print(gp)
# dev.off()
# print(gp)
# 
# # - compute the mean and standard deviation of trial durations
# df_mean = df[,
#              .(TRIAL_DURATION = mean(TRIAL_DURATION),
#                trial_time_sd
#                = sd(TRIAL_DURATION)),
#              by = .(trial)]
# 
# # - plot the mean and standard deviation of trial durations
# pdf("trial_duration_per.pdf", width=3.5, height=3)
# gp = ggplot(df_mean, aes(x = trial,  y = TRIAL_DURATION))
# gp = gp + geom_ribbon(aes(
#     ymax = TRIAL_DURATION + trial_time_sd,
#     ymin = TRIAL_DURATION - trial_time_sd), fill = "#aaaaaa")
# gp = gp + geom_line(show.legend = FALSE, color = "#000000")
# gp = gp + xlab("Trials")
# gp = gp + ylab("Trial duration\n(timesteps)")
# print(gp)
# dev.off()
# print(gp)
# 
