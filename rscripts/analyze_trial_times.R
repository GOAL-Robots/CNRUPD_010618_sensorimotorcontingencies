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
N_GOALS = 25
all_trials <- fread("all_trials")
names(all_trials) <- c("LEARNING_TYPE", "INDEX", "TIMESTEPS", "GOAL")
TS = max(all_trials$TIMESTEPS)
N = length(all_trials$TIMESTEPS)

# - compute the duration of each trial in timesteps
attach(all_trials)
all_trials$TRIAL_DURATION = c(TIMESTEPS[1], 
                          TIMESTEPS[2:N] - TIMESTEPS[1:(N - 1)])
detach(all_trials)

# - Create dataset with histories of trial 
#   durations for each goal
dfs = list()
for (g in 1:N_GOALS)
{
    # data for a single goal
    g_trials = subset(all_trials, GOAL == g - 1)
    # smoothing of the history of trial durations
    f = filter(
        g_trials$TRIAL_DURATION, rep(1, 50) / 50)
    f = f[!is.na(f)]
    # create a data.table for the goal
    df = data.table(
        goal = rep(g, length(f)), 
        trial = 1:length(f),
        TRIAL_DURATION = f)
    dfs[[g]] = df
}

# merge all goal data.tables into a single dataset
df = rbindlist(dfs)

# - plot

pdf("trial_duration_per_goal.pdf", width=3.5, height=3)
# plot trial duration histories for each trial
gp = ggplot(df, aes(x = trial, 
                    y = TRIAL_DURATION, 
                    group = factor(goal), 
                    color = factor(goal) ) )
gp = gp + geom_line(show.legend = FALSE)
gp = gp + xlab("Trials")
gp = gp + ylab("Trial duration\n(timesteps)")
print(gp)
dev.off()
print(gp)

# - compute the mean and standard deviation of trial durations
df_mean = df[,
             .(TRIAL_DURATION = mean(TRIAL_DURATION),
               trial_time_sd
               = sd(TRIAL_DURATION)),
             by = .(trial)]

# - plot the mean and standard deviation of trial durations
pdf("trial_duration_per.pdf", width=3.5, height=3)
gp = ggplot(df_mean, aes(x = trial,  y = TRIAL_DURATION))
gp = gp + geom_ribbon(aes(
    ymax = TRIAL_DURATION + trial_time_sd,
    ymin = TRIAL_DURATION - trial_time_sd), fill = "#aaaaaa")
gp = gp + geom_line(show.legend = FALSE, color = "#000000")
gp = gp + xlab("Trials")
gp = gp + ylab("Trial duration\n(timesteps)")
print(gp)
dev.off()
print(gp)

