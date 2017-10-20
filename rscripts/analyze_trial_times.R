# INTRO ------------------------------------------------------------------------
#
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

plot.offline = FALSE
if (file.exists("OFFLINE")) { plot.offline = TRUE }


# UTILS ------------------------------------------------------------------------


# If the timeseries ends converging
# to a maximum thids function returns the
# index of the plateau start and the value
# at the plateau
find_plateau <- function(time.series) {

    time.series.max <- max(time.series)
    time.series.length <- length(time.series)
    time.series.maxes <- 1*(time.series == time.series.max)
    time.series.maxes.diffs <- c(0, diff(time.series.maxes))
    res <- list()
    for(idx in which(time.series.maxes.diffs == 1)) {
        time.series.maxes.diffs.after <- time.series.maxes.diffs[idx:time.series.length]
        test <- any(time.series.maxes.diffs.after < 0)
        if(test == FALSE) {
            res$idx <- idx
            res$val <- time.series.max
            break
        }
    }

    res # returns (idx, val)
}


# CONSTS -----------------------------------------------------------------------

timesteps.gap <- 50e+3
simulation.index <- 1

# TRIALS -----------------------------------------------------------------------

# __ load trials dataset ====
trials <- fread("all_trials")
names(trials) <- c("learning.type",
                   "index",
                   "timesteps",
                   "goal")
trials <- subset(trials, index == simulation.index)

# __ read variables ====
timesteps.max <- max(trials$timesteps)
timesteps.number <- length(trials$timesteps)
goal.labels <- unique(trials$goal)
goal.number <- length(goal.labels[goal.labels >= 0])

# __ compute duration ====
# - compute the duration of each trial in timesteps and
#   add it to the data.table
attach(trials)
trials$trial.duration <- c(timesteps[1], diff(timesteps))
detach(trials)

# __ throw off trials where no match was encounterd ====
trials <- trials[goal != -1]
# __ throw off trials where trial_duration is zero ====
trials <- trials[trial.duration > 0]

# PREDICTIONS -----------------------------------------------------------------

# __ load prediction dataset ====
predictions <- fread("all_predictions")
names(predictions) <- c("learning.type",
                        "index",
                        "timesteps",
                        format(1:goal.number),
                        "current.goal")

predictions <- subset(predictions, index == simulation.index)

# __ melt by goal columns making a goal factor ====
predictions <- melt(
    predictions,
    id.vars = c("timesteps"),
    measure.vars = format(1:goal.number),
    variable.name = "goal",
    value.name = "prediction" )

# we need goal labels to be 0..(goal.max - 1)
predictions$goal <-  as.numeric(predictions$goal) - 1



# find the correct final timestep
plateau.indices <- c()
plateau.timesteps <- c()
for(goal.el in unique(sort(predictions$goal))) {
    predictions.goal <- predictions[goal == goal.el]
    plateau.index <- find_plateau(predictions.goal$prediction)$idx
    plateau.indices <- c(plateau.indices,  plateau.index)
    plateau.timesteps <- c(plateau.timesteps,
                           predictions.goal$timesteps[plateau.index])
}
plateau.all.index = max(plateau.indices)


# INNER JOIN ------------------------------------------------------------------

# __ set the keys ====
setkey(trials, timesteps, goal)
setkey(predictions, timesteps, goal)

# __ inner join ====
prediction.trials <- trials[predictions, nomatch = 0]

# __ add trial.seq.num ====
# add trial.seq.num defining the sequential order of trials for each goal
prediction.trials <-
    prediction.trials[, .(timesteps,
                          prediction,
                          trial.duration,
                          trial.seq.num = (0:(length(timesteps)-1))),
                      by = .(goal)]

prediction.trials <-
    subset(prediction.trials,
           timesteps >=0  & timesteps <=
               plateau.timesteps[plateau.indices ==
                                     plateau.all.index] +timesteps.gap)


# add trial.seq.num defining the sequential order of trials for each goal
prediction.trials$trial.seq.num =
    floor(prediction.trials$trial.seq.num * 1000) / 1000



# __ means of trial durations ====
# means of trial durations between goals for each trial.seq.num
prediction.trials.mean <-
    prediction.trials[, .(trials.mean = mean(trial.duration),
                          trials.dev = sd(trial.duration),
                          preds.mean = mean(prediction),
                          preds.dev = sd(prediction)),
                      by = .(trial.seq.num)]

# __ means of trial durations ====
# means of trial durations between goals for each trial.seq.num
prediction.trials.mean.all <-
    prediction.trials[, .(trials.mean = mean(trial.duration),
                          trials.dev = sd(trial.duration),
                          preds.mean = mean(prediction),
                          preds.dev = sd(prediction)),
                      by = .(trial.seq.num)]

# __ smooted trial durations ====
window = 10
prediction.trials.smoothed <- trials[predictions, nomatch = 0]

prediction.trials.smoothed <-
    prediction.trials[, .(timesteps,
                          prediction,
                          trial.duration = filter(trial.duration,
                                                  rep(1, window)/window)),
                      by = .(goal)]


prediction.trials.smoothed$trial.seq.num =
    floor(prediction.trials.smoothed$trial.seq.num * 1000) / 1000


# __ smoothed trial duration means ====
prediction.trials.smoothed.mean <-
    prediction.trials.mean[, .(trials.mean = filter(trials.mean,
                                                    rep(1, window)/window),
                               trials.dev = filter(trials.dev,
                                                   rep(1, window)/window),
                               preds.mean = filter(preds.mean,
                                                   rep(1, window)/window),
                               preds.dev = filter(preds.dev,
                                                  rep(1, window)/window),
                          trial.seq.num)]


# PLOTS -----------------------------------------------------------------------

# __ plot of the sequence of trial durations for each goal ====
gp <- ggplot(prediction.trials.smoothed,
            aes(y = trial.duration,
                x = trial.seq.num,
                group = goal,
                color = factor(goal)))
gp <- gp + geom_line(alpha = 0.6)
gp <- gp + xlab("Trial sequence number")
gp <- gp + ylab("Trial duration")
gp <- gp + guides(colour = guide_legend(override.aes = list(size = 4),
                                       title = "Goal"))

if(plot.offline == TRUE) {
    pdf("trial_duration_sequence_per_goal.pdf", width = 5, height = 3)
    print(gp)
    dev.off()
} else {
    print(gp)
}

# __ plot of the sequence of trial durations for each goal - smoothed ====
gp <- ggplot(prediction.trials.smoothed,
             aes(y = trial.duration,
                 x = trial.seq.num,
                 group = goal,
                 color = factor(goal)))
gp <- gp + geom_line(alpha = 0.6)
gp <- gp + xlab("Trial sequence number")
gp <- gp + ylab("Trial duration")
gp <- gp + guides(colour = guide_legend(override.aes = list(size = 4),
                                        title = "Goal"))

if(plot.offline == TRUE) {
    pdf("trial_duration_sequence_per_goal_smoothed.pdf",
        width = 5, height = 3)
    print(gp)
    dev.off()
} else {
    print(gp)
}

# __ plot of trial mean  ====
gp <- ggplot(prediction.trials.mean,
             aes(y = trials.mean,
                 x = trial.seq.num))
gp <- gp + geom_ribbon(aes(ymin = trials.mean - trials.dev,
                           ymax = trials.mean + trials.dev),
                       fill = "#888888")
gp <- gp + geom_line()
gp <- gp + xlab("Trial sequence number")
gp <- gp + ylab("Mean trial duration")

if(plot.offline == TRUE) {
    pdf("trial_duration_mean.pdf", width = 5, height = 3)
    print(gp)
    dev.off()
} else {
    print(gp)
}

# __ plot of trial mean - smoothed ====
gp <- ggplot(prediction.trials.smoothed.mean,
            aes(y = trials.mean,
                x = trial.seq.num))
gp <- gp + geom_ribbon(aes(ymin = trials.mean - trials.dev,
                          ymax = trials.mean + trials.dev),
                      fill = "#888888")
gp <- gp + geom_line()
gp <- gp + xlab("Trial sequence number")
gp <- gp + ylab("Mean trial duration")

if(plot.offline == TRUE) {
    pdf("trial_duration_mean_smoothed.pdf", width = 5, height = 3)
    print(gp)
    dev.off()
} else {
    print(gp)
}


# __ plot trial duration over predictions  ====
gp <- ggplot(subset(prediction.trials,
                    trial.seq.num >= 0 &
                        trial.seq.num <= 1000 ),
             aes(x = prediction,
                 y = trial.duration,
                 color = trial.seq.num))

gp <- gp + geom_density_2d(aes(alpha=..level.., color=..level..),
                           geom="polygon",
                           h = c(0.5, 100),
                           show.legend = FALSE)

gp <- gp + geom_point(size = .1, alpha = .5)

gp <- gp + scale_color_gradientn(colours = c("#ff0000",
                                             "#00ff00",
                                             "#0000ff"))
gp <- gp + xlab("Mean of Predictions")
gp <- gp + ylab("Mean of trial durations")
gp <- gp + guides(colour = guide_legend( override.aes = list(size = 4),
                                         title = "Trial\nsequence\nnumber"))


if(plot.offline == TRUE) {
    pdf("trial_duration_vs_prediction.pdf", width = 5, height = 3)
    print(gp)
    dev.off()
} else {
    print(gp)
}

cat(prediction.trials.mean$trial.seq.num)
cat("\n")
# __ plot trial means over prediction means ====
gp <- ggplot(subset(prediction.trials.mean),
            aes(x = preds.mean,
                y = trials.mean,
                color = trial.seq.num))
gp <- gp + geom_point(size = 1.3)
gp <- gp + scale_color_gradientn( colours = c("#ff0000",
                                             "#00ff00",
                                             "#0000ff"))
gp <- gp + xlab("Mean of Predictions")
gp <- gp + ylab("Mean of trial durations")
gp <- gp + guides(colour = guide_legend( override.aes = list(size = 4),
                                        title = "Trial\nsequence\nnumber"))

if(plot.offline == TRUE) {
    pdf("trial_duration_vs_prediction_mean.pdf", width = 5, height = 3)
    print(gp)
    dev.off()
} else {
    print(gp)
}
