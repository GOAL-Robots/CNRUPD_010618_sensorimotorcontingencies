rm(list = ls())

# INTRO ------------------------------------------------------------------------

# __ list of required packages ====
toInstall <- c("extrafont",
               "ggplot2",54
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
    scales <- 10 ^ seq(1, 10)
    x.scaled <- x / scales
    test.scales <- x.scaled < 10 & x.scaled > 1

    scales[which(test.scales == TRUE)]
}

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

# PREDICTIONS ------------------------------------------------------------------

# __ load prediction dataset ====
predictions <- fread("all_predictions")
goals.number <-
    dim(predictions)[2] - 4 # first 4 columns are not goals
goals.labels <- paste("goal.", 1:goals.number, sep = "")
names(predictions) <- c("learning.type",
                        "index",
                        "timesteps",
                        goals.labels,
                        "goal.current")

predictions <- subset(predictions, index == simulation.index)



# __ melt by goal columns making a goal factor ====
predictions <- melt(
    predictions,
    id.vars = c("learning.type",
                "index",
                "timesteps",
                "goal.current"),
    measure.vars = goals.labels,
    variable.name = "goal",
    value.name = "prediction"
)


# find the correct final timestep
plateau.indices <- c()
plateau.timesteps <- c()
for(goal.el in levels(predictions$goal)) {
    predictions.goal <- predictions[goal == goal.el]
    plateau.index <- find_plateau(predictions.goal$prediction)$idx
    plateau.indices <- c(plateau.indices,  plateau.index)
    plateau.timesteps <- c(plateau.timesteps,
                           predictions.goal$timesteps[plateau.index])
}
plateau.all.index = max(plateau.indices)
predictions <-
    subset(predictions,
           timesteps <=
               plateau.timesteps[plateau.indices ==
                                     plateau.all.index] +timesteps.gap/2)

timesteps.max <- max(predictions$timesteps)


# __ convert goal into a factor ====
predictions$goal = factor(predictions$goal)

# __ convert goal.current into a factor ====
predictions$goal.current <-
    factor(
        goals.labels[predictions$goal.current + 1],
        levels = levels(predictions$goal),
        labels = levels(predictions$goal)
    )

# __ maintain only needed variables ====
predictions <- predictions[, .(timesteps,
                               goal.current,
                               goal,
                               prediction)]

# __ timestep bins ====
predictions$timesteps_bins <-
    floor(predictions$timesteps / 1000) * 1000

# __ means ====
predictions.means <- predictions[,
                                 .(
                                     pred.mean = mean(prediction),
                                     pred.sd = sd(prediction),
                                     pred.err = std.error(prediction),
                                     pred.min = min(prediction),
                                     pred.max = max(prediction)
                                 ),
                                 by = .(timesteps = timesteps_bins)]
predictions.means$th = 1


# WEIGHTS ----------------------------------------------------------------------
#
weights <- fread("all_weights")
names(weights) <- c("learning.type",
                    "index",
                    "timesteps",
                    "kohonen",
                    "echo")

weights <- subset(weights, index == simulation.index)
weights <- subset(weights, timesteps <= timesteps.max)

timesteps.number <- length(weights$timesteps)
timesteps.all <- timesteps.max


# PLOTS ------------------------------------------------------------------------

plot_preds_per_goal <- function(timesteps.start, timesteps.stop, goal_focus, raster_height = 0.4) {

    #data
    weights.current <- subset(weights,
                              timesteps >= timesteps.start &
                                  timesteps <= timesteps.stop)
    predictions.current <- subset(predictions,
                                  timesteps >= timesteps.start &
                                      timesteps <= timesteps.stop)

    # tics
    trials.number <- length(weights.current$timesteps)
    scale <- find.decimal.scale(trials.number)
    trials.seq <- 1:trials.number
    trials.breaks <- trials.seq[trials.seq %% (scale) == 0]
    trials.breaks.timesteps = weights.current$timesteps[trials.breaks]
    scale <-
        find.decimal.scale(timesteps.stop - timesteps.start)
    timesteps.seq <- timesteps.start:timesteps.stop
    timesteps.breaks <-
        c(0, timesteps.seq[timesteps.seq %% (scale) == 0])

    #main ggplot
    gp <- ggplot(data = predictions.current,
                 aes(x = timesteps,
                     y = pred.mean))

    # rasterplot of matches
    gp <- gp + geom_point(
        data = predictions.current,
        aes(
            x = timesteps,
            y = 1.2 + raster_height * as.numeric(goal.current) / max(goals.number),
            group = goal.current),
        size = 0.3,
        stroke = 0,
        inherit.aes = FALSE,
        show.legend = FALSE
    )

    # rasterplot of matches
    gp <- gp + geom_point(
        data = predictions.current[as.numeric(goal.current) == goal_focus],
        aes(
            x = timesteps,
            y = 1.2 + raster_height * as.numeric(goal.current) / max(goals.number),
            group = goal.current),
        size = 1,
        stroke = 0,
        inherit.aes = FALSE,
        show.legend = FALSE
    )
    gp <- gp + geom_line(
        aes(x = timesteps,
            y = prediction,
            group = goal,
            color = goal),
        size = 0.3,
        inherit.aes = FALSE,
        show.legend = FALSE
    )

    gp <- gp + geom_line(
        data = predictions.current[as.numeric(goal) == goal_focus],
        aes(x = timesteps,
            y = prediction,
            group = goal,
            color = goal),
        size = 1,
        inherit.aes = FALSE,
        show.legend = FALSE
    )
    gp <- gp + scale_y_continuous(
        limits = c(0, 1.5),
        breaks = c(0, .5, 1),
        labels = c("0.0", "0.5", "1.0")
    )

    gp <- gp + scale_x_continuous(
        limits = c(timesteps.start, timesteps.stop),
        breaks = trials.breaks.timesteps,
        labels = trials.breaks,
        sec.axis = sec_axis(
            ~ .,
            name = "Timesteps",
            breaks = timesteps.breaks,
            labels = timesteps.breaks
        )
    )
    gp <- gp + xlab("Trials")
    gp <- gp + ylab("")
    gp <- gp + theme_bw()

    gp <- gp + theme(
        text = element_text(size = 11, family = "Verdana"),
        panel.border = element_blank(),
        legend.title = element_blank(),
        legend.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
    )

    gp
}

plot_means <- function(timesteps.start, timesteps.stop, raster_height = 0.4) {

    #data
    weights.current <- subset(weights,
                              timesteps >= timesteps.start &
                                  timesteps <= timesteps.stop)
    predictions.current <- subset(predictions,
                                  timesteps >= timesteps.start &
                                      timesteps <= timesteps.stop)
    predictions.means.current <- subset(predictions.means,
                                        timesteps >= timesteps.start &
                                            timesteps <= timesteps.stop)

    # ticks
    trials.number <- length(weights.current$timesteps)
    scale <- find.decimal.scale(trials.number)
    trials.seq <- 1:trials.number
    trials.breaks <- trials.seq[trials.seq %% (scale) == 0]
    trials.breaks.timesteps = weights.current$timesteps[trials.breaks]
    scale <-
        find.decimal.scale(timesteps.stop - timesteps.start)
    timesteps.seq <- timesteps.start:timesteps.stop
    timesteps.breaks <-
        c(0, timesteps.seq[timesteps.seq %% (scale) == 0])

    # main ggplot
    gp <- ggplot(data = predictions.means.current,
                 aes(x = timesteps,
                     y = pred.mean))

    # rasterplot of matches
    gp <- gp + geom_point(
        data = predictions.current,
        aes(
            x = timesteps,
            y = 1.2 + raster_height * as.numeric(goal.current) / max(goals.number)
        ),
        size = 0.3,
        stroke = 0,
        inherit.aes = FALSE
    )

    # span of predictions (min vs max)
    gp <- gp + geom_ribbon(
        data = predictions.means.current,
        aes(ymin = pred.min,
            ymax = pred.max),
        colour = NA,
        fill = "#dddddd",
        size = 0.0
    )

    # variance of predictions
    gp <- gp + geom_ribbon(
        data = predictions.means.current,
        aes(
            ymin = pmax(0, pred.mean - pred.sd),
            ymax = pmin(1, pred.mean + pred.sd)
        ),
        colour = NA,
        fill = "#aaaaaa",
        size = 0.0
    )

    # mean
    gp <- gp + geom_line(size = .5,
                         colour = "#000000")

    # upper threshold (probability 1)
    gp <- gp + geom_line(
        aes(x = timesteps,
            y = th),
        size = 0.1,
        inherit.aes = FALSE,
        show.legend = FALSE
    )

    # scales
    gp <- gp + scale_y_continuous(
        limits = c(0, 1.5),
        breaks = c(0, .5, 1),
        labels = c("0.0", "0.5", "1.0")
    )

    gp <- gp + scale_x_continuous(
        limits = c(timesteps.start, timesteps.stop),
        breaks = trials.breaks.timesteps,
        labels = trials.breaks,
        sec.axis = sec_axis(
            ~ .,
            name = "Timesteps",
            breaks = timesteps.breaks,
            labels = timesteps.breaks
        )
    )

    # styles
    gp <- gp + xlab("Trials")
    gp <- gp + ylab("")
    gp <- gp + theme_bw()
    gp <- gp + theme(
        text = element_text(size = 11, family = "Verdana"),
        panel.border = element_blank(),
        legend.title = element_blank(),
        legend.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
    )

    #return
    gp

}

gp_all = plot_means(timesteps.start = 0,
                    timesteps.stop = timesteps.max, raster_height = 0.25)

if (plot.offline == TRUE) {
    pdf("means_all.pdf", width = 7, height = 3)
    print(gp_all)
    dev.off()
} else {
    print(gp_all)
}

gp_first = plot_means(timesteps.start = 0,
                      timesteps.stop = timesteps.gap,
                      raster_height = 0.25)
if (plot.offline == TRUE) {
    pdf("means_first.pdf", width = 7, height = 3)
    print(gp_first)
    dev.off()
} else {
    print(gp_first)
}

gp_last = plot_means(timesteps.start = timesteps.max - timesteps.gap,
                     timesteps.stop = timesteps.max, raster_height = 0.25)
if (plot.offline == TRUE) {
    pdf("means_last.pdf", width = 7, height = 3)
    print(gp_last)
    dev.off()
} else {
    print(gp_last)
}

gp <- ggdraw()
gp <- gp + draw_plot(gp_all, 0, 0.5, 1, 0.5)
gp <- gp + draw_plot(gp_first, 0, 0, 0.48, 0.5)
gp <- gp + draw_plot(gp_last, 0.5, 0, 0.48, 0.5)
gp_comp <- gp

if (plot.offline == TRUE) {
    pdf("prediction_history.pdf", width = 7, height = 3)
    print(gp_comp)
    dev.off()
} else {
    print(gp_comp)
}

gp_per_goal = plot_preds_per_goal(timesteps.start = 0,
                                  timesteps.stop = timesteps.max,
                                  goal_focus = -1)

if (plot.offline == TRUE) {
    pdf("predictions_per_goal.pdf", width = 7, height = 3)
    print(gp_per_goal)
    dev.off()
} else {
    print(gp_per_goal)
}


gp_per_goal = plot_preds_per_goal(timesteps.start = 0,
                                  timesteps.stop = 30000,
                                  goal_focus = 13)

if (plot.offline == TRUE) {
    pdf("means_per_goal_example.pdf", width = 7, height = 3)
    print(gp_per_goal)
    dev.off()
} else {
    print(gp_per_goal)
}
