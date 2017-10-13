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
if (file.exists("OFFLINE")) {
    plot.offline = TRUE
}

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

timesteps.gap <- 20e+3
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
timesteps.max <-
    plateau.timesteps[plateau.indices ==
                          plateau.all.index] +timesteps.gap*4
# WEIGHTS ----------------------------------------------------------------------

weights <- fread("all_weights")
names(weights) <- c("learning.type", "index", "timesteps",
                    "kohonen", "echo")
weights <- subset(weights, index == simulation.index)
weights <- subset(weights, timesteps <= timesteps.max )

gp <- ggplot(weights, aes(x = timesteps))
gp <- gp + geom_line(aes(y = kohonen))
gp <- gp + xlab("timesteps")
gp <- gp + ylab("")
gp <- gp + theme_bw()

gp <- gp + theme(
    text = element_text(size = 11, family = "Verdana"),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)

gp_kohonen <- gp

gp <- ggplot(weights, aes(x = timesteps))
gp <- gp + geom_line(aes(y = echo))
gp <- gp + xlab("timesteps")
gp <- gp + ylab("")
gp <- gp + theme_bw()

gp <- gp + theme(
    text = element_text(size = 11, family = "Verdana"),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)
gp_echo <- gp


aligned_plots <- align_plots(gp_kohonen, gp_echo, align="hv")

gp_all <- ggdraw()

gp_all <- gp_all + draw_plot(aligned_plots[[1]],
                    x = 0.0, y = 0.0,
                    width = 1, height = 0.5)

gp_all <- gp_all + draw_plot(aligned_plots[[2]],
                             x = 0.0, y = 0.5,
                             width = 1, height = 0.5)


if(plot.offline == TRUE) {
    pdf("weights.pdf", width = 6, height = 3)
    print(gp_all)
    dev.off()
} else {
    print(gp_all)
}
