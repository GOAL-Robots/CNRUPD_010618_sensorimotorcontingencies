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
rm(list = ls())

# INTRO ------------------------------------------------------------------------

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

# WEIGHTS ----------------------------------------------------------------------

weights <- fread("all_weights")
names(weights) <- c("learning.type",
                    "index",
                    "timesteps",
                    "kohonen",
                    "echo")
weights <- subset(weights, index == simulation.index)

#timesteps.max <- max(weights$timesteps)
#timesteps.number <- length(weights$timesteps)
timesteps.number <- length(weights[timesteps <= timesteps.max]$timesteps)
timesteps.all <- timesteps.max

# PLOTS ------------------------------------------------------------------------

plot_hystory <- function(timesteps.start, timesteps.stop, blocks = 5)
{
    predictions.current <-
        subset(predictions, timesteps >= timesteps.start &
                   timesteps <= timesteps.stop)


    timesteps <- predictions.current$timesteps
    timesteps.length = as.integer(length(timesteps))
    blocks.number <- blocks
    timesteps.slices.breaks <-
        timesteps[seq(1, timesteps.length,
                      length.out = blocks.number)]
    predictions.slices <-
        subset(predictions.current, timesteps %in% timesteps.slices.breaks)

    predictions.slices <- predictions.slices[order(timesteps)]

    goal.width <- sqrt(max(as.numeric(predictions.slices$goal)))

    predictions.slices$row <-
        (as.numeric(predictions.slices$goal) - 1) %% goal.width
    predictions.slices$col <-
        (as.numeric(predictions.slices$goal) - 1) %/% goal.width


    gps = list()

    for (timestep in 1:length(timesteps.slices.breaks))
    {
        data <-
            subset(predictions.slices,
                   timesteps == timesteps.slices.breaks[timestep])
        gp <- ggplot(data)
        gp <- gp + geom_rect(
            aes(
                xmin = col - 0.4 * prediction,
                xmax = col + 0.4 * prediction,
                ymin = row - 0.4 * prediction,
                ymax = row + 0.4 * prediction
            ),
            fill = "black"
        )
        gp <- gp + geom_text(aes(
            x = 1,
            y = 4.8,
            label = paste("Timestep:",  timesteps.slices.breaks[timestep])
        ),
        size = 1.5,
        inherit.aes = FALSE)

        gp <- gp + scale_x_continuous(limits = c(-1, 5),
                                      breaks = 0:4)
        gp <- gp + scale_y_continuous(limits = c(-1, 5),
                                      breaks = 0:4)

        gp <- gp + xlab("")
        gp <- gp + ylab("")
        gp = gp + theme_bw()

        gp <- gp + theme(
            text = element_text(size = 11, family = "Verdana"),
            legend.title = element_blank(),
            legend.background = element_blank(),
            panel.grid.major = element_blank(),
            plot.margin = unit(c(.0, .0, .0, .0), "in"),
            panel.grid.minor = element_blank()
        )


        gps[[timestep]] <- ggplotGrob(gp)
    }

    gp = ggdraw()
    for(x in 1:(blocks.number)) {
        gp = gp + draw_plot(gps[[x]],
                            x = (x-1)/(blocks.number),
                            y = 0,
                            width = 1/(blocks.number),
                            height = 1)
    }

    gp

}

gp1 = plot_hystory(0, timesteps.max*(5/10), blocks = 5)
gp2 = plot_hystory(timesteps.max*(6/10), timesteps.max, blocks = 5)

gp_all = ggdraw()

gp_all = gp_all + draw_plot(gp1,
                            x = 0,
                            y = 0.5,
                            width=1,
                            height=0.5)

gp_all = gp_all + draw_plot(gp2,
                            x = 0,
                            y = 0.0,
                            width=1,
                            height=0.5)
if (plot.offline == TRUE) {
    pdf("pred_hist.pdf", width = 6, height = 2.4)
    print(gp_all)
    dev.off()
} else {
    print(gp_all)
}

