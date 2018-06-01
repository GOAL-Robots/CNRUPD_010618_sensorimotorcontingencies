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
reset <- function() {

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
    scales <- 10^seq(1, 10)
    x.scaled <- x/scales
    test.scales <- x.scaled < 10 & x.scaled > 1

    scales[which(test.scales == TRUE)]
}


analyze_predictions <- function(offline = False) {
    # CONSTS -----------------------------------------------------------------------

    timesteps.last <- 410e+3
    timesteps.gap <- 20e+3

    weights <- fread("all_weights")
    names(weights) <- c("learning.type",
                        "index",
                        "timesteps",
                        "kohonen",
                        "echo")
    timesteps.max <- max(weights$timesteps)
    timesteps.number <- length(weights$timesteps)
    timesteps.all <- timesteps.max


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
    predictions$timesteps <-
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
                                     by = .(timesteps)]
    predictions.means$th = 1

    # PLOTS ------------------------------------------------------------------------

    # __ Plot story function ====

    plot_story <- function(timesteps.start, timesteps.stop) {
        weights.current <- subset(weights,
                                  timesteps >= timesteps.start &
                                      timesteps <= timesteps.stop)
        predictions.current <- subset(predictions,
                                      timesteps >= timesteps.start &
                                          timesteps <= timesteps.stop)
        predictions.means.current <- subset(predictions.means,
                                            timesteps >= timesteps.start &
                                                timesteps <= timesteps.stop)

        trials.number <- length(weights.current$timesteps)
        scale <- find.decimal.scale(trials.number)
        trials.seq <- 1:trials.number
        trials.breaks <- trials.seq[trials.seq %% (scale) == 0]
        trials.breaks.timesteps = weights.current$timesteps[trials.breaks]

        scale <- find.decimal.scale(timesteps.stop - timesteps.start)
        timesteps.seq <- timesteps.start:timesteps.stop
        timesteps.breaks <-
            c(0, timesteps.seq[timesteps.seq %% (scale) == 0])


        gp <- ggplot(data = predictions.means.current,
                     aes(x = timesteps,
                         y = pred.mean))

        gp <- gp + geom_point(
            data = predictions.current,
            aes(
                x = timesteps,
                y = 1.2 + 0.4 * as.numeric(goal.current) / max(goals.number)
            ),
            size = 0.3,
            stroke = 0,
            inherit.aes = FALSE
        )

        gp <- gp + geom_ribbon(
            data = predictions.means.current,
            aes(ymin = pred.min,
                ymax = pred.max),
            colour = NA,
            fill = "#dddddd",
            size = 0.0
        )

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

        gp <- gp + geom_line(size = .5,
                             colour = "#000000")

        gp <- gp + geom_line(
            aes(x = timesteps,
                y = th),
            size = 0.1,
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

    gp_all = plot_story(timesteps.start = 0,
                        timesteps.stop = timesteps.max)

    pdf("means_all.pdf", width = 7, height = 3)

    print(gp_all)
    dev.off()
    print(gp_all)

    gp_first = plot_story(timesteps.start = 0,
                          timesteps.stop = timesteps.gap)
    pdf("means_first.pdf", width = 7, height = 3)
    print(gp_first)
    dev.off()
    print(gp_first)

    gp_last = plot_story(timesteps.start = timesteps.max - timesteps.gap,
                         timesteps.stop = timesteps.max)
    pdf("means_last.pdf", width = 7, height = 3)
    print(gp_last)
    dev.off()
    print(gp_last)


    gp = ggdraw()
    gp = gp + draw_plot(gp_all, 0, 0.5, 1, 0.5)
    gp = gp + draw_plot(gpfirst, 0, 0, 0.48, 0.5)
    gp = gp + draw_plot(gp_last, 0.5, 0, 0.48, 0.5)


    gp_comp = gp
    pdf("means_comp.pdf", width = 7, height = 4)
    print(gp_comp)
    dev.off(gp)
    print(gp_comp)

}
