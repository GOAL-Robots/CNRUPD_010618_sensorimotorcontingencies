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
    time.series.maxes <- 1 * (time.series == time.series.max)
    time.series.maxes.diffs <- c(0, diff(time.series.maxes))
    res <- list()
    for (idx in which(time.series.maxes.diffs == 1)) {
        time.series.maxes.diffs.after <-
            time.series.maxes.diffs[idx:time.series.length]
        test <- any(time.series.maxes.diffs.after < 0)
        if (test == FALSE) {
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
for (goal.el in levels(predictions$goal)) {
    predictions.goal <- predictions[goal == goal.el]
    plateau.index <- find_plateau(predictions.goal$prediction)$idx
    plateau.indices <- c(plateau.indices,  plateau.index)
    plateau.timesteps <- c(plateau.timesteps,
                           predictions.goal$timesteps[plateau.index])
}
plateau.all.index = max(plateau.indices)
timesteps.max <-
    plateau.timesteps[plateau.indices ==
                          plateau.all.index] + timesteps.gap * 4

# WEIGHTS ----------------------------------------------------------------------

weights <- fread("all_weights")
names(weights) <- c("learning.type", "index", "timesteps",
                    "kohonen", "echo")
weights <- subset(weights, index == simulation.index)
weights <- subset(weights, timesteps <= timesteps.max)

# PLOTS ------------------------------------------------------------------------

gp <- ggplot(weights, aes(x = timesteps))
gp <- gp + geom_line(aes(y = kohonen))
gp <- gp + xlab("Timesteps")
gp <- gp + ylab("")
gp <- gp + theme_bw()
gp <- gp + scale_x_continuous(labels = function(x) format(x, scientific = FALSE))

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
gp <- gp + xlab("Timesteps")
gp <- gp + ylab("")
gp <- gp + theme_bw()
gp <- gp + scale_x_continuous(labels = function(x) format(x, scientific = FALSE))
gp <- gp + theme(
    text = element_text(size = 11, family = "Verdana"),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)

gp_echo <- gp

aligned_plots <- align_plots(gp_kohonen, gp_echo, align = "hv")

gp_all <- ggdraw()

gp_all <- gp_all + draw_plot(
    aligned_plots[[1]],
    x = 0.0,
    y = 0.0,
    width = 1,
    height = 0.5
)

gp_all <- gp_all + draw_plot(
    aligned_plots[[2]],
    x = 0.0,
    y = 0.5,
    width = 1,
    height = 0.5
)

if (plot.offline == TRUE) {
    pdf("weights.pdf", width = 6.5, height = 3)
    print(gp_all)
    dev.off()
    postscript("weights.eps", horizontal = FALSE,
               onefile = FALSE, paper = "special",
               fonts = "Verdana",
               width = 6.5, height = 3)
    print(gp_all)
    dev.off()
} else {
    print(gp_all)
}


# WEIGHT GRID ------------------------------------------------------------------

if (file.exists("weights")) {
    weights.final <-
        fread("weights")
    rows <- dim(weights.final)[1]
    goal.side.length <- sqrt(rows)
    cols <- dim(weights.final)[2]
    retina.side.length <- sqrt(cols)

    weights.final$goal <- 1:rows
    weights.final <- melt(weights.final,
                          variable.name = "cell",
                          measure.vars = 1:cols)

    weights.final$cell <-
        as.numeric(sub("V", "", weights.final$cell)) - 1
    weights.final$r_row <-
        weights.final$cell %/% retina.side.length + 1
    weights.final$r_col <-
        weights.final$cell %% retina.side.length + 1

    # WEIGHT_GRID PLOTS ------------------------------------------------------------------------

    grbs = list()

    for (row in 1:rows)
    {
        w <- subset(weights.final, goal == row)
        gp <- ggplot(w, aes(
            x = r_col,
            y = r_row,
            fill = value
        ))
        gp <- gp + geom_raster(show.legend = FALSE)
        gp <- gp + xlab("")
        gp <- gp + ylab("")
        gp <- gp + theme_bw()
        gp <-
            gp + scale_fill_gradient(low = "#ffffff", high = "#000000")
        gp <- gp + theme(
            text = element_text(size = 11, family = "Verdana"),
            axis.ticks = element_blank(),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            #panel.border = element_blank(),
            legend.title = element_blank(),
            legend.background = element_blank(),
            panel.grid.major = element_blank(),
            plot.margin = unit(c(.1, .1, .0, .0), "in"),
            panel.grid.minor = element_blank()
        )
        grbs[[row]] <- ggplotGrob(gp)

    }

    aligned_plots <- align_plots(plotlist = grbs, align = "hv")

    gp_weight_grid <- ggdraw()
    for (row in 1:goal.side.length) {
        for (col in 1:goal.side.length) {
            gp_weight_grid <- gp_weight_grid + draw_plot(
                grbs[[(row - 1) * goal.side.length + col]],
                x = (col - 1) / goal.side.length,
                y = (row - 1) / goal.side.length,
                width = 1 / goal.side.length,
                height = 1 / goal.side.length
            )
        }
    }

    if (plot.offline == TRUE) {
        pdf("weights_grid.pdf",
            width = 6,
            height = 6 * 0.55)
        print(gp_weight_grid)
        dev.off()
    } else {
        print(gp_weight_grid)
    }

}
# POSITION GRID ----------------------------------------------------------------

if (file.exists("positions")) {
    positions <- fread("positions")

    names(positions) <- c("goal", "x", "y")
    positions$joint <- rep(1:8, rows)
    grbs = list()
    for (row in 1:rows)
    {
        w = subset(positions, goal == row - 1)

        gp <-
            ggplot(w, aes(
                x = as.numeric(x),
                y = as.numeric(y),
                order = joint
            ))
        gp <- gp + geom_path(size = 1.5, color = "#555555")
        gp <- gp + geom_point()
        gp <- gp + xlab("")
        gp <- gp + ylab("")
        gp <- gp + theme_bw()
        gp <- gp + scale_x_continuous(limits = c(-3.5, 3.5))
        gp <- gp + scale_y_continuous(limits = c(-0.1, 3))
        gp <- gp + coord_fixed()
        gp <- gp + theme(
            text = element_text(size = 11, family = "Verdana"),
            axis.ticks = element_blank(),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            #panel.border = element_blank(),
            legend.title = element_blank(),
            legend.background = element_blank(),
            panel.grid.major = element_blank(),
            plot.margin = unit(c(.0, .0, .0, .0), "in"),
            panel.grid.minor = element_blank()
        )
        grbs[[row]] = ggplotGrob(gp)
    }

    aligned_plots <- align_plots(plotlist = grbs, align = "hv")

    gp_position_grid <- ggdraw()
    for (row in 1:goal.side.length) {
        for (col in 1:goal.side.length) {
            gp_position_grid <- gp_position_grid + draw_plot(
                grbs[[(row - 1) * goal.side.length + col]],
                x = (col - 1) / goal.side.length,
                y = (row - 1) / goal.side.length,
                width = 1 / goal.side.length,
                height = 1 / goal.side.length
            )
        }
    }

    if (plot.offline == TRUE) {
        pdf("positions_grid.pdf",
            width = 6,
            height = 6 * 0.55)
        print(gp_position_grid)
        dev.off()
    } else {
        print(gp_position_grid)
    }
}
