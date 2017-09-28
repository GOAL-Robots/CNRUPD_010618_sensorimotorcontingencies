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



# CONSTS -----------------------------------------------------------------------


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

# PLOTS ------------------------------------------------------------------------

plot_hystory <- function(timesteps.start, timesteps.stop)
{
    presictions.current <-
        subset(predictions, timesteps >= timesteps.start &
                   timesteps <= timesteps.stop)


    timesteps <- presictions.current$timesteps
    timesteps.length = as.integer(length(timesteps))
    blocks.number <- 5
    timesteps.slices.breaks <-
        timesteps[seq(1, timesteps.length,
                      length.out = blocks.number)]
    predictions.slices <-
        subset(presictions.current, timesteps %in% timesteps.slices.breaks)

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
    for(x in 1:blocks.number) {
        gp = gp + draw_plot(gps[[x]],
                            x = (x - 1)/blocks.number,
                            y = 0,
                            width = 1/blocks.number,
                            height = 1 )
    }

    gp

}

gp = plot_hystory(0, 240000)

if (plot.offline == TRUE) {
    pdf("pred_hist.pdf", width = 6, height = 1.2)
    print(gp)
    dev.off()
} else {
    print(gp)
}

