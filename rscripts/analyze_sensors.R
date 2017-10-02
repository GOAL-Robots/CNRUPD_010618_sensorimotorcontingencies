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

prediction.th <- 0.9
sensors.activation.th <- 0.0

# global.predictions ------------------------------------------------------------------

# __ load prediction dataset ====
global.predictions <- fread("all_predictions")
goals.number <-
    dim(global.predictions)[2] - 4 # first 4 columns are not goals
goals.labels <- paste("goal.", 1:goals.number, sep = "")
names(global.predictions) <- c("learning.type",
                        "index",
                        "timesteps",
                        goals.labels,
                        "goal.current")

# __ melt by goal columns making a goal factor ====
global.predictions <- melt(
    global.predictions,
    id.vars <- c("learning.type",
                 "index",
                 "timesteps",
                 "goal.current"),
    measure.vars <- goals.labels,
    variable.name <- "goal",
    value.name <- "prediction"
)

# find the correct final timestep
timesteps.gap <- 50e+3

plateau.indices <- c()
plateau.timesteps <- c()
for(goal.el in levels(global.predictions$goal)) {
    global.predictions.goal <- global.predictions[goal == goal.el]
    plateau.index <- find_plateau(global.predictions.goal$prediction)$idx
    plateau.indices <- c(plateau.indices,  plateau.index)
    plateau.timesteps <- c(plateau.timesteps,
                           global.predictions.goal$timesteps[plateau.index])
}
plateau.all.index = max(plateau.indices)
global.predictions <-
    subset(global.predictions,
           timesteps <=
               plateau.timesteps[plateau.indices ==
                                     plateau.all.index] +timesteps.gap/2)

timesteps.all.learnt <- plateau.timesteps[plateau.indices ==
                                              plateau.all.index]

# __ convert goal.current into a factor ====
global.predictions$goal.current <-
    factor(
        goals.labels[global.predictions$goal.current + 1],
        levels = levels(global.predictions$goal),
        labels = levels(global.predictions$goal)
    )

global.predictions <- subset(global.predictions, goal == goal.current)
global.predictions$goal.current <- as.numeric(global.predictions$goal.current) - 1

# __ maintain only needed variables ====
global.predictions <- global.predictions[, .(timesteps,
                               goal.current,
                               prediction)]

# SENSORS ----------------------------------------------------------------------

global.sensors <- fread('all_sensors')
global.sensors.number <-
    dim(global.sensors)[2] - 4  # first 4 columns are not goals
global.sensors.labels <- paste("S", 1:global.sensors.number, sep = "")
names(global.sensors) <- c("learning.type",
                    "index",
                    "timesteps",
                    global.sensors.labels,
                    "goal.current")

# __ melt by goal columns making a sensor factor ====
global.sensors <- melt(
    global.sensors,
    id.vars = c("learning.type",
                "index",
                "timesteps",
                "goal.current"),
    measure.vars = global.sensors.labels,
    variable.name = "sensor",
    value.name = "activation"
)

# __ Maintain only needed variables and add prediction ====
global.sensors <- global.sensors[, .(timesteps,
                       goal.current,
                       sensor,
                       activation,
                       prediction = global.predictions[
                           global.predictions$goal.current ==
                               goal.current]$prediction),
                   by = .(goal.current)]

# TOUCHES ----------------------------------------------------------------------

global.touches <-
    fread("log_cont_sensors")
global.touches.number = dim(global.touches)[2] - 2
names(global.touches) = c("timesteps",
                   1:global.touches.number,
                   "goal")

global.touches <- melt(
    global.touches,
    id.vars = c("timesteps", "goal"),
    variable.name = "sensor",
    value.name = "touch"
)

global.touches <- global.touches[, .(touch = sum(touch > 0.2)),
                   by = .(sensor)]
global.touches$touch.prop <- global.touches$touch / max(global.touches$touch)

# sensors data -----------------------------------------------------------------

sensors_data = function(timesteps.start, timesteps.stop)
{
    res = list()

    res$predictions = global.predictions

    # __ Define sensors subset ====
    res$sensors <- subset(
        global.sensors,
            timesteps >= timesteps.start &
            timesteps <= timesteps.stop
    )

    # current means
    res$sensors.means <- res$sensors[, .(
        activation.mean = mean(activation),
        activation.count =  sum(activation > sensors.activation.th),
        activation.sd = sd(activation)
    ),
    by = sensor]

    res$sensors.means.max <- max(res$sensors.means$activation.mean)


    # SENSORSxGOAL-----------------------------------------------------------------

    # __  sensors means per goal ====
    res$sensors.goal.means <- res$sensors[, .(
        activation.mean = mean(activation),
        activation.count = sum(activation > sensors.activation.th),
        activation.sd = sd(activation),
        activation.error = std.error(activation)
    ),
    by = .(sensor, goal.current)]

    # Order goals per maximum
    res$sensors.goal.means.max <-
        res$sensors.goal.means[, .(activation.max = max(activation.mean)),
                           by = .(goal.current)]

    activation.max.indices <- c()
    for (activation.max in res$sensors.goal.means.max$activation.max) {
        activation.max.indices <-
            c(
                activation.max.indices,
                which(res$sensors.goal.means$activation.mean ==
                          activation.max)[1]
            )
    }

    res$sensors.goal.means.max$sensor <-
        res$sensors.goal.means[activation.max.indices]$sensor
    res$sensors.goal.means.max <-
        res$sensors.goal.means.max[order(res$sensors.goal.means.max$sensor)]

    res$sensors.goal.means$goal.current.ordered <-
        factor(res$sensors.goal.means$goal.current,
               levels = res$sensors.goal.means.max$goal.current)


    # TOUCHES ----------------------------------------------------------------------

    res$touches <- global.touches

    # return ----

    res
}

# PLOTS ------------------------------------------------------------------------

plot_sensors <- function(sensors_df,
                         xaxis = TRUE,
                         yaxis = TRUE,
                         sd = TRUE)
{
    attach(sensors_df)

    # main ggplot
    gp <- ggplot(sensors.means)

    # variance
    if (sd == TRUE) {
        gp <- gp + geom_ribbon(
            aes(
                x = as.numeric(sensor),
                ymin = 0,
                ymax = activation.mean + activation.sd
            ),
            colour = "#666666",
            fill = "#dddddd",
            inherit.aes = FALSE
        )
    }

    # Activation mean
    gp <- gp + geom_line(
        aes(x = as.numeric(sensor),
            y = activation.mean),
        size = 1,
        colour = "#000000",
        inherit.aes = FALSE
    )

    # Activation counts
    gp <- gp + geom_bar(
        aes(
            x = as.numeric(sensor),
            y = activation.count / max(activation.count) * sensors.means.max
        ),
        stat = "identity",
        alpha = .2,
        inherit.aes = FALSE
    )

    # Touch map 1
    gp <- gp + geom_line(
        data = touches,
        aes(x = as.numeric(sensor),
            y = touch.prop * sensors.means.max),
        size = 1.5,
        colour = "#000000",
        inherit.aes = FALSE
    )

    # Touch map 2
    gp <- gp + geom_line(
        data = touches,
        aes(x = as.numeric(sensor),
            y = touch.prop * sensors.means.max),
        size = 1.0,
        colour = "#ffffff",
        inherit.aes = FALSE
    )

    gp <- gp + xlab("Sensors")
    gp <- gp + ylab("Means of sensor activation")

    gp <-
        gp + scale_y_continuous(
            breaks = c(0, 0.5, 1) / sensors.means.max,
            labels = c(0, 0.5, 1),
            sec.axis = sec_axis(
                trans = ~ .,
                breaks = c(0, 0.5, 1),
                name = "Proportion of touches"
            )
        )

    gp <- gp + theme_bw()

    if (xaxis == TRUE) {
        axis.text.x = element_text(size = 8,
                                   angle = 90,
                                   hjust = 1)
        axis.ticks.x = element_line()
        axis.title.x = element_text()
    } else {
        axis.text.x = element_blank()
        axis.ticks.x = element_blank()
        axis.title.x = element_blank()
    }

    if (yaxis == TRUE) {
        axis.text.y = element_text(size = 10,)
        axis.ticks.y = element_line()
        axis.title.y = element_text()
    } else {
        axis.text.y = element_blank()
        axis.ticks.y = element_blank()
        axis.title.y = element_blank()
    }

    gp <- gp + theme(
        text = element_text(size = 11, family = "Verdana"),
        axis.text.x = axis.text.x,
        axis.text.y = axis.text.y,
        axis.ticks.x = axis.ticks.x,
        axis.ticks.y = axis.ticks.y,
        axis.title.x = axis.title.x,
        axis.title.y = axis.title.y,
        panel.border = element_blank(),
        legend.title = element_blank(),
        legend.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
    )

    detach(sensors_df)

    gp

}

##################################################################


plot_sensors_per_goals <- function(sensors_df)
{
    attach(sensors_df)

    count.max = max(sensors.goal.means$activation.count)

    gp <- ggplot(sensors.goal.means,
                 aes(x = sensor, y = activation.mean))
    gp <-
        gp + geom_bar(aes(y = activation.count), stat = "identity")
    gp <- gp + facet_grid(goal.current.ordered ~ .)
    gp <- gp + scale_y_continuous(
        limits = c(0, count.max*4/3),
        breaks = c(0, 60),
        sec.axis = sec_axis(
            trans = ~ .,
            breaks = c(),
            name = "Goals"
        )
    )
    gp <- gp + xlab("Sensors")
    gp <- gp + ylab("Number of touches")
    gp <- gp + theme_bw()
    gp <- gp + theme(
        text = element_text(size = 11, family = "Verdana"),
        panel.border = element_blank(),
        axis.text.x = element_text(
            size = 8,
            angle = 90,
            hjust = 1
        ),
        axis.text.y = element_text(size = 6),
        strip.text.y = element_text(
            size = 8,
            angle = 0,
            face = "bold"
        ),
        strip.background = element_rect(colour = "#FFFFFF", fill =
                                            "#FFFFFF"),
        legend.title = element_blank(),
        legend.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
    )

    detach(sensors_df)

    gp

}

# MAIN -------------------------------------------------------------------------

sdf <- sensors_data(timesteps.start = timesteps.all.learnt,
                    timesteps.stop = timesteps.all.learnt + timesteps.gap)

gp <- plot_sensors(sdf)


gp_sensors = gp

attach(sdf)
if (plot.offline == TRUE) {
    pdf(
        paste("sensors", ".pdf", sep = ""),
        width = 7,
        height = 3,
        family = "Verdana"
    )
    print(gp_sensors)
    dev.off()
} else {
    print(gp_sensors)
}
detach(sdf)


gp <- plot_sensors_per_goals(sdf)

gp_sensors_per_goal = gp
attach(sdf)
if (plot.offline == TRUE) {
    pdf(
        paste("sensors_per_goal", ".pdf", sep = ""),
        width = 7,
        height = 3,
        family = "Verdana"
    )
    print(gp_sensors_per_goal)
    dev.off()
} else {
    print(gp_sensors_per_goal)
}
detach(sdf)


gps = list()
interval.gap = 10000
interval.number = 5
interval.timesteps = seq(interval.gap,
                         (interval.number) * interval.gap,
                         by = interval.gap)
for (timestep in 1:interval.number) {
    sdf <-
        sensors_data(timesteps.start =
                         interval.timesteps[timestep] -
                         interval.gap,
                     timesteps.stop =
                         interval.timesteps[timestep])
        gp <- plot_sensors(sdf, xaxis = FALSE,
                           yaxis = FALSE,
                           sd = FALSE)

    gps[[timestep]] = gp
}

attach(sdf)
gp_all <- ggdraw()
for (timestep in 1:interval.number) {
    gp_all <- gp_all + draw_plot(gps[[timestep]] ,
                         0,
                         (1/interval.number)*(timestep-1),
                         1,
                         (1/interval.number))
}
print(gp_all)
detach(sdf)
