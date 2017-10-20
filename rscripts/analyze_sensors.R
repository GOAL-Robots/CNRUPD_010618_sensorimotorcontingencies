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

mid.prop.idx = function(x, scale=1) {
    x = as.vector(t(x%o%rep(1,scale)))
    last(which(cumsum(x)/sum(x) < 3/4))/scale
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

prediction.th <- 0.9
sensors.activation.th <- 0.0
simulation.index <- 1
timesteps.gap <- 50e+3


sensors_labels_uniform <- seq(0, 1, length.out = 30)

sensors_labels <- c(
    0.00000000,
    0.05357143,
    0.10714286,
    0.16071429,
    0.21428571,
    0.26785714,
    0.32142857,
    0.37500000,
    0.42857143,
    0.48214286,
    0.53571429,
    0.58928571,
    0.64285714,
    0.69642857,
    0.75000000,
    0.76666667,
    0.78333333,
    0.80000000,
    0.81666667,
    0.83333333,
    0.85000000,
    0.86666667,
    0.88333333,
    0.90000000,
    0.91666667,
    0.93333333,
    0.95000000,
    0.96666667,
    0.98333333,
    1.00000000
)

sensors_labels <- sensors_labels_uniform

# PREDICTIONS ------------------------------------------------------------------

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

global.predictions <- subset(global.predictions,
                             index == simulation.index)

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

# __Find the correct final timestep ====

# find plateau in predictions
plateau.indices <- c()
plateau.timesteps <- c()
for (goal.el in levels(global.predictions$goal)) {
    global.predictions.goal <- global.predictions[goal == goal.el]
    plateau.index <-
        find_plateau(global.predictions.goal$prediction)$idx
    plateau.indices <- c(plateau.indices,  plateau.index)
    plateau.timesteps <- c(plateau.timesteps,
                           global.predictions.goal$timesteps[plateau.index])
}

plateau.all.index = max(plateau.indices)
timesteps.all.learnt <- plateau.timesteps[plateau.indices ==
                                              plateau.all.index]
# select all timesteps to the plateau
global.predictions <-
    subset(global.predictions,
           timesteps <= timesteps.all.learnt + timesteps.gap * 4)

# __ convert goal.current into a factor ====
global.predictions$goal.current <-
    factor(
        goals.labels[global.predictions$goal.current + 1],
        levels = levels(global.predictions$goal),
        labels = levels(global.predictions$goal)
    )

global.predictions <-
    subset(global.predictions, goal == goal.current)
global.predictions$goal.current <-
    as.numeric(global.predictions$goal.current) - 1

# __ maintain only needed variables ====
global.predictions <-
    global.predictions[, .(timesteps,
                           goal.current,
                           prediction)]

timesteps.max <- max(global.predictions$timesteps)

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

global.touches <-
    global.touches[, .(sensor,
                       touch,
                       any = sum(touch) > 0),
                   by = .(timesteps, goal)]

global.touches <- subset(global.touches, any == TRUE)

global.touches <-
    global.touches[, cbind(
        .SD,
        sensor.num = sensors_labels[as.numeric(global.touches$sensor)])]

global.touches <-
    global.touches[, .(touch = mean(touch)),
                   by = sensor.num]
# SENSORS ----------------------------------------------------------------------


# __ load sensors dataset ====
global.sensors <- fread('all_sensors')
global.sensors.number <-
    dim(global.sensors)[2] - 4  # first 4 columns are not goals
global.sensors.labels <-
    paste("S", 1:global.sensors.number, sep = "")
names(global.sensors) <- c("learning.type",
                           "index",
                           "timesteps",
                           global.sensors.labels,
                           "goal.current")

global.sensors <- subset(global.sensors,
                         index == simulation.index)

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

global.sensors <-
    global.sensors[, .(timesteps,
                       goal.current,
                       sensor,
                       activation,
                       any = sum(activation) > 0),
                   by = .(timesteps, goal.current)]

global.sensors <- subset(
    global.sensors,
    any == TRUE
)

get_sensors_window <- function(timesteps.start, timesteps.end) {

    global.sensors.current <-
        subset(global.sensors,
               timesteps >= timesteps.start &
               timesteps <= timesteps.end )

    global.sensors.means <-
        global.sensors.current[, .(activation = mean(activation)),
                       by = .(sensor, goal.current)]
    global.sensors.means <-
        global.sensors.means[, .(activation,
                                 sensor,
                                 goal.current.ordered =
                                     mid.prop.idx(activation, scale=30)),
                             by = .(goal.current)]

    sensor_indices = as.numeric(global.sensors.means$sensor)
    global.sensors.means$sensor.num = sensors_labels[sensor_indices]

    global.sensors.means.global <-
        global.sensors.means[, .(activation = mean(activation)),
                             by = sensor.num]

    res <- list(
        means.goal = global.sensors.means,
        means = global.sensors.means.global )

    res
}
# PLOT FUNCTIONS ---------------------------------------------------------------

plot.sensors.per.goal <- function(means.per.goal) {

    # __ sensors per goal ====
    gp = ggplot(means.per.goal, aes(x = sensor.num, y = activation))

    gp = gp + geom_bar(stat = "identity")

    # goal labels for the ordered goals
    to_label = means.per.goal[, .(ordered = first(goal.current.ordered)),
                              by = goal.current]
    labelled = as.array(as.character(to_label$goal.current))
    names(labelled) = to_label$ordered

    gp = gp + facet_grid(goal.current.ordered ~ .,
                         labeller =
                             labeller(goal.current.ordered =
                                          labelled))


    gp <- gp + scale_y_continuous(
        limits = c(0, 2),
        breaks = c(0, 60),
        sec.axis = sec_axis(
            trans = ~ .,
            breaks = c(),
            labels = means.per.goal$goal.current,
            name = "Goals"
        )
    )

    gp <- gp + scale_x_continuous(
        limits = c(0, 1),
        breaks = sensors_labels,
        labels = round(sensors_labels, 2)
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

    gp
}

plot.sensors <- function(means, xaxis = TRUE, yaxis = TRUE) {

    # __ sensors ====

    gp = ggplot(means,
                aes(x = sensor.num,
                    y = activation))

    gp = gp + geom_ribbon(aes(ymin = 0,
                              ymax = activation),
                          color = "black",
                          fill = "grey")

    gp = gp + geom_line(data = global.touches,
                        aes(x = sensor.num,
                            y = touch),
                        inherit.aes = TRUE)
    gp <-
        gp + scale_y_continuous(
            limits = c(0, 1),
            breaks = c(0, 0.25, 0.5),
            labels = c(0, 0.25, 0.5),
            sec.axis = sec_axis(
                trans = ~ .,
                breaks = c(0, 0.5, 1),
                name = "Proportion of touches"
            )
        )

    gp <-
        gp + scale_x_continuous(
            limits = c(0, 1),
            breaks = sensors_labels,
            labels = round(sensors_labels, 2)
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


    gp
}

# PLOT -------------------------------------------------------------------------

sens <- get_sensors_window(
    timesteps.start = timesteps.all.learnt,
    timesteps.end = timesteps.all.learnt + timesteps.gap)


gp_sensors_per_goal <- plot.sensors.per.goal(sens$means.goal)

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

gp_sensors <- plot.sensors(sens$means)

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


gap = 2000
intervals = 10
gps = list()
gp_ints <- ggdraw()

for(int in 1:intervals) {
    sens.int <- get_sensors_window(timesteps.start = (int - 1) * gap,
                                   timesteps.end = int * gap)

    gp <- plot.sensors(sens.int$means, xaxis = F, yaxis = F)
    gp_ints <- gp_ints + draw_plot(
        gp,
        x = 0,
        y = (int - 1) / intervals,
        width = 1,
        height = gap/(intervals*gap)
    )
}

if (plot.offline == TRUE) {
    pdf(
        paste("sensors_density_history", ".pdf", sep = ""),
        width = 3,
        height = 7,
        family = "Verdana"
    )
    print(gp_ints)
    dev.off()
} else {
    print(gp_ints)
}
