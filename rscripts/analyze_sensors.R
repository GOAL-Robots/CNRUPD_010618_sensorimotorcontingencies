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

# CONSTS -----------------------------------------------------------------------

prediction.th <- 0.9
sensors.activation.th <- 0.1

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
    id.vars <- c("learning.type",
                 "index",
                 "timesteps",
                 "goal.current"),
    measure.vars <- goals.labels,
    variable.name <- "goal",
    value.name <- "prediction"
)

# __ convert goal.current into a factor ====
predictions$goal.current <-
    factor(
        goals.labels[predictions$goal.current + 1],
        levels = levels(predictions$goal),
        labels = levels(predictions$goal)
    )

predictions <- subset(predictions, goal == goal.current)
predictions$goal.current <- as.numeric(predictions$goal.current) - 1

# __ maintain only needed variables ====
predictions <- predictions[, .(timesteps,
                               goal.current,
                               prediction)]

# SENSORS ----------------------------------------------------------------------

sensors_data = function(timesteps.start, timesteps.stop)
{
    res = list()

    sensors <- fread('all_sensors')
    sensors.number <-
        dim(sensors)[2] - 4  # first 4 columns are not goals
    sensors.labels <- paste("S", 1:sensors.number, sep = "")
    names(sensors) <- c("learning.type",
                        "index",
                        "timesteps",
                        sensors.labels,
                        "goal.current")

    # __ melt by goal columns making a sensor factor ====
    sensors <- melt(
        sensors,
        id.vars = c("learning.type",
                    "index",
                    "timesteps",
                    "goal.current"),
        measure.vars = sensors.labels,
        variable.name = "sensor",
        value.name = "activation"
    )

    # __ Maintain only needed variables and add prediction ====
    sensors <- sensors[, .(timesteps,
                           goal.current,
                           sensor,
                           activation,
                           prediction = predictions[predictions$goal.current == goal.current]$prediction),
                       by = .(goal.current)]



    # __ Define sensors subset ====
    sensors <- subset(
        sensors,
        prediction >= prediction.th &
            timesteps >= timesteps.start &
            timesteps <= timesteps.stop
    )

    # current means
    sensors.means <- sensors[, .(
        activation.mean = mean(activation),
        activation.count =
            sum(activation > sensors.activation.th),
        activation.sd = sd(activation)
    ),
    by = sensor]

    sensors.means.max <- max(sensors.means$activation.mean)


    res$sensors <- sensors
    res$sensors.means <- sensors.means
    res$sensors.means.max <- sensors.means.max


    # SENSORSxGOAL-----------------------------------------------------------------

    # __  sensors means per goal ====
    sensors.goal.means <- sensors[, .(
        activation.mean = mean(activation),
        activation.count = sum(activation > sensors.activation.th),
        activation.sd = sd(activation),
        activation.error = std.error(activation)
    ),
    by = .(sensor, goal.current)]

    # Order goals per maximum
    sensors.goal.means.max <-
        sensors.goal.means[, .(activation.max = max(activation.mean)),
                           by = .(goal.current)]

    activation.max.indices <- c()
    for (activation.max in sensors.goal.means.max$activation.max) {
        activation.max.indices <-
            c(
                activation.max.indices,
                which(sensors.goal.means$activation.mean ==
                          activation.max)[1]
            )
    }

    sensors.goal.means.max$sensor <-
        sensors.goal.means[activation.max.indices]$sensor
    sensors.goal.means.max <-
        sensors.goal.means.max[order(sensors.goal.means.max$sensor)]

    sensors.goal.means$goal.current.ordered <-
        factor(sensors.goal.means$goal.current,
               levels = sensors.goal.means.max$goal.current)


    res$sensors.goal.means <- sensors.goal.means
    res$sensors.goal.means.max <- sensors.goal.means.max

    # TOUCHES ----------------------------------------------------------------------

    touches <-
        fread("~/tmp/sensorimotor/local/count/1/data/log_cont_sensors")
    touches.number = dim(touches)[2] - 2
    names(touches) = c("timesteps",
                       1:touches.number,
                       "goal")

    touches <- melt(
        touches,
        id.vars = c("timesteps", "goal"),
        variable.name = "sensor",
        value.name = "touch"
    )

    touches <- touches[, .(touch = sum(touch > 0.2)),
                       by = .(sensor)]
    touches$touch.prop <- touches$touch / max(touches$touch)

    res$touches <- touches

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
            y = activation.count / max(activation.count) *
                sensors.means.max
        ),
        stat = "identity",
        alpha = .2,
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

    gp <- ggplot(sensors.goal.means,
                 aes(x = sensor, y = activation.mean))
    gp <-
        gp + geom_bar(aes(y = activation.count), stat = "identity")
    gp <- gp + facet_grid(goal.current.ordered ~ .)
    gp <- gp + scale_y_continuous(
        limits = c(0, 70),
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

sdf <- sensors_data(timesteps.start = 200000, 250000)

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
