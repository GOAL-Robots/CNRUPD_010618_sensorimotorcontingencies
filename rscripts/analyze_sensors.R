# INTRO ------------------------------------------------------------------------

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

offline.plot <- TRUE

# UTILS ------------------------------------------------------------------------

# standard error
std.error <- function(x) {
    sd(x) / sqrt(length(x))
}

# CONSTS -----------------------------------------------------------------------

prediction.th <- 0.9
timesteps.start <- 200000
timesteps.stop <- 250000
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
sensors <- fread('all_sensors')
sensors.number <- dim(sensors)[2] - 4  # first 4 columns are not goals
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
sensors <- sensors[, .(
    timesteps,
    goal.current,
    sensor,
    activation,
    prediction = predictions[predictions$goal.current == goal.current]$prediction),
by = .(goal.current)]


# __ Define sensors subset ====
sensors <- subset(sensors, prediction >= prediction.th)
sensors <- subset(sensors,
                 timesteps >= timesteps.start &
                     timesteps < timesteps.stop)

# __  sensors means ====
sensors.means <- sensors[, .(
    activation.mean = mean(activation),
    activation.count = sum(activation > sensors.activation.th),
    activation.sd = sd(activation),
    activation.error = std.error(activation)
),
by = .(sensor)]

# __  sensors means per goal ====
sensors.goal.means <- sensors[, .(
    activation.mean = mean(activation),
    activation.count = sum(activation > sensors.activation.th),
    activation.sd = sd(activation),
    activation.error = std.error(activation)
),
by = .(sensor, goal.current)]

#---------------------------------------------------------------------

# Order goals per maximum
sensors.goal.means.max <-
    sensors.goal.means[, .(
        activation.max = max(activation.mean)),
        by = .(goal.current)]

activation.max.indices <- c()
for(activation.max in sensors.goal.means.max$activation.max) {
    activation.max.indices <-
        c(activation.max.indices,
          which(sensors.goal.means$activation.mean ==
                    activation.max)[1])
}

sensors.goal.means.max$sensor <-
    sensors.goal.means[activation.max.indices]$sensor
sensors.goal.means.max = sensors.goal.means.max[order(sensors.goal.means.max$sensor)]

sensors.goal.means$goal.current.ordered <-
    factor(sensors.goal.means$goal.current,
           levels = sensors.goal.means.max$goal.current)

#---------------------------------------------------------------------

sensors.activation.count.tot = sum(sensors.means$activation.count)

gp <- ggplot(sensors.means,
             aes(x = sensor,
                 y = activation.mean, group=1))


gp <- gp + geom_ribbon(
    aes(ymin = 0, ymax = activation.mean + activation.sd),
    colour = "#666666",
    fill = "#dddddd"
)
gp <- gp + geom_line(size = 1.5, colour = "#000000")

gp <- gp + geom_bar(
    aes(y = 5 * activation.count /
            sensors.activation.count.tot),
    stat = "identity",
    alpha = .2
)

gp <- gp + xlab("Sensors")
gp <- gp + ylab("Means of sensor activation")
gp <- gp + scale_y_continuous(
    sec.axis = sec_axis(trans = ~ . / 5,
                        name = "Proportion of touches"))

gp <- gp + theme_bw()
gp <- gp + theme(
    text = element_text(size = 11, family = "Verdana"),
    axis.text.x = element_text(
        size = 8,
        angle = 90,
        hjust = 1
    ),
    axis.text.y = element_text(size = 10),
    panel.border = element_blank(),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)

gp_sensors <- gp

basename <- "sensors"
pdf(
    paste(basename, ".pdf", sep = ""),
    width = 7,
    height = 3,
    family = "Verdana"
)
print(gp_sensors)
dev.off()
bitmap(
    type = "png256",
    paste(basename, ".png", sep = ""),
    width = 7,
    height = 3,
    family = "Verdana"
)
print(gp_sensors)
dev.off()
print(gp_sensors)


gp <- ggplot(
    subset(sensors.goal.means),
    aes(x = sensor, y = activation.mean)
)
gp <- gp + geom_bar(aes(y = activation.count), stat = "identity")
gp <- gp + facet_grid(goal.current.ordered ~ .)
gp <- gp + scale_y_continuous(
    limits = c(0, 90),
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
    axis.text.y = element_text(size = 8),
    strip.text.y = element_text(size = 8,
                                angle = 0,
                                face = "bold"),
    strip.background = element_rect(colour = "#FFFFFF", fill =
                                        "#FFFFFF"),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)


gp_sensors <- gp

pdf(
    "sensors_per_goal.pdf",
    width = 7,
    height = 7,
    family = "Verdana"
)
print(gp_sensors)
dev.off()
bitmap(
    type = "png256",
    paste(basename, ".png", sep = ""),
    width = 7,
    height = 7,
    family = "Verdana"
)
print(gp_sensors)
dev.off()
print(gp_sensors)

