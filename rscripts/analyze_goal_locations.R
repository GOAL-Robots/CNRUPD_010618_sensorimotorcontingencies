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

# last non-occurrence of a given number in timeseries
# :param x: the timeseries
# :param a: a number to search for in the timeseries
# :return: a vector(length(x)) of zeros with a one in the
#          position of the last non-occurence of 'a'
last.non.occurr <- function(x, a)
{
    idx <- last(which( x != a))

    res <- rep(0, length(x))

    if (length(idx) > 0) {
        after <- c(rep(0, idx - 1),   # zeros before idx
                   rep(1, length(x) - (idx - 1)))   # ones after idx
        res <- c(0, diff(after)) * (x > 0)    # ositive difference
    }

    res
}

# first occurrence of a given number in timeseries
# :param x: the timeseries
# :param a: a number to search for in the timeseries
# :return: a vector(length(x)) of zeros with a one in the
#          position of the first occurence of 'a'
first.occurr <- function(x, a)
{
    idx <- first(which( x != a))

    res <- rep(0, length(x))

    if (length(idx) > 0) {
        res[idx] = 1
    }

    res
}

# CONSTS -----------------------------------------------------------------------

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
    value.name = "prediction")

# convert goal into a factor
predictions$goal = factor(predictions$goal)

# convert goal.current into a factor
predictions$goal.current <-
    factor(
        goals.labels[predictions$goal.current + 1],
        levels = levels(predictions$goal),
        labels = levels(predictions$goal)
    )

# select only current goals
predictions <- subset(predictions, goal == goal.current)

# maintain only needed variables
predictions <- predictions[, .(timesteps,
                               goal.current,
                               prediction)]


# SENSORS ----------------------------------------------------------------------

# __ load prediction dataset ====
sensors <- fread('all_sensors')
sensors.number <-
    dim(sensors)[2] - 4 # first 4 columns are not sensors
sensors.labels <- paste("sensor.",
                        1:sensors.number,
                        sep = "")
names(sensors) <- c("learning.type",
                    "index",
                    "timesteps",
                    sensors.labels,
                    "goal.current")
sensors <- subset(sensors, index == simulation.index)

# __ melt by sensor columns making a sensor factor ====
sensors$prediction =  predictions$prediction
sensors = melt(
    sensors,
    id.vars = c(
        "learning.type",
        "index",
        "timesteps",
        "goal.current",
        "prediction"
    ),
    measure.vars = sensors.labels,
    variable.name = "sensor",
    value.name = "activation"
)

# maintain only needed variables
sensors <- sensors[, .(timesteps,
                       goal.current,
                       prediction,
                       sensor,
                       activation)]


sensors <- sensors[, .(timesteps,
                       prediction,
                       activation,
                       seq = order(timesteps)), # <---- added ordinal sequences of matches
                   by = .(goal.current, sensor)]


which_weight_mean <- function(x) {
    errs <- abs(x - mean(x))
    idx <- which(errs == min(errs))
    idx
}
sensors.max  <-
    sensors[, .(sensor = sensor[which_weight_mean(activation)],
                timesteps = timesteps[which_weight_mean(activation)],
                prediction = prediction[which_weight_mean(activation)],
                activation = activation[which_weight_mean(activation)]),
            by = .(goal.current, seq)]

sensors.max <- sensors.max[activation != 0]

sensors.max  <- sensors.max[, .(seq = order(seq),
                                sensor = sensor,
                                timesteps = timesteps,
                                prediction = prediction,
                                activation = activation),
                            by = .(goal.current)]


window = 2
sensors.smoothed <- sensors.max[,.(
    goal.current = goal.current,
    seq = seq,
    timesteps = timesteps,
    sensor = filter(as.numeric(sensor), rep(1,window)/window))]

# PLOTS ------------------------------------------------------------------------



gp <- ggplot(sensors.smoothed,
             aes(
                 y = sensor,
                 x = timesteps,
                 group = factor(goal.current),
                 color = factor(goal.current)
             ))

gp <- gp + geom_line(size = 0.5, show.legend = FALSE)
print(gp)
