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

touches <- fread("log_cont_sensors")
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

touches <- touches[, .(touch = sum(touch > 0.1)),
                   by = .(sensor)]

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

sensors.max  <-
    sensors[, .(sensor = sensor[activation == max(activation)],
                timesteps = timesteps[activation == max(activation)],
                prediction = prediction[activation == max(activation)],
                activation = activation[activation == max(activation)]),
            by = .(goal.current, seq)]

sensors.max <- sensors.max[activation != 0]

sensors.max  <- sensors.max[, .(seq = order(seq),
                                sensor = sensor,
                                timesteps = timesteps,
                                prediction = prediction,
                                activation = activation),
                            by = .(goal.current)]


sensors.smoothed <- sensors.max[,.(
    goal.current = goal.current,
    seq = seq,
    timesteps = timesteps,
    sensor = filter(as.numeric(sensor), rep(1,200)/200))]

# PLOTS ------------------------------------------------------------------------


# # __ Plot the sequences of max match touches for each goal ====
# gp <- ggplot(sensors.max,
#              aes(
#                  y = as.numeric(sensor),
#                  x = seq,
#                  group = factor(goal.current),
#                  color = factor(goal.current)
#              ))
#
# gp <- gp + geom_line(size = 0.5, show.legend = FALSE)
# gp <- gp + geom_path(
#     data = touches,
#     aes(x = touch / max(touch) * 32000 - 35000,
#         y = as.numeric(sensor)),
#     size = 4,
#     color = "#000000",
#     inherit.aes = FALSE
# )


gp <- ggplot(sensors.smoothed,
             aes(
                 y = sensor,
                 x = timesteps,
                 group = factor(goal.current),
                 color = factor(goal.current)
             ))

gp <- gp + geom_line(size = 0.5, show.legend = FALSE)
print(gp)
