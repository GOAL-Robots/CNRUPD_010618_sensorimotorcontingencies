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

offline.plot <- FALSE

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

    if(length(idx) > 0) {
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

    if(length(idx) > 0) {
        res[idx] = 1
    }

    res
}

# PREDICTIONS ------------------------------------------------------------------
# __ load prediction dataset ====
predictions <- fread("all_predictions")
goals.number <- dim(predictions)[2] - 4 # first 4 columns are not goals
goals.labels <- paste("goal.", 1:goals.number, sep = "")
names(predictions) <- c("learning.type",
                       "index",
                       "timesteps",
                       goals.labels,
                       "goal.current")

# __ melt by goal columns making a goal factor ====
predictions <- melt(predictions,
                    id.vars = c("learning.type",
                                "index",
                                "timesteps",
                                "goal.current"),
                    measure.vars = goals.labels,
                    variable.name = "goal",
                    value.name = "prediction" )

# convert goal into a factor
predictions$goal = factor(predictions$goal)

# convert goal.current into a factor
predictions$goal.current <-
    factor(predictions$goal.current,
           labels = levels(predictions$goal))

# select only current goals
predictions <- subset(predictions, goal == goal.current)

# maintain only needed variables
predictions <- predictions[, .(timesteps,
                              goal.current,
                              prediction) ]

# SENSORS ----------------------------------------------------------------------

# __ load prediction dataset ====
sensors <- fread('all_sensors')
sensors.number <- dim(sensors)[2] - 4 # first 4 columns are not sensors
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
sensors = melt(sensors,
               id.vars = c("learning.type",
                           "index",
                           "timesteps",
                           "goal.current",
                           "prediction"),
               measure.vars = sensors.labels,
               variable.name = "sensor",
               value.name = "activation")

# maintain only needed variables
sensors <- sensors[, .(timesteps,
                       goal.current,
                       prediction,
                       sensor,
                       activation) ]

# select only the sensor with maximal activity for
#   each goal.current x prediction x timesteps
sensors <- sensors[,
                   .(activation = max(activation),
                     sensor = sensor[activation == max(activation)]),
                   by = .(goal.current, prediction, timesteps)]

# add info abotut the init of maximal stable prediction
#   (moment of end of learning)
sensors <- sensors[,
                   .(timesteps,
                     goal = goal.current,
                     prediction,
                     sensor,
                     prediction.switch = first.occurr(prediction,1)),
                   by = .(goal.current)]

# select rows where switch takes place
sensors.switch <- subset(sensors, prediction.switch == 1)

# TOUCHES ----------------------------------------------------------------------

# __ load touches dataset ====
touches <- fread("log_cont_sensors")
touches_sensors.number <- dim(touches)[2] - 2 # first 2 columns are not sensors
names(touches) <- c('timesteps',
                   paste("sensor.",
                         1:touches_sensors.number,
                         sep = ""),
                   "goal")

# __ melt by sensor columns making a sensor factor ====
touches <- melt(touches,
               id.vars = c("timesteps", "goal"),
               variable.name = "sensor",
               value.name = "touch.count")



# INNER JOIN sensors x touches -------------------------------------------------

# __ set the keys ====
setkey(sensors, timesteps, goal)
setkey(touches, timesteps, goal)

# __ inner join ====
touches.sensors <- touches[subset(sensors,
                                  prediction.switch == 1),
                           nomatch = 0]


gp <- ggplot(touches.sensors,
            aes(x = sensor,
                y = touch.count + goal*3,
                group = goal))
gp <- gp + geom_line()
print(gp)

gp <- ggplot(sensors.switch,
             aes(x = as.numeric(sensor),
                 y = order(timesteps)))
gp <- gp + geom_bar(stat = "identity")
print(gp)

