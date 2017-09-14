toInstall <- c("extrafont", "ggplot2", "data.table", "cowplot")
for(pkg in toInstall)
{
    if(!require(pkg, character.only=TRUE) )
    {
        install.packages(pkg, repos = "http://cran.us.r-project.org")
    }
}

require(data.table)
require(ggplot2)
require(cowplot)
library(extrafont)

if (!("Verdana" %in% fonts()) )
{
    font_import()
    loadfonts()
}
###############################################################################################################################

sem<-function(x) sd(x)/sqrt(length(x))

###############################################################################################################################

LASTIMESTEPS = 000
AMP_TH = 0.1

predictions <- fread("all_predictions")
N_GOALS = dim(predictions)[2] - 4 
names(predictions) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", paste("G", 1:N_GOALS, sep=""),"CURR_GOAL")

goal_labels = paste("G", 1:N_GOALS, sep = "")
gpredictions = melt(predictions, 
             id.vars = c("LEARNING_TYPE",  "INDEX", "TIMESTEPS", "CURR_GOAL"), 
             measure.vars = goal_labels, 
             variable.name="GOAL", 
             value.name="prediction" )
gpredictions$CGOAL = paste("G", gpredictions$CURR_GOAL+1, sep="")
gpredictions$CCGOAL = gpredictions$CGOAL == gpredictions$GOAL
gpredictions = subset(gpredictions, CCGOAL==TRUE)
gpredictions = gpredictions[with(gpredictions, order(LEARNING_TYPE,INDEX,TIMESTEPS)),]

sensors = fread('all_sensors')
N_SENSORS = (dim(sensors)[2] - 4)
sens_labels = paste("S", 1:N_SENSORS, sep = "")
names(sensors) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", sens_labels, "CURR_GOAL")
sensors = sensors[with(sensors, order(LEARNING_TYPE,INDEX,TIMESTEPS)),]
sensors$prediction =  gpredictions$prediction

sensors = melt(sensors, 
             id.vars = c("LEARNING_TYPE", "INDEX", "TIMESTEPS", "CURR_GOAL"), 
             measure.vars = sens_labels, 
             variable.name="sensor", 
             value.name="amp" )

TS = max(sensors$TIMESTEPS)
TS_TICS = as.integer(floor(seq(1, TS, length.out = 10)))

for(tic in 2:10)
{
  curr_sensors = subset(sensors, TIMESTEPS < TS_TICS[tic] )
  amps = curr_sensors[, .(mamp = mean(amp),
                     s = sensor[amp == max(amp)]),
                 by = .(CURR_GOAL)]
  amps = amps[order(s)]
  gp = ggplot(amps, aes(x = as.numeric(s), y = mamp))
  gp = gp + geom_bar(stat = "identity")
  gp = gp + scale_y_continuous(limits=c(0,0.3))
  print(gp)
}
