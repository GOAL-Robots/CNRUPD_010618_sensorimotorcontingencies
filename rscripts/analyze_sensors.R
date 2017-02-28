require(data.table)
require(ggplot2)

toInstall <- c("extrafont")
for(pkg in toInstall)
    if(!require(pkg, character.only=TRUE) )
    {
        install.packages(pkg, repos = "http://cran.us.r-project.org")
        library(extrafont)
        font_import()
    }

library(extrafont)

###############################################################################################################################

sem<-function(x) sd(x)/sqrt(length(x))

###############################################################################################################################

LASTIMESTEPS = 50000

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

TH_PREDICTION = 1.0
sensors = subset(sensors, prediction >= TH_PREDICTION)

sensors = melt(sensors, 
             id.vars = c("LEARNING_TYPE", "INDEX", "TIMESTEPS", "CURR_GOAL"), 
             measure.vars = sens_labels, 
             variable.name="sensor", 
             value.name="amp" )

sensors_last = subset(sensors, TIMESTEPS > (max(TIMESTEPS)-LASTIMESTEPS))

means = sensors_last[,.(a_mean = mean(amp), 
                         a_count = sum(amp>0.2),  
                         a_sd = sd(amp),  
                         a_err = sem(amp), 
                         a_min = min(amp),
                         a_max = max(amp)), by=.(LEARNING_TYPE, INDEX, sensor)]


count_tot = sum(means$a_count)

for(idx in unique(means$INDEX)) 
{ 
    pdf(paste("g_means",format(idx),".pdf",sep=""))
    gp = ggplot(subset(means, INDEX==idx ), aes(x = sensor, y = a_mean, group = LEARNING_TYPE))
    gp = gp + geom_ribbon(aes(ymin = a_mean - a_sd, ymax = a_mean + a_sd), colour = "#666666", fill = "#dddddd")
    gp = gp + geom_line(size = 1.5, colour = "#000000")
    gp = gp + geom_bar(aes(y=a_count/count_tot),stat="identity", alpha=.3)
    gp = gp + theme_bw() 
    gp = gp + facet_grid(LEARNING_TYPE~.)
    gp = gp + theme( 
                    text=element_text(size=14, family="Verdana"), 
                    panel.border=element_blank(),
                    legend.title = element_blank(),
                    legend.background = element_blank(),
                    panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank()
                    )
    print(gp)
    dev.off()
}
