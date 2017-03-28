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

TH_PREDICTION = 0.8
TH_TIMESTEPS = max(sensors$TIMESTEPS)*(7/8)
sensors = subset(sensors, prediction >= TH_PREDICTION)
sensors = subset(sensors, TIMESTEPS > TH_TIMESTEPS)
 
sensors = melt(sensors, 
             id.vars = c("LEARNING_TYPE", "INDEX", "TIMESTEPS", "CURR_GOAL"), 
             measure.vars = sens_labels, 
             variable.name="sensor", 
             value.name="amp" )


means = sensors[,.(a_mean = mean(amp), 
                         a_count = sum(amp>AMP_TH),  
                         a_sd = sd(amp),  
                         a_err = sem(amp)), by=.(LEARNING_TYPE, INDEX, sensor)]


means_goal = sensors[,.(a_mean = mean(amp), 
                         a_count = sum(amp>AMP_TH),  
                         a_sd = sd(amp),  
                         a_err = sem(amp)), by=.(LEARNING_TYPE, INDEX, sensor, CURR_GOAL)]


#---------------------------------------------------------------------
# find the sensor with maximum touch for each goal
maxes = means_goal[,.(mx=max(a_mean)), by=.(CURR_GOAL) ]
idcs = c()
for(mx in maxes$mx)  idcs = c(idcs, which(means_goal$a_mean==mx)[1])
maxes$sensor = means_goal[idcs]$sensor
maxes = maxes[order(maxes$sensor)]

means_goal$CURR_GOAL_ORDERED=factor(means_goal$CURR_GOAL, levels=maxes$CURR_GOAL)
#---------------------------------------------------------------------

count_tot = sum(means$a_count)

for(idx in unique(means$INDEX)) 
{ 
    pdf(paste("gs_means",format(idx),".pdf",sep=""))
    gp = ggplot(subset(means, INDEX==idx ), aes(x = sensor, y = a_mean, group = LEARNING_TYPE))
    gp = gp + geom_ribbon(aes(ymin = a_mean - a_sd, ymax = a_mean + a_sd), colour = "#666666", fill = "#dddddd")
    gp = gp + geom_line(size = 1.5, colour = "#000000")
    gp = gp + geom_bar(aes(y=a_count/count_tot),stat="identity", alpha=.3)
    gp = gp + theme_bw() 
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
  
    pdf(paste("gs_means_goal",format(idx),".pdf",sep=""))
    gp = ggplot(subset(means_goal, INDEX==idx & LEARNING_TYPE=="mixed-2"), aes(x = sensor, y = a_mean))
    gp = gp + geom_bar(aes(y=a_count/count_tot),stat="identity", alpha=.3)    
    gp = gp + facet_grid(CURR_GOAL_ORDERED~.)
    gp = gp + theme_bw() 
    gp = gp + theme( 
                    text=element_text(size=14, family="Verdana"), 
                    panel.border=element_blank(),
                    axis.text.x = element_text(angle = 90, hjust = 1),
                    legend.title = element_blank(),
                    legend.background = element_blank(),
                    panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank()
                    )

    print(gp)
    dev.off()
}
