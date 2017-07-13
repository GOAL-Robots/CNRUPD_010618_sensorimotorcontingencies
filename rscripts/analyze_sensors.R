require(data.table)
require(ggplot2)

toInstall <- c("extrafont")
for(pkg in toInstall)
    if(!require(pkg, character.only=TRUE) )
    {
        install.packages(pkg, repos = "http://cran.us.r-project.org")
        library(extrafont)
        font_import()
        loadfonts()
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
TH_TIMESTEPS = max(sensors$TIMESTEPS)-50000
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

    gp1 = ggplot(subset(means, INDEX==idx ), aes(x = sensor, y = a_mean, group = LEARNING_TYPE))
    gp1 = gp1 + geom_ribbon(aes(ymin = 0, ymax = a_mean + a_sd), colour = "#666666", fill = "#dddddd")
    gp1 = gp1 + geom_line(size = 1.5, colour = "#000000")
    gp1 = gp1 + geom_bar(aes(y=5*a_count/count_tot),stat="identity", alpha=.2)
    gp1 = gp1 + xlab("Sensors")
    gp1 = gp1 + ylab("Means of sensor activation")
    gp1 = gp1 + scale_y_continuous(sec.axis = sec_axis(trans = ~./5, name = "Proportion of touches"))

    gp1 = gp1 + theme_bw() 
    gp1 = gp1 + theme( 
                    text=element_text(size=11, family="Verdana"), 
                    axis.text.x = element_text(size=8, angle = 90, hjust = 1),
                    axis.text.y = element_text(size=8),
                    panel.border=element_blank(),
                    legend.title = element_blank(),
                    legend.background = element_blank(),
                    panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank()
                    )
    
    basename = paste("gs_means",format(idx),sep="")
    pdf(paste(
        basename,".pdf",sep=""), 
        width = 7, 
        height = 3, 
        family="Verdana")
    print(gp1)
    dev.off()
    png(paste(
        basename,".png",sep=""),  
        width = 700, 
        height = 300, 
        family = "Verdana")
    print(gp1)
    dev.off()
    
  
    gp2 = ggplot(subset(means_goal, INDEX==idx & LEARNING_TYPE=="SIM"), aes(x = sensor, y = a_mean))
    gp2 = gp2 + geom_bar(aes(y=a_count),stat="identity")    
    gp2 = gp2 + facet_grid(CURR_GOAL_ORDERED~.)
    gp2 = gp2 + scale_y_continuous(limits=c(0, 90), breaks= c(0, 60),
                                   sec.axis = sec_axis(trans = ~., breaks=c(), 
                                                       name = "Goals"))
    gp2 = gp2 + xlab("Sensors") 
    gp2 = gp2 + ylab("Number of touches") 
    gp2 = gp2 + theme_bw() 
    gp2 = gp2 + theme( 
                    text=element_text(size=11, family="Verdana"), 
                    panel.border=element_blank(),
                    axis.text.x = element_text(size=8, angle = 90, hjust = 1),
                    axis.text.y = element_text(size=8),
                    strip.text.y = element_text(size=8, angle = 0, face = "bold"),
                    strip.background = element_rect(colour="#FFFFFF", fill="#FFFFFF"),
                    legend.title = element_blank(),
                    legend.background = element_blank(),
                    panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank()
                    )
    
    basename = paste("gs_means_goal",format(idx),sep="")
    pdf(paste(
        basename,".pdf",sep=""), 
        width = 7, 
        height = 7, 
        family="Verdana")
    print(gp2)
    dev.off()
    png(paste(
        basename,".png",sep=""),  
        width = 700, 
        height = 700, 
        family = "Verdana")
    print(gp2)
    dev.off()
    
}
