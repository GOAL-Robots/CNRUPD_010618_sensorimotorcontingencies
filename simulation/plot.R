require(reshape2)
require(ggplot2)
library(grid)
pkg = "gridExtra"
if(!require(pkg, character.only=TRUE) )
{ 
    install.packages(pkg, repos = "http://cran.us.r-project.org")
}
library(gridExtra)

parameters = scan("src/model/parameters.py", character(0), sep = "\n" )
N_GOALS = strtoi(sub("(.*)=\\s*([0-9]+)\\s+$","\\2", parameters[grepl("^GOAL_NUMBER =", parameters)]))

predictions = read.table("test/log_predictions")
pred_names = paste("G",1:N_GOALS, sep="")
names(predictions) = c("TIMESTEPS",pred_names ,"CURR_GOAL")
predictions = melt(predictions, id.vars = c('TIMESTEPS',"CURR_GOAL"), measure.vars = pred_names, variable.name = "GOAL", value.name = "PREDICTION")
    
goals = predictions[,c(1,2)]
names(goals) = c("ts","goal")
goals$value=1
goals$variable=""
goals$goal = (goals$goal/N_GOALS)*0.3 +1.2
goals=data.frame(goals)

pdf("plot.pdf", width=5.0, heigh = 3.0, pointsize=11)
gp = ggplot(predictions, aes(x=TIMESTEPS, y=PREDICTION, group=GOAL, color=GOAL)) 
#gp = gp + geom_point(size=.01, stroke=0)
gp = gp + geom_line(size=.5)
# gp = gp + geom_raster(aes(fill=PREDICTION))
gp = gp + geom_point(data = goals, aes(x=ts, y=goal, guide="none"), color="black",size = 0.00001, inherit.aes=FALSE )
gp = gp + labs(list(x = "timesteps", y = "prediction"))
print(gp)
dev.off()





