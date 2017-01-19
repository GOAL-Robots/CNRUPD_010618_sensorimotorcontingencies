require(reshape2)
require(ggplot2)
library(grid)
library(gridExtra)
pkg = "gridExtra"
if(!require(pkg, character.only=TRUE) )
{ 
    install.packages(pkg, repos = "http://cran.us.r-project.org")
}

d =read.table("test/log_predictions")
dd = melt(d, id.vars=c(1,11), measure.vars=2:10 )
    
goals = dd[,c(1,2)]
names(goals) = c("ts","goal")
goals$value=1
goals$variable=""
goals$goal = goals$goal/20+1.2
goals=data.frame(goals)

dd$goal = dd$variable
levels(dd$goal) = paste("G", 1:9, sep="")
pdf("plot.pdf", width=5.0, heigh = 3.0, pointsize=11)
gp = ggplot(dd, aes(x=V1, y=value, group=goal, color=goal)) + geom_line(size=.5)
gp = gp + geom_point(data = goals, aes(x=ts, y=goal, guide="none"), color="black",size = 0.00001, inherit.aes=FALSE )
gp = gp + labs(list(x = "timesteps", y = "prediction"))
print(gp)
dev.off()




