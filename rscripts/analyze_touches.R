require(data.table)
require(ggplot2)

d = fread("log_cont_sensors")
n_sensors=dim(d)[2] - 2
names(d) = c('ts', paste("S",1:n_sensors,sep=""),"goal")
dd = melt(d, id.vars=c("ts","goal"), variable.name="sensor", value.name="touch")
dd = dd[,.(touch=sum(touch>0.1)), by = .(sensor)]

gp = ggplot(dd, aes(x=sensor, y=touch)) + geom_bar(stat="identity")
ggsave("touches.pdf")
