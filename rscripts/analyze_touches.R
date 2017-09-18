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

cont_sensors = fread("log_cont_sensors")
n_sensors=dim(cont_sensors)[2] - 2
names(cont_sensors) = c('ts', paste("S",1:n_sensors,sep=""),"goal")
melted_cont_sensors = melt(cont_sensors, id.vars=c("ts","goal"), variable.name="sensor", value.name="touch")
melted_cont_sensors = melted_cont_sensors[,.(touch=sum(touch>0.1)), by = .(sensor)]

gp = ggplot(melted_cont_sensors, aes(x=sensor, y=touch)) + geom_bar(stat="identity")
print(gp)
ggsave("touches.pdf")
