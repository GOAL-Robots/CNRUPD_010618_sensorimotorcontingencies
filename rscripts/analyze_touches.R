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

d = fread("log_cont_sensors")
n_sensors=dim(d)[2] - 2
names(d) = c('ts', paste("S",1:n_sensors,sep=""),"goal")
dd = melt(d, id.vars=c("ts","goal"), variable.name="sensor", value.name="touch")
dd = dd[,.(touch=sum(touch>0.1)), by = .(sensor)]

gp = ggplot(dd, aes(x=sensor, y=touch)) + geom_bar(stat="identity")
ggsave("touches.pdf")
