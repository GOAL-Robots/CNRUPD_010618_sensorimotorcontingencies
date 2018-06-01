##
## Copyright (c) 2018 Francesco Mannella. 
## 
## This file is part of sensorimotor-contingencies
## (see https://github.com/GOAL-Robots/CNRUPD_010618_sensorimotorcontingencies).
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
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
