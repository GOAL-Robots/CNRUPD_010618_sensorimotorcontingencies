toInstall <- c("extrafont", "ggplot2", "data.table", "cowplot", "grid", "gridExtra")
for(pkg in toInstall)
{
    if(!require(pkg, character.only=TRUE) )
    {
        install.packages(pkg, repos = "http://cran.us.r-project.org")
    }
}

require(data.table)
require(ggplot2)
require(gridExtra)
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

which_dec_scale <- function(x)
{
    scales = 10^seq(1,10)
    x_scaled = x/scales
    test_scales = x_scaled <10 & x_scaled >1 
    
    scales[which(test_scales == TRUE)]
}


all_predictions <- fread("all_predictions")
N_GOALS = dim(all_predictions)[2] - 4 
names(all_predictions) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", paste("G", 1:N_GOALS, sep=""),"CURR_GOAL")

all_weights <- fread("all_weights")
names(all_weights) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS", "KOHONEN", "ECHO")
TS = max(all_weights$TIMESTEPS)
scale = which_dec_scale(TS)
trials = 1:length(all_weights$TIMESTEPS)
tlbrk = trials[trials%%(scale/200) == 0]
tsbrk = all_weights$TIMESTEPS[tlbrk]
TSS = 1:TS
tslbrk = TSS[TSS%%(scale)==0]

predictions = melt(all_predictions, 
             id.vars = c("LEARNING_TYPE", "TIMESTEPS", "INDEX"), 
             measure.vars = paste("G", 1:N_GOALS, sep = ""), 
             variable.name="GOAL", 
             value.name="prediction" )

ts = all_predictions$TIMESTEPS
l = as.integer(length(ts)*(3/20))
ts = ts[seq(1, l, length.out = 5)]
m = subset(predictions, TIMESTEPS %in% ts)
m$r=(strtoi(sub("G","", m$GOAL))-1)%%5
m$c=(strtoi(sub("G","", m$GOAL))-1)%/%5

gps = list()

for(row in 1:length(ts))
{
    data = subset(m, TIMESTEPS == ts[row])
    gp = ggplot(data)
    gp = gp + geom_rect(
        aes(
            xmin = c - 0.4 * prediction,
            xmax = c + 0.4 * prediction,
            ymin = r - 0.4 * prediction,
            ymax = r + 0.4 * prediction),
        fill="black"
        )
    gp = gp + geom_text(aes(x = 1, y = 4.8,
                             label = paste("Timestep:",ts[row])),   
                         size = 1.5,
                         inherit.aes = FALSE)
    gp = gp + scale_x_continuous(limits = c(-1,5), breaks=0:4)
    gp = gp + scale_y_continuous(limits = c(-1,5), breaks=0:4)
    gp = gp + xlab("")
    gp = gp + ylab("")
    gp = gp + theme_bw()

    gp = gp + theme( 
        text=element_text(size=11, family="Verdana"), 
        legend.title = element_blank(),
        legend.background = element_blank(),
        panel.grid.major = element_blank(),
        plot.margin = unit(c(.0,.0,.0,.0), "in"),
        panel.grid.minor = element_blank()
    )
        

    gps[[row]] = ggplotGrob(gp)
}

pdf("pred_hist.pdf", width=6, height=1.2)
grid.arrange(grobs = gps, nrow=1, ncol=5)
dev.off()