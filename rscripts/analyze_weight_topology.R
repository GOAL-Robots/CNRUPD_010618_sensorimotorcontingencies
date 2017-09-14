rm(list=ls())

toInstall <- c("extrafont", "ggplot2", "data.table", "cowplot", "grid", "gridExtra")

for(pkg in toInstall)
{
    if(!require(pkg, character.only=TRUE) )
    {
        install.packages(pkg, repos = "http://cran.us.r-project.org")
    }
}

if (!("Verdana" %in% fonts()) )
{
    font_import()
    loadfonts()
}

###############################################################################################################################

TS_ALL = 250e+3
TS_GAP =  20e+3

all_weights <- fread("all_weights")
names(all_weights) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS",
                            "kohonen", "echo")

all_weights = subset(all_weights, TIMESTEPS <= TS_ALL)

gp1 = ggplot(all_weights, aes(x = TIMESTEPS))
gp1 = gp1 + geom_line(aes(y=kohonen))
gp1 = gp1 + xlab("Timesteps") 
gp1 = gp1 + ylab("") 
gp1 = gp1 + theme_bw() 

gp1 = gp1 + theme(
    text=element_text(size=11, family="Verdana"),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)

gp2 = ggplot(all_weights, aes(x = TIMESTEPS))
gp2 = gp2 + geom_line(aes(y=echo) )
gp2 = gp2 + xlab("Timesteps") 
gp2 = gp2 + ylab("") 
gp2 = gp2 + theme_bw() 

gp2 = gp2 + theme(
    text=element_text(
        size=11, family="Verdana"),
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)

pdf("weights.pdf", width = 6, height = 3)
grid.arrange(gp1, gp2, nrow=2, ncol=1)

dev.off()

###############################################################################################################################

weights <- fread("weights")
rows = dim(weights)[1]
rgoals = sqrt(rows)
cols = dim(weights)[2]
rretina = sqrt(cols)

weights$goal=factor(1:rows)
weights = melt(weights, variable.name= "cell", 
               variable.factor = TRUE,
               measure.vars = 1:cols)
weights$cell = factor(weights$cell)



attach(weights)
distances = by(value, goal, 
               function(k) by(value, goal, 
                              function(h) dist(rbind(k,h)) ))
detach(weights)

distances = do.call(rbind, distances) 
distances = distances/max(distances)
distances = distances*upper.tri(distances)

dist2d<-function(d1, d2, r)
{
  x1 = d1 %% r
  y1 =  d1 %/% r
  x2 = d2 %% r
  y2 =  d2 %/% r  
  sqrt((x2-x1)**2 + (y2-y1)**2)
}

idx_distances=matrix(rep(0,rows*rows), rows, rows)
for (x in 1:rows) for (y in 1:rows ) idx_distances[x,y]=dist2d(x-1, y-1, rgoals)


melt_4<-function(x, r)
{
  
  melted = melt(x)
  d = data.frame(value = melted$value)
  d$x1 = (melted$Var1-1)%%r
  d$y1 = (melted$Var1-1)%/%r
  d$x2 = (melted$Var2-1)%%r
  d$y2 = (melted$Var2-1)%/%r
  
  d
}

grid_goals= melt_4(distances, rgoals)
grid_dists= melt_4(idx_distances, rgoals)
grid_goals$dist=grid_dists$value

grid_goals_nearby = subset(grid_goals, dist <= 1.5)
gp = ggplot(grid_goals_nearby, aes(x=x1, y=y1, xend=x2,yend=y2, 
                                   colour=1-value)) 
gp = gp+ geom_segment(show.legend = FALSE) 
gp = gp + scale_colour_continuous(limits=c(0,1), low = "#ffffff", high = "#000000")
print(gp)