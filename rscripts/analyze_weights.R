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

TS_ALL = 1000e+3
TS_GAP =   20e+3

all_weights <- fread("all_weights")
names(all_weights) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS",
                            "kohonen", "echo")

all_weights = subset(all_weights, TIMESTEPS <= TS_ALL)
TS = max(all_weights$TIMESTEPS)
TS_ALL=TS

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

weights$goal=1:rows
weights = melt(weights, variable.name="cell", 
               measure.vars = 1:cols)

weights$cell = as.numeric(sub("V","",weights$cell)) -1
weights$r_row = weights$cell%/%rretina + 1
weights$r_col = weights$cell%%rretina + 1

grbs = list()

for(row in 1:rows)
{
    cat(paste(row,"\n"))
    w = subset(weights, goal == row)
    gp = ggplot(w, aes(x = r_col, y = r_row, fill = value))
    gp = gp + geom_raster( show.legend = FALSE)
    gp = gp + xlab("")
    gp = gp + ylab("")
    gp = gp + theme_bw()
    gp = gp + scale_fill_gradient(low = "#ffffff", high = "#000000")
    gp = gp + theme( 
        text=element_text(size=11, family="Verdana"), 
        axis.ticks = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(), 
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        #panel.border = element_blank(),
        legend.title = element_blank(),
        legend.background = element_blank(),
        panel.grid.major = element_blank(),
        plot.margin = unit(c(.1,.1,.0,.0), "in"),
        panel.grid.minor = element_blank()
    )
    grbs[[row]] = ggplotGrob(gp)

}

pdf("weights_grid.pdf", width=6, height=6*0.55)
p = grid.arrange(grobs = grbs, nrow=5, ncol=5, 
             layout_matrix=matrix(1:rows,rgoals,rgoals))
dev.off()

# ###############################################################################################################################

positions <- fread("positions")
names(positions) <- c("goal", "x", "y")
positions$order <- rep(1:8, rows)
grbs = list()
for (row in 1:rows)
{
    w = subset(positions, goal == row-1)

    print(w)
    
    gp = ggplot(w, aes(x = as.numeric(x), y = as.numeric(y), order = order))
    gp = gp + geom_path(size = 1.5, color="#555555")
    gp = gp + geom_point()
    gp = gp + xlab("")
    gp = gp + ylab("")
    gp = gp + theme_bw()
    gp = gp + scale_x_continuous(limits = c(-3.5,3.5))
    gp = gp + scale_y_continuous(limits = c(-0.1,3))
    gp = gp + coord_fixed()
    gp = gp + theme(
        text = element_text(size = 11, family = "Verdana"),
        axis.ticks = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        #panel.border = element_blank(),
        legend.title = element_blank(),
        legend.background = element_blank(),
        panel.grid.major = element_blank(),
        plot.margin = unit(c(.0, .0, .0, .0), "in"),
        panel.grid.minor = element_blank()
    )
    grbs[[row]] = ggplotGrob(gp)
}

pdf("positions_grid.pdf", width=6, height=6*0.5571429)
p = grid.arrange(grobs = grbs, nrow=5, ncol=5, 
                 layout_matrix=matrix(1:rows,rgoals,rgoals))
dev.off()
