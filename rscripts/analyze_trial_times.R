rm(list=ls())

toInstall <- c("extrafont", "ggplot2", 
               "data.table", "cowplot", 
               "grid", "gridExtra")

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

N_GOALS=25
all_trials <- fread("all_trials")
names(all_trials) <- c("LEARNING_TYPE", "INDEX","TIMESTEPS","GOAL")

TS = max(all_trials$TIMESTEPS)
N = length(all_trials$TIMESTEPS)
attach(all_trials)
all_trials$TRIAL_TIME = c(TIMESTEPS[1], TIMESTEPS[2:N]-TIMESTEPS[1:(N-1)])
detach(all_trials)

gg_m = list()
for (g in 1:N_GOALS)
{
  g_trials = subset(all_trials, GOAL == g-1)
  f = filter(g_trials$TRIAL_TIME, rep(1, 70)/70)
  f = f[!is.na(f)]
  gg_m[[g]] = f
  cat(g_trials$TRIAL_TIME)
}

dev.new()
plot(1e10,1e10,xlim=c(0,400), ylim=c(0,100))

for(gl in  gg_m)
{
  lines(gl)
}

# gp = ggplot(g_trials, aes(x=IDX, y=TRIAL_TIME) )
# gp = gp + geom_line()
# print(gp)
            
# 
# attach(all_trials)
# TRIAL_TIME = c(TIMESTEPS[1], TIMESTEPS[2:N]-TIMESTEPS[1:(N-1)])
# detach(all_trials)
# all_trials$TRIAL_TIME = TRIAL_TIME
# gp = ggplot(subset(all_trials, TIMESTEPS>15000), 
#             aes(x=TIMESTEPS, 
#                 y=TRIAL_TIME+GOAL*200,
#                 label=factor(GOAL),
#                 group=factor(GOAL),
#                 color=factor(GOAL)))
# gp = gp + geom_line(show.legend = FALSE)
# gp = gp + geom_text(aes(x=10,y=GOAL*200 ), show.legend = FALSE)
# print(gp)
