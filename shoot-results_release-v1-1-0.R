library(tidyverse); library(runjags); library(coda); library(ggmcmc)

source("functions_release.R")
plot_trace <- F

## Load and format data ----
# Full data set, one row for each day's shooting
datFull <- read.csv("full_shoot_data.csv", header=T) %>%
  mutate(key = factor(key))

# Grouped by operation
datFullGp <- read.csv("grouped_shoot_data.csv", header=T) 

## Ivlev functional response with refuge parameter b ----
cat("model {
 # Priors
 alpha ~ dgamma(0.1, 0.1)
 beta ~ dgamma(0.01, 0.01)
 delta ~ dgamma(0.1, 0.1)
 shape ~ dgamma(0.01, 0.01)

 # Likelihood
 for(i in 1:nobs){
    y[i] ~ dgamma(shape, rate[i])
    # expected kills per hour
    mean[i] <- -beta + alpha * (1-exp(-(dhat[i] * delta)))
    rate[i] <- shape/mean[i]
    
 } #i
 
 # Predictive distribution
 for(j in 1:npred){
   predy[j] ~ dgamma(shape, pred_rate[j])
   pred_rate[j] <- shape/pred_mean[j]
   pred_mean[j] <- -beta + alpha * (1-exp(-(predx[j] * delta)))
 } #j

}", fill=TRUE, file="ivlev_b.txt")

predx <- seq(0.5,40,0.5)

fr_dat <- list(y = datFull$kills_hour,
               dhat = datFull$Dhat,
               nobs = nrow(datFull),
               predx = predx,
               npred = length(predx))

fr_pars <- c("alpha", "beta", "delta", "shape", "predy")

fr_inits <- function(){
  list(alpha = rnorm(1, 100, 20),
       beta = rnorm(1, 0.01, 0.01),
       delta = rnorm(1, 0.04, 0.01))
}

ni <- 20000
nc <- 7
nadapt <- 10000
nb <- 5000

set.seed(1080)
ivb_mod <- run.jags(method = "parallel",
                   model = "ivlev_b.txt",
                   monitor = fr_pars,
                   data = fr_dat,
                   inits = fr_inits(),
                   sample = ni,
                   n.chains = nc,
                   adapt = nadapt,
                   burnin = nb,
                   summarise = F,
                   plots = F,
                   silent.jags = F)

ivb_list <- as.mcmc.list(ivb_mod)
ivb_sum <- mod_results(ivb_list)
ivb_modmat <- ivb_sum$modMat

head(ivb_sum$sumTab) 

pred_a_ivb_mcmc <- ivb_sum$sumTab$Mean[which(ivb_sum$sumTab$Par == "alpha")]
pred_b_ivb_mcmc <- ivb_sum$sumTab$Mean[which(ivb_sum$sumTab$Par == "beta")]
pred_d_ivb_mcmc <- ivb_sum$sumTab$Mean[which(ivb_sum$sumTab$Par == "delta")]

pred_iv_fun <- function(a,b,d,x){
  return(-b + a*(1-exp(-(d * x))))
}

# MCMC uncertainty
pred_mean_dist_ivb <- matrix(NA, nrow = nrow(ivb_modmat), ncol = length(predx))
for (i in 1:nrow(pred_mean_dist_ivb)){
	pred_mean_dist_ivb[i,] <- pred_iv_fun(ivb_modmat[i,"alpha"], 
	                                      ivb_modmat[i,"beta"], 
	                                      ivb_modmat[i,"delta"], 
	                                      predx)
}

ivb_df <- data.frame(x = predx,
                     mean = apply(pred_mean_dist_ivb, MARGIN = 2, mean),
                     cri_lo = apply(pred_mean_dist_ivb, MARGIN = 2, quantile, prob = 0.025), # credible int
                     cri_up = apply(pred_mean_dist_ivb, MARGIN = 2, quantile, prob = 0.975),
                     pi_lo = apply(ivb_modmat[,6:ncol(ivb_modmat)], 2, quantile, prob = 0.025), # prediction int
                     pi_up = apply(ivb_modmat[,6:ncol(ivb_modmat)], 2, quantile, prob = 0.975))

## Setup Effort Outcomes data ----
dat_eo <- datFullGp %>%
  dplyr::select(key, species, dhat, nhat = init_nhat, se_nhat, deer_hours, hours_km, kills, kills_km) %>%
  mutate(hours_stnd = hours_km / dhat * 1000,
         mortality = kills / nhat,
         resid_dhat = dhat - (mortality*dhat),
         resid_nhat = nhat - (mortality*nhat)) %>%
  arrange(species, desc(deer_hours))

# add se for dhat
dhat_se <- data.frame(key = levels(factor(dat_eo$key))) %>%
  mutate(se_dhat = c(1.68, 5.68, 2.17, 5.13, 0.22, 11.34, 0.91, 3.35, 2.88, 2.01, 15.23, 10.58))

dat_eo <- dat_eo %>%
  left_join(dhat_se)

# Estimate posterior distribution of mortality for each site
# and derive residual density
cat("model {
 # Priors
 for(i in 1:ncase){
  alpha[i] ~ dnorm(0, 0.001) 
  n[i] ~ dpois(nhat[i])

  dhat1[i] ~ dnorm(dhat[i], tau_d[i])T(0, 100)
   tau_d[i] <- 1/(se_dhat[i]*se_dhat[i])
 } #i

 for(i in 1:ncase){
     y[i] ~ dbin(p[i], n[i]) 
     logit(p[i]) <- alpha[i]
  } #i
 
 # Derived residual density
 for(dd in 1:ncase){
    dhat2[dd] <- dhat1[dd] * (1-p[dd])
 } #dd

}", fill=TRUE, file="mortality.txt")


mort_pars <- c("n", "p", "dhat1", "dhat2")

mort_dat <- list(y = dat_eo$kills,
                 nhat = as.integer(dat_eo$nhat),
                 se_nhat = dat_eo$se_nhat,
                 ncase = nrow(dat_eo),
                 site = seq(1,nrow(dat_eo),1),
                 dhat = dat_eo$dhat,
                 se_dhat = dat_eo$se_dhat,
                 area = dat_eo$nhat/dat_eo$dhat) 
set.seed(1080)
mort_mod <- run.jags(method = "parallel",
                   model = "mortality.txt",
                   monitor = mort_pars,
                   data = mort_dat,
                   sample = 10000,
                   n.chains = nc,
                   adapt = nadapt,
                   burnin = nb,
                   summarise = F,
                   plots = F,
                   silent.jags = F)

mort_list <- as.mcmc.list(mort_mod)
mort_sum <- mod_results(mort_list)
mort_modmat <- mort_sum$modMat
head(mort_sum$sumTab)

# Proportion reduction credible intervals and sd into EO dataframe
dat_eo$prop_lwr <- mort_sum$sumTab$lcri[grep("p", mort_sum$sumTab$Par)]
dat_eo$prop_upr <- mort_sum$sumTab$ucri[grep("p", mort_sum$sumTab$Par)]
dat_eo$prop_se <- mort_sum$sumTab$SD[grep("p", mort_sum$sumTab$Par)]

# Initial denstiy (dhat) credible intervals into EO dataframe
dat_eo$dhat_lwr <- mort_sum$sumTab$lcri[grep("dhat1", mort_sum$sumTab$Par)]
dat_eo$dhat_upr <- mort_sum$sumTab$ucri[grep("dhat1", mort_sum$sumTab$Par)]

# Residual denstiy (dhat2) into EO dataframe
dat_eo$dhat2 <- mort_sum$sumTab$Mean[grep("dhat2", mort_sum$sumTab$Par)]
dat_eo$dhat2_lwr <- mort_sum$sumTab$lcri[grep("dhat2", mort_sum$sumTab$Par)]
dat_eo$dhat2_upr <- mort_sum$sumTab$ucri[grep("dhat2", mort_sum$sumTab$Par)]
dat_eo$dhat2_se <- mort_sum$sumTab$SD[grep("dhat2", mort_sum$sumTab$Par)]

# Initial density, mortality and residual density
dat_eo <- dat_eo %>%
  select(key, species, hours_stnd, dhat, se_dhat, 
         mortality, prop_lwr, prop_upr, 
         dhat2, dhat2_lwr, dhat2_upr)

## Michaelis-Menten function for effort:outcomes relationship ----

cat("model {
 # Priors
 alpha ~ dbeta(0.5, 0.5)
 delta ~ dunif(0, 100)
 phi ~ dunif(0, 10)

 # Likelihood
 for(i in 1:nobs){
    y[i] ~ dbeta(a[i], b[i])
    a[i] <- mu[i]*phi
    b[i] <- (1-mu[i])*phi
    # expected mortality at each level of standardised control effort (shoot hours / km2)/(nhat / km2 * 1000)
    mu[i] <- alpha * hours[i] / (hours[i] + delta)
 } #i
 
 # Predicted knockdown at different levels of effort
 for(j in 1:npred){
     predy[j] ~ dbeta(pred_a[j], pred_b[j])
     pred_a[j] <- pred_mu[j]*phi
     pred_b[j] <- (1-pred_mu[j])*phi
     # expected mortality at each level of standardised control effort (shoot hours / km2)/(nhat / km2 * 1000)
     pred_mu[j] <- alpha * predx[j] / (predx[j] + delta)
     } #j

}", fill=TRUE, file="mm_eo.txt")

predx_eo <- seq(0.001,60,length.out=60)

mm_eo_dat <- list(y = dat_eo$mortality,
                  hours = dat_eo$hours_stnd,
                  nobs = nrow(dat_eo),
                  predx = predx_eo,
                  npred = length(predx_eo))

eo_inits <- function(){
  list(alpha = runif(1, 0, 1),
       delta = rnorm(1, 30, 10))
}

set.seed(1080)

mm_eo_mod <- run.jags(method = "parallel",
                   model = "mm_eo.txt",
                   monitor = fr_pars,
                   data = mm_eo_dat,
                   inits = eo_inits(),
                   sample = ni,
                   n.chains = nc,
                   adapt = nadapt,
                   burnin = nb,
                   summarise = F,
                   plots = F,
                   silent.jags = F)

mm_eo_list <- as.mcmc.list(mm_eo_mod)
mm_eo_sum <- mod_results(mm_eo_list)
mm_eo_modmat <- mm_eo_sum$modMat
head(mm_eo_sum$sumTab)

if(plot_trace == T){
  s_mm <- mm_eo_sum$S
  ta <- ggs_traceplot(s_mm, family = "alpha")
  td <- ggs_traceplot(s_mm, family = "delta")
  grid.arrange(ta, td)
}

if(plot_trace == T){
  da <- ggs_density(s_mm, family = "alpha")
  dd <- ggs_density(s_mm, family = "delta")
  grid.arrange(da, dd)
}

pred_a_mm_eo_mcmc <- mm_eo_sum$sumTab$Mean[which(mm_eo_sum$sumTab$Par == "alpha")]
pred_d_mm_eo_mcmc <- mm_eo_sum$sumTab$Mean[which(mm_eo_sum$sumTab$Par == "delta")]
pred_b_mm_eo_mcmc <- 0

# Credible and Posterior Predictive distribution
pred_fun_mm <- function(a,b,d,x){return(-b + a*(x/(x+d)))}

pred_mean_dist_mm <- matrix(NA, nrow = nrow(mm_eo_modmat), ncol = length(predx_eo))
for (i in 1:nrow(pred_mean_dist_mm)){
	pred_mean_dist_mm[i,] <- pred_fun_mm(mm_eo_modmat[i,"alpha"], 
	                                     0,
	                                     mm_eo_modmat[i,"delta"],
	                                     predx_eo)
}

pred_eo_index <- grep("pred", names(mm_eo_modmat))
mm_eo_df <- data.frame(hours = predx_eo,
                       mean = apply(pred_mean_dist_mm, MARGIN = 2, mean),
                       mean_samp = apply(mm_eo_modmat[,pred_eo_index], MARGIN = 2, mean),
                       cri_lo = apply(pred_mean_dist_mm, MARGIN = 2, quantile, prob = 0.025), # credible int
                       cri_up = apply(pred_mean_dist_mm, MARGIN = 2, quantile, prob = 0.975),
                       pi_lo = apply(mm_eo_modmat[,pred_eo_index], 2, quantile, prob = 0.025), # prediction int
                       pi_up = apply(mm_eo_modmat[,pred_eo_index], 2, quantile, prob = 0.975))

## shots per deer ----
# Data from Hampton et al. (2021) Animal welfare outcomes of helicopter-based shooting of deer in Australia. Wildlife Research
shots_deer <- 4.14
shots_deer_sd <- 2.2

## fixed and variable costs ----
jet_hr <- 1534           # Jet Ranger hourly cost (wet)
sql_hr <- 2475           # Squirrel hourly cost (wet)
r44_hr <- 905
shtr_hr <- nav_hr <- 183 # Hourly cost for shooter and navigator
ammo <- 1.54             # Cost per round
deer_hr <- seq(0,100,1)  # range of deer per hour scenarios

costs <- data.frame(deer_hr = deer_hr,
                    jet = total_cost_hr(heli_hr = jet_hr)$hr_cost,
                    sql = total_cost_hr(heli_hr = sql_hr)$hr_cost,
                    r44 = total_cost_hr(heli_hr = r44_hr)$hr_cost) %>%
  pivot_longer(cols = c(jet, sql, r44),
               names_to = "heli", values_to = "cost") %>%
  arrange(heli)

## Total cost model ----
cat("model {
 # Priors
 alpha ~ dgamma(0.1, 0.1)
 beta ~ dgamma(0.01, 0.01)
 delta ~ dgamma(0.1, 0.1)
 shape ~ dgamma(0.01, 0.01)
 
 shots_deer ~ dpois(4.1) T(0,)

 # Likelihood, Ivlev functional response
 for(i in 1:nobs){
    y[i] ~ dgamma(shape, rate[i])
    # expected kills per hour at each density level
    mean[i] <- -beta + alpha * (1-exp(-(dhat[i] * delta)))
    rate[i] <- shape/mean[i]
 } #i
 
 # Predicted cost per hour for three helicopter types and 
 # population densities 0 to 40
 for(h in 1:nheli){ # Number of helicopters to assess cost over
  for(j in 1:npred){
   cost_hr[j,h] <- (predy[j,h]*shots_deer*cost_shot) + cost_heli[h] + 2*cost_staff
   predy[j,h] ~ dgamma(shape, pred_rate[j,h]) # deer per hour
   pred_rate[j,h] <- shape/pred_mean[j,h]
   pred_mean[j,h] <- -beta + alpha * (1-exp(-(predx[j] * delta)))
  } #j
 } #h

}", fill=TRUE, file="cost_hour.txt")

cost_hr_pars <- c("alpha", "beta",  "delta", "shape", "cost_hr")
cost_hr_dat <- list(y = fr_dat$y,
                    dhat = fr_dat$dhat,
                    nobs = fr_dat$nobs,
                    predx = fr_dat$predx,
                    npred = fr_dat$npred,
                    cost_heli = c(r44_hr, jet_hr, sql_hr),
                    nheli = 3,
                    cost_shot = 1.54,
                    cost_staff = 183)

set.seed(1080)
cost_hr_mod <- run.jags(method = "parallel",
                        model = "cost_hour.txt",
                        monitor = cost_hr_pars,
                        data = cost_hr_dat,
                        inits = fr_inits(),
                        sample = 10000,
                        n.chains = 4,
                        adapt = 5000,
                        burnin = nb,
                        summarise = F,
                        plots = F,
                        silent.jags = F)

cost_hr_list <- as.mcmc.list(cost_hr_mod)
cost_hr_sum <- mod_results(cost_hr_list)
cost_hr_modmat <- cost_hr_sum$modMat

## predicted cost per hour ----
r44_index <- grep(",1]", names(cost_hr_modmat))
jet_index <- grep(",2]", names(cost_hr_modmat))
sql_index <- grep(",3]", names(cost_hr_modmat))

# Posterior Predictive distribution
cost_hr_df <- data.frame(Helicopter = c(rep("R44", length(predx)),
                                        rep("Jet Ranger", length(predx)),
                                        rep("Squirrel", length(predx))),
                         x = rep(predx, 3),
                         mean = c(apply(cost_hr_modmat[,r44_index], MARGIN = 2, mean),
                                  apply(cost_hr_modmat[,jet_index], MARGIN = 2, mean),
                                  apply(cost_hr_modmat[,sql_index], MARGIN = 2, mean)),
                         pi_lo = c(apply(cost_hr_modmat[,r44_index], 2, quantile, prob = 0.025),
                                   apply(cost_hr_modmat[,jet_index], 2, quantile, prob = 0.025),
                                   apply(cost_hr_modmat[,sql_index], 2, quantile, prob = 0.025)),
                         pi_up = c(apply(cost_hr_modmat[,r44_index], 2, quantile, prob = 0.975),
                                   apply(cost_hr_modmat[,jet_index], 2, quantile, prob = 0.975),
                                   apply(cost_hr_modmat[,sql_index], 2, quantile, prob = 0.975))) %>%
  mutate(Helicopter = factor(Helicopter, levels=c("R44", "Jet Ranger", "Squirrel")))


## Effort needed for knockdown ----
# Express desired knockdown in terms of percentage alpha (alpha = pred_a_mm_eo_mcmc)
predkd  <- seq(0.05, 0.75, .01)
predkd_a <- predkd / pred_a_mm_eo_mcmc

kd <- 0.35
a <- pred_a_mm_eo_mcmc
d <- pred_d_mm_eo_mcmc

cat("model {
 # Priors
 alpha ~ dbeta(0.5,0.5)
 delta ~ dunif(0, 100)
 phi ~ dunif(0, 10)
 shape ~ dgamma(0.01, 0.01)

 # Effort:Outcomes
 for(i in 1:nobs){
    y[i] ~ dbeta(a[i],b[i])
    a[i] <- mu[i]*phi
    b[i] <- (1-mu[i])*phi
    # expected mortality at each level of standardised control effort (shoot hours / km2)/(nhat / km2 * 1000)
    mu[i] <- alpha * hours[i] / (hours[i] + delta)
    } #i
 
 # Predicted knockdown at different levels of effort
 for(j in 1:npred){
     predy[j] ~ dbeta(pred_a[j], pred_b[j])
     pred_a[j] <- pred_mu[j]*phi
     pred_b[j] <- (1-pred_mu[j])*phi
     # expected mortality at each level of standardised control effort (shoot hours / km2)/(nhat / km2 * 1000)
     pred_mu[j] <- alpha * predx[j] / (predx[j] + delta)
     } #j
    
 # Predicted effort to achieve knockdown[k]
 for(k in 1:nkd){
    pred_hrs[k] <- (kd[k]*delta)/(1-kd[k])
   } #k
   
}", fill=TRUE, file="kd_eo_beta.txt")

kd_eo_dat <- list(y = dat_eo$mortality,
                  hours = dat_eo$hours_stnd,
                  nobs = nrow(dat_eo),
                  predx = predx_eo,
                  npred = length(predx_eo),
                  kd = predkd_a,
                  nkd = length(predkd_a))

eo_inits <- function(){
  list(alpha = runif(1, 0, 1),
       beta = runif(1, 0.00001, 1),
       delta = rnorm(1, 30, 10))
}

kd_pars <- c("alpha", "beta", "delta", "shape", "predy", "pred_hrs")

set.seed(702)
kd_eo_mod <- run.jags(method = "parallel",
                      model = "kd_eo_beta.txt",
                      monitor = kd_pars,
                      data = kd_eo_dat,
                      inits = eo_inits(),
                      sample = 10000,
                      n.chains = 4,
                      adapt = nadapt,
                      burnin = nb,
                      summarise = F,
                      plots = F,
                      silent.jags = F)

kd_eo_list <- as.mcmc.list(kd_eo_mod)
kd_eo_sum <- mod_results(kd_eo_list)
kd_eo_modmat <- kd_eo_sum$modMat 

pred_a_kd_eo_mcmc <- kd_eo_sum$sumTab$Mean[which(kd_eo_sum$sumTab$Par == "alpha")]
pred_d_kd_eo_mcmc <- kd_eo_sum$sumTab$Mean[which(kd_eo_sum$sumTab$Par == "delta")]
pred_b_kd_eo_mcmc <- 0

# Posterior Predictive distribution of hours needed for knockdown[i]
# hours needed = (knockdown * delta) / (1-knockdown)
pred_kd_fun <- function(kd, delta){
  return(kd*delta)/(1-kd)
}
pred_kd_fun(kd=0.35, delta=pred_d_kd_eo_mcmc)
(pred_mean_kd <- pred_kd_fun(predkd_a, pred_d_kd_eo_mcmc))

pred_mean_dist_kd <- matrix(NA, nrow = nrow(kd_eo_modmat), ncol = length(predkd_a))
for (i in 1:nrow(pred_mean_dist_kd)){
	pred_mean_dist_kd[i,] <- (predkd_a * kd_eo_modmat[i,"delta"]) / (1 - kd_eo_modmat[i,"delta"])
}

hrs_modmat_indx <- grep("pred_hrs", colnames(kd_eo_modmat))
kd_modmat_indx <- grep("predy", colnames(kd_eo_modmat))

# Predicted knockdown, given hours
eo_df_kd <- data.frame(hours = predx_eo,
                       mean_kd = apply(kd_eo_modmat[,kd_modmat_indx], MARGIN = 2, mean),
                       pi_kd_lo = apply(kd_eo_modmat[,kd_modmat_indx], 2, quantile, prob = 0.025), 
                       pi_kd_up = apply(kd_eo_modmat[,kd_modmat_indx], 2, quantile, prob = 0.975))

# Predicted hours to achieve knockdown
eo_df_hrs <- data.frame(knockdown = predkd,
                       mean_hrs = apply(kd_eo_modmat[,hrs_modmat_indx], MARGIN = 2, mean),
                       pi_hrs_lo = apply(kd_eo_modmat[,hrs_modmat_indx], 2, quantile, prob = 0.025), 
                       pi_hrs_up = apply(kd_eo_modmat[,hrs_modmat_indx], 2, quantile, prob = 0.975))

interval_tab <- kd_eo_sum$sumTab %>%
  filter(grepl("predy", Par)) %>%
  mutate(hours = predx_eo,
         Mean = round(Mean, 2),
         lcri = round(lcri,2),
         ucri = round(ucri,2))

means <- ucri <- numeric(length(predkd))
for(i in 1:length(means)){
  means[i] <- median(interval_tab$hours[which(near(interval_tab$Mean, predkd[i], tol=0.05)==T)])
  ucri[i]  <- median(interval_tab$hours[which(near(interval_tab$ucri, predkd[i], tol=0.05)==T)])
}
pred_df <- data.frame(kd = predkd, mean_hrs = means, min_hrs = ucri)

## Combine effort with cost ----
jr_cost <- cost_hr_modmat %>%
  dplyr::select(contains(",2]"))

nsamp <- 50000
densities <- seq(0.5, 40, 0.5)
morts <- seq(0.25, 0.75, 0.10)
area <- 135
kd_modmat <- kd_eo_modmat %>%
  dplyr::select(contains("pred_hrs"))

# Make an empty data frame to hold output from nested loops
# Rows = samples nested in different mortality levels
# Columns = initial population density levels
cost_df <- data.frame(matrix(NA, ncol=length(densities)+1))
names(cost_df) <- c("Mortality", paste0("d", densities))

for(m in 1:length(morts)){
  # Data frame to hold output
  df_i <- data.frame(matrix(NA, ncol=length(densities)+1, nrow=nsamp))
  names(df_i) <- c("Mortality", paste0("d", densities))
  df_i$Mortality <- morts[m]
  
  for(d in 1:length(densities)){
    dhat <- densities[d]
    nhat <- dhat * area
    
    ## How many hours needed to reduce nhat by desired mortality
    des_mort <- morts[m]
  
    # Grab a sample of draws of hours for the desired mortality
    # these are samples of the number of hours per 1000 deer
  
    # Which column of the mcmc draws holds the desired mortality
    column_hours <- which(predkd == des_mort)
    row_hours <- round(runif(nsamp, 1, length(kd_eo_modmat)))
    hours_samp <- kd_modmat[row_hours, column_hours]
    # How many hours in total should be needed to achieve desired mortality
    hours_samp_tot <- hours_samp * nhat / 1000
  
    # Grab a matching sample from the cost per hour mcmc draws
    # Columns in jr_cost are costs per hour for a Jet Ranger at density[d]
    column_cost <- which(predx == dhat)
    row_cost <- round(runif(nsamp, 1, length(cost_hr_modmat)))
    cost_samp <- jr_cost[row_cost, column_cost]
  
    # Multiply hours needed by cost per hour
    tot_cost <- hours_samp_tot * cost_samp
    
    # Store in dataframe
    df_i[,d+1] <- round(tot_cost)
    } # d
  
    cost_df <- rbind(cost_df, df_i) %>%
      filter(is.na(Mortality)==F)
} #m

cost_representative <- cost_df %>%
  group_by(Mortality) %>%
  summarise(across(everything(), ~ mean(.x, na.rm = TRUE))) %>%
  pivot_longer(cols=-Mortality) %>%
  mutate(Density = as.numeric(gsub("d", "", name))) %>%
  dplyr::select(Mortality, Density, Cost=value)

lwr <- cost_df %>% 
  group_by(Mortality) %>%
  summarise(across(everything(), ~ quantile(.x, probs=0.025))) %>%
  pivot_longer(cols=-Mortality) %>%
  mutate(Density = as.numeric(gsub("d", "", name))) %>%
  dplyr::select(Mortality, Density, Cost_lwr=value)

upr <- cost_df %>% 
  group_by(Mortality) %>%
  summarise(across(everything(), ~ quantile(.x, probs=0.975))) %>%
  pivot_longer(cols=-Mortality) %>%
  mutate(Density = as.numeric(gsub("d", "", name))) %>%
  dplyr::select(Mortality, Density, Cost_upr=value)

# Summarise estimated cost operations over 135 km^2 (using Jet Ranger)
# as a function of initial population density and desired mortality

cost_representative <- cost_representative %>%
  mutate(cost_lwr = lwr$Cost_lwr,
         cost_upr = upr$Cost_upr)

