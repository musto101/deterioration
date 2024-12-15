#install.packages('../AdniDeterioration/', repos = NULL, type="source") # installing locally created packages
#install.packages('../RSurvivalML/RSurvivalML/', repos = NULL, type="source")

source('preprocessing_func/preprocessing.R')

#library(AdniDeterioration)
#library(RSurvivalML)
library(caret)
library(tidyverse) # loading libraries
#library(doParallel)

adni_slim2 <- read.csv('data/adni1_slim.csv') # loading in the adni2 data WITH csf predictors.

dat <- preprocessing(dat = adni_slim2, perc = 0.9, clinGroup = 'MCI') #isolating the MCI group and dummy coding categorical variables.

write.csv(dat, 'data/mci_preprocessed.csv') # writes the resultant data.frame to CSV.

dat <- preprocessing(dat = adni_slim2, perc = 0.9, clinGroup = 'CN') # Same process as above for the CN group.

write.csv(dat, 'data/cn_preprocessed.csv') # writes the resultant data.frame to CSV.
