pre <- Sys.time()

library(NADIA) # this is a package that introduces KNN imputation for missing values to mlr3
library(mlr3verse) # this is a package available on cran
library(mlr3proba) # this is only available on github remotes::install_github("mlr-org/mlr3proba")
library(mlr3extralearners) # this is only available on github https://github.com/mlr-org/mlr3extralearners

mcRep <- 1 # defines the number of reps for the monte-carlo process. Currently set at 1.

mcPerf <- data.frame(calib = numeric()) # defines the dataframe to store the calibration for each rep of the monte-carlo process.

target <- 'last_DX' # define target and time variables
time <- 'last_visit'

dat <- read.csv('data/cn_preprocessed_wo_csf.csv') # reads the cn data into R.

dat$X <- NULL # removes artifact

dat$last_DX <- ifelse(dat$last_DX == 'CN', 0, 1) # turns outcome numeric

table(dat$last_DX) #sanity check

learner = lrn("surv.coxph") # define learner as cox PH model.

knn_impute  = NADIA::PipeOpVIM_kNN$new(k = 5) # defining the KNN imputation process for missing values, with K = 5.

task = as_task_surv(dat, time = time, event = target) # creating an MLR3 friendly version of the data and explicitly denoting the time and event variables.

task$col_roles$stratum = task$target_names # indentifying the target values to be used for stratification of the data amongst splits.

grn <- as_learner(knn_impute %>>% learner) # we incorporate the KNN impute process
# alongside the coxph learner as a learner object, such that they will be appied to every split of the outer loop, on the train dataset.

outer_resampling = rsmp("cv", folds = 5) # we now define the outer loop.
# ncol(dat)/3 was taking too long so I put it at 5, which did not adversely affect the stability of the models. See MC results.


# the following for loop runs the nested cross-validation and stores the resulting Calibration, before binding it to mcPerf.
for (i in 1:mcRep) {

  print(i)

  outer_resampling$instantiate(task) # applies the outer loop splits to the task, which takes into account stratification


  rr = resample(task, grn, outer_resampling, store_models = F)

  V = as.vector(rr$aggregate(msr("surv.calib_alpha")))

  mcPerf <- as.data.frame(rbind(mcPerf, V))
  #rm(rr)

}

post <- Sys.time()



