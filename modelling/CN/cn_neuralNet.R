pre <- Sys.time()
# Just a note to say I never managed to get this working locally. Neural nets in mlr3 rely on the reticulate package which I could never get working. But it works on Rdev on dsm2.
library(NADIA) # this is a package that introduces KNN imputation for missing values to mlr3
library(mlr3verse) # this is a package available on cran
library(mlr3proba) # this is only available on github remotes::install_github("mlr-org/mlr3proba")
library(mlr3extralearners) # this is only available on github https://github.com/mlr-org/mlr3extralearners

target <- 'last_DX' # define target and time variables
time <- 'last_visit'

dat <- read.csv('data/cn_preprocessed.csv') # reads the cn data into R.

dat$X <- NULL # removes artifact

dat$last_DX <- ifelse(dat$last_DX == 'CN', 0, 1) # turns outcome numeric

table(dat$last_DX) #sanity check

grid <- list(num_nodes = c(48, 56, 64),
             activation = c('relu'),
             epochs        = 20:25,
             batch_size = c(48, 56, 64), learning_rate = 0.001) # defines the hyperparameter search space.

learner = lrn("surv.deephit",
              num_nodes  = to_tune(grid$num_nodes),
              activation = to_tune(grid$activation),
              batch_size = to_tune(grid$batch_size),
              epochs = to_tune(grid$epochs),
              learning_rate = to_tune(grid$learning_rate),
              optimizer = 'adam',
              early_stopping = F) # survival neural network learner initialised with hyper param search space embedded and optimizer defined as stochastic gradient descent.

at = auto_tuner(method = tnr("grid_search"), # defines the tuning for the inner loop incl type of hyper param tuning
                learner = learner, resampling = rsmp("cv", folds = 5), # defines the learner and the resampling technique (5-fold cv)
                measure = msr("surv.cindex"), # defines the metric used to evaluate the performance of the model (C_index) for each fold
                terminator = trm("run_time", secs = 60)) # a largely redundant failsafe denoting that the processing will stop if it takes longer than 60 seconds.

knn_impute  = NADIA::PipeOpVIM_kNN$new(k = 5) # defining the KNN imputation process for missing values, with K = 5.

task = as_task_surv(dat, time = time, event = target) # creating an MLR3 friendly version of the data and explicitly denoting the time and event variables.

task$col_roles$stratum = task$target_names # indentifying the target values to be used for stratification of the data amongst splits.

grn <- as_learner(knn_impute %>>% at) # we incorporate the KNN impute process
# alongside the tuner as a learner object, such that they will be appied to every split of the outer loop, on the train dataset.

#grn$plot() plotting for sanity check

outer_resampling = rsmp("cv", folds = 5) # we now define the outer loop.
# ncol(dat)/3 was taking too long so I put it at 5, which did not adversely affect the stability of the models. See MC results.

outer_resampling$instantiate(task) # applies the outer loop splits to the task, which takes into account stratification

rr = resample(task, grn, outer_resampling, store_models = T) # runs the nested CV

rr$aggregate() # prints the final C-index for the whole dataset, which results from the nested CV process.

post <- Sys.time()
