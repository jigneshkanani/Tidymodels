library(tidymodels)
tidymodels_prefer()
library(readxl)
library(rsample)
library(ranger)
library(skimr)
library(caret)

# speed up computation with parrallel processing
library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

setwd("~/R")

eda_data = read_excel("eda_data.xlsx")

eda_data <- eda_data |> 
  mutate(year = factor(year), qty = log(qty)) |> 
  mutate_if(is.character,as.factor) |> 
  mutate(month =factor(month, levels = c ("Jan","Feb","March","Apr","May","Jun","July","Aug","Sept","Oct","Nov","Dec"))) |> 
  arrange(year,month,weekday) |> 
  dplyr::select('qty', 'year','month', 'weekday','product_name', 'state', "hmonth","temperature") |> 
  collect()

set.seed(93689)
data_split <- initial_split(eda_data, prop = 0.68, strata = qty)
data_train <- training(data_split)
data_test <- testing(data_split)


data_train_dummy <- dummyVars("~ .", data = data_train,fullRank = T)
data_train_encoded <- data.frame(predict(data_train_dummy, newdata = data_train))

data_test_dummy <- dummyVars("~ .", data = data_test,fullRank = T)
data_test_encoded <- data.frame(predict(data_test_dummy, newdata = data_test))

###############################################################
#glm model

glm_set <- linear_reg(penalty = 1) |> set_engine("glmnet") |> translate()

glm_model <- glm_set |> fit(qty~., data = data_train_encoded)


glm_model |> extract_fit_engine() |> summary()


# data_test_small <- data_test |> select(!qty) |>  slice(1:10) 
# 
# 
# x <- data_test |> 
#   select(qty) |> 
#   slice(1:10) |> 
#   bind_cols(exp(predict(glm_model, data_test_small))) |>
#   mutate(qty = exp(qty))
# 
data_matrics <- metric_set(rmse,rsq,mae)
# data_matrics(x, truth = qty, estimate = .pred)


data_test_prdtr <- data_test_encoded |> 
  select(!qty)

test_result <- data_test_encoded |> 
  select(qty) |> 
  bind_cols(predict(glm_model, data_test_prdtr)) |>
  mutate(qty = qty)

data_matrics(test_result, truth = qty, estimate = .pred)


# A tibble: 3 × 3
# .metric .estimator .estimate
# <chr>   <chr>          <dbl>
# 1 rmse    standard       2.19 
# 2 rsq     standard       0.259
# 3 mae     standard       1.80 

ggplot(test_result,aes(x= qty, y = .pred)) +
  geom_abline(lty = 2) +
  geom_point(alpha=0.5) +
  labs(y = "Predicted qty (log)", x = "Qty (log)") +
  coord_obs_pred()

#####################################################################################
# feature engineering

library(earth) 
marsModel <- earth(qty ~ ., data=data_train) # build model
ev <- evimp (marsModel)
print(ev)



# Decide if a variable is important or not using Boruta
library(Boruta)
boruta_output <- Boruta(qty ~ ., data=na.omit(data_train), doTrace=2)  # perform Boruta search
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])  # collect Confirmed and Tentative variables
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  # plot variable importance


library(MASS)
base.mod <- lm(qty ~ 1 , data= data_train)  # base intercept only model
all.mod <- lm(qty ~ . , data= data_train) # full model with all predictors
stepMod <- stepAIC(base.mod, scope = list(lower = base.mod, upper = all.mod), direction = "both", trace = 0, steps = 1000)  # perform step-wise algorithm
tidy(stepMod) |> print(n=99)
shortlistedVars <- names(unlist(stepMod[[1]])) # get the shortlisted variable.
shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"]  # remove intercept 
print(shortlistedVars)

#The p.value column is often used to assess the statistical significance of each coefficient. 
#If the p-value is less than a chosen significance level (commonly 0.05), 
#it suggests that the corresponding predictor variable is statistically significant.

#---------------------------------------------------------------------------------------------------
####################################################################################################
# Randome Forest Model 

# Define the recipe
rf_recipe <- recipe(qty ~ ., data = data_train) %>%
  step_normalize() |>   
  step_dummy(all_nominal_predictors(), one_hot = TRUE)


# Define the random forest model specification
rf_model <- rand_forest(trees = 599, mtry = 5) %>%
  set_mode("regression") %>%
  set_engine("randomForest")

# Combine recipe and model into a workflow
rf_workflow <- workflow() |> 
  add_model(rf_model) |> 
  add_recipe(rf_recipe)



# Train the random forest model using the workflow
rf_fit <- rf_workflow %>%
  fit(data_train)


# Make predictions on the test set
test_processed  <-  rf_fit |> predict(data_test)

test_resutl <- test_processed |> bind_cols(data_test |> select(qty))

data_matrics <- metric_set(rmse,rsq,mae)

data_matrics(data =test_resutl, truth = qty, estimate = .pred)
# .metric .estimator .estimate
# <chr>   <chr>          <dbl>
# 1 rmse    standard       1.14 
# 2 rsq     standard       0.803
# 3 mae     standard       0.880

#-----------------------------------------------------------------------------------------------------

######################################################################################################

# XGBoost model specification
xgb_recipe <- recipe(qty ~ ., data = data_train) %>%
  step_normalize() |>   
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
  prep()

data_cv_folds <- recipes::bake(
  xgb_recipe, 
  new_data = training(data_split)
) |> 
  rsample::vfold_cv(v = 5)

xgboost_model <- 
  parsnip::boost_tree(
    mode = "regression",
    trees = 500,
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
  ) %>%
  set_engine("xgboost", objective = "reg:squarederror")

# grid specification
# We use the tidymodel dials package to specify the parameter set.
xgboost_params <- 
  dials::parameters(
    min_n(),
    tree_depth(),
    learn_rate(),
    loss_reduction()
  )

# Next we set up the grid space. The dails::grid_* functions support several methods 
# for defining the grid space. We are using the dails::grid_max_entropy() function which covers the 
# hyperparameter space such that any portion of the space has an observed 
# combination that is not too far from it.

xgboost_grid <- 
  dials::grid_max_entropy(
    xgboost_params, 
    size = 20
  )


knitr::kable(head(xgboost_grid))

xgboost_wf <- 
  workflows::workflow() |> 
  add_model(xgboost_model)  |>  
  add_formula(qty ~ .)


# hyperparameter tuning
xgboost_tuned <- tune::tune_grid(
  object = xgboost_wf,
  resamples = data_cv_folds,
  grid = xgboost_grid,
  metrics = yardstick::metric_set(rmse, rsq, mae),
  control = tune::control_grid(verbose = TRUE)
)


xgboost_tuned |> 
  tune::show_best(metric = "rsq") |> 
  knitr::kable()


#Next, isolate the best performing hyperparameter values.
xgboost_best_params <- xgboost_tuned %>%
  tune::select_best("rsq")

knitr::kable(xgboost_best_params)

#Finalize the XGBoost model to use the best tuning parameters.
xgboost_model_final <- xgboost_model %>% 
  finalize_model(xgboost_best_params)


# Boosted Tree Model Specification (regression)
# 
# Main Arguments:
#   trees = 500
# min_n = 2
# tree_depth = 13
# learn_rate = 0.044186530948561
# loss_reduction = 1.10936407822299e-09
# 
# Engine-Specific Arguments:
#   objective = reg:squarederror
# 
# Computational engine: xgboost 




train_processed <- bake(xgb_recipe,  new_data = training(data_split))

train_prediction <- xgboost_model_final %>%
  # fit the model on all the training data
  fit(
    formula = qty ~ ., 
    data    = train_processed
  ) %>%
  # predict the sale prices for the training data
  predict(new_data = train_processed) %>%
  bind_cols(training(data_split))

xgboost_score_train <- 
  train_prediction %>%
  yardstick::metrics(qty, .pred) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ","))

knitr::kable(xgboost_score_train)

#   |.metric |.estimator |.estimate |
#   |:-------|:----------|:---------|
#   |rmse    |standard   |0.47      |
#   |rsq     |standard   |0.96      |
#   |mae     |standard   |0.34      |


#### Testing
test_processed  <- bake(xgb_recipe, new_data = testing(data_split))
test_prediction <- xgboost_model_final %>%
  # fit the model on all the testing data
  fit(
    formula = qty ~ ., 
    data    = train_processed
  ) %>%
  # use the testing model fit to predict the test data
  predict(new_data = test_processed) %>%
  bind_cols(testing(data_split))

# measure the accuracy of our model using `yardstick`
xgboost_score <- 
  test_prediction %>%
  yardstick::metrics(qty, .pred) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ","))


knitr::kable(xgboost_score)

#   |.metric |.estimator |.estimate |
#   |:-------|:----------|:---------|
#   |rmse    |standard   |0.69      |
#   |rsq     |standard   |0.91      |
#   |mae     |standard   |0.48      |


qty_prediction_residual <- test_prediction %>%
  arrange(.pred) %>%
  mutate(residual_pct = (qty - .pred) / .pred) %>%
  select(.pred, residual_pct)

ggplot(qty_prediction_residual, aes(x = .pred, y = residual_pct)) +
  geom_point() +
  xlab("Predicted Sale Price") +
  ylab("Residual (%)") +
  scale_x_continuous(labels = scales::number_format()) +
  scale_y_continuous(labels = scales::percent)






##############################################################################################################
# Define the xgboost model specification
xgboost_model <- boost_tree(
  mtry = 6,  # Set the number of features to consider for splitting at each split
  trees = 1000,
  min_n = 11,  # Minimum number of observations in terminal nodes
  tree_depth = 13,  # Maximum depth of the trees
  learn_rate = 0.044186530948561,  # Learning rate
  loss_reduction = 1.1093, # Loss Reduction
) %>%
  set_mode("regression") %>%
  set_engine("xgboost")

# Combine recipe and model into a workflow
xgboost_workflow <- workflow() %>%
  add_model(xgboost_model) |>
  add_recipe(xgb_recipe)

# Train the xgboost model using the workflow
xgboost_fit <- xgboost_workflow %>%
  fit(data_train)

# Make predictions on the test set

test_processed  <-  xgboost_fit |> predict(data_test)

test_resutl <- test_processed |> bind_cols(data_test |> select(qty))

data_matrics <- metric_set(rmse,rsq,mae)

knitr::kable(data_matrics(data =test_resutl, truth = qty, estimate = .pred))

# 
#   |.metric |.estimator | .estimate|
#   |:-------|:----------|---------:|
#   |rmse    |standard   | 0.7544442|
#   |rsq     |standard   | 0.8904388|
#   |mae     |standard   | 0.5332951|

############################################################################################################################

# Support Vector model specification
svm_recipe <- recipe(qty ~ ., data = data_train) %>%
  step_normalize() |>   
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
  prep()

data_cv_folds <- recipes::bake(
  svm_recipe, 
  new_data = training(data_split)
) |> 
  rsample::vfold_cv(v = 5)

svm_model <- 
  parsnip::svm_rbf(
    mode = "regression",
    cost = tune(),
    margin = tune()
    #rbf_sigma = tune()
  ) %>%
  set_engine("kernlab")

# grid specification
# We use the tidymodel dials package to specify the parameter set.
svm_params <- 
  dials::parameters(
    cost(),
    svm_margin()
  )

# Next we set up the grid space. The dails::grid_* functions support several methods 
# for defining the grid space. We are using the dails::grid_max_entropy() function which covers the 
# hyperparameter space such that any portion of the space has an observed 
# combination that is not too far from it.

svm_grid <- 
  dials::grid_max_entropy(
    svm_params, 
    size = 20
  )


knitr::kable(head(svm_grid))

svm_wf <- 
  workflows::workflow() |> 
  add_model(svm_model)  |>  
  add_formula(qty ~ .)


# hyperparameter tuning

library(kernlab)

svm_tuned <- tune::tune_grid(
  object = svm_wf,
  resamples = data_cv_folds,
  grid = svm_grid,
  metrics = yardstick::metric_set(rmse, rsq, mae),
  control = tune::control_grid(verbose = TRUE)
)


svm_tuned |> 
  tune::show_best(metric = "rsq") |> 
  knitr::kable()


#Next, isolate the best performing hyperparameter values.
svm_best_params <- svm_tuned %>%
  tune::select_best("rsq")

knitr::kable(svm_best_params)

#Finalize the XGBoost model to use the best tuning parameters.
svm_model_final <- svm_model %>% 
  finalize_model(svm_best_params)