# This is the full protocol for the nested CV analysis of landslides 
# in Norway during Storm Hans, used in the paper

# ...

# but only using the data from forested areas

# The code performs fully nested CV during hyperparameter tuning of 
# GPBoost models and model re-training on all 5 folds, to then get mean AUC values
# before fitting the final "global" model for interpretation purposes
# Eight models are fit, using four different CV methods, with and without
# a spatial GP term. 

# each method is run one at a time, with final models and optimum parameters
# saved in case users want to skip straight to the models, but it could 
# probably be more strealined with looping for example

# Code written by Mark Gillespie, with help from Claude AI Sonnet 5

# load required packages
library(tidyverse)
library(gpboost)
library(ggplot2)
library(gridExtra)
library(viridis)
library(sf)
library(pROC)
library(spatialsample)


# load data
Lands_forest = read.table("data/Landslides_forest_update.txt", header=T, sep="\t", stringsAsFactors = T)

# remove points with no trees
Lands_forest= Lands_forest |> 
  filter(treeheight!=0 &
           canopy!=0 &
           treenumber!=0)

# convert precipitation intensity
Lands_forest$max_precip_3h = sqrt(Lands_forest$max_precip_3h)

# make a vector of covariate names and categorical variable names
names(Lands_forest)
covars_forest = names(Lands_forest)[c(5:11,13:23)]

cat_feat_forest <- colnames(Lands_forest)[which(colnames(Lands_forest) %in% c("bedrock2","landforms2", 
                                                                              "deposit_thickness", "treetype",
                                                                              "deposit2"))]
# this is a repeat of the full data protocol in the script 
# "Full nested CV protocol.R" - therefore commenting is kept to a minimum

# spatial clustering ####

# convert to an sf object for the clustering
sf_Lands_forest <- st_as_sf(Lands_forest, 
                          coords = c("x", "y"), crs = 32632)

set.seed(8)
spatial_folds <- spatial_clustering_cv(
  data = sf_Lands_forest,
  v = 5,  # 5 folds
  cluster_function = "kmeans"
)
spatial_folds
autoplot(spatial_folds)

extract_folds = function(spatial_cv_object, n_data) {
  fold_ids = rep(NA, n_data)
  for(i in 1:length(spatial_cv_object$splits)) {
    ass_idx = setdiff(1:n_data, spatial_cv_object$splits[[i]]$in_id)
    fold_ids[ass_idx] = i 
  }
  return(fold_ids)
}


fold_id <- extract_folds(spatial_folds, nrow(Lands_forest))

Lands_forest$fold <- fold_id

# NOW check balance
table(Lands_forest$fold, Lands_forest$landslide)


# spatial inner loop ####
# Assign one block as test set, one at a time
test_idx1 <- which(Lands_forest$fold == 1)  
test_idx2 <- which(Lands_forest$fold == 2)  
test_idx3 <- which(Lands_forest$fold == 3)  
test_idx4 <- which(Lands_forest$fold == 4)  
test_idx5 <- which(Lands_forest$fold == 5)  

train_val_idx1 <- which(Lands_forest$fold != 1)  
train_val_idx2 <- which(Lands_forest$fold != 2)  
train_val_idx3 <- which(Lands_forest$fold != 3)  
train_val_idx4 <- which(Lands_forest$fold != 4)  
train_val_idx5 <- which(Lands_forest$fold != 5)  

# Lock away the test set
test_data1 <- Lands_forest[test_idx1, ]
test_data2 <- Lands_forest[test_idx2, ]
test_data3 <- Lands_forest[test_idx3, ]
test_data4 <- Lands_forest[test_idx4, ]
test_data5 <- Lands_forest[test_idx5, ]


train_val_data1 <- Lands_forest[train_val_idx1, ]  
train_val_data2 <- Lands_forest[train_val_idx2, ]  
train_val_data3 <- Lands_forest[train_val_idx3, ]  
train_val_data4 <- Lands_forest[train_val_idx4, ]  
train_val_data5 <- Lands_forest[train_val_idx5, ]  

# STEP 2: Hyperparameter tuning with CV on training set ONLY
# Create NEW spatial folds within the training data
train_val_sf1 <- st_as_sf(train_val_data1, coords = c("x", "y"), crs = 32632)
train_val_sf2 <- st_as_sf(train_val_data2, coords = c("x", "y"), crs = 32632)
train_val_sf3 <- st_as_sf(train_val_data3, coords = c("x", "y"), crs = 32632)
train_val_sf4 <- st_as_sf(train_val_data4, coords = c("x", "y"), crs = 32632)
train_val_sf5 <- st_as_sf(train_val_data5, coords = c("x", "y"), crs = 32632)

set.seed(5)

spatial_folds_train1 <- spatial_clustering_cv(
  data = train_val_sf1,
  v = 4,  # 5 folds
  cluster_function = "kmeans"
)

fold_id_train1 <- extract_folds(spatial_folds_train1, nrow(train_val_sf1))

train_val_data1$fold <- fold_id_train1
table(train_val_data1$fold , train_val_data1$landslide)

set.seed(9)

spatial_folds_train2 <- spatial_clustering_cv(
  data = train_val_sf2,
  v = 4,  # 5 folds
  cluster_function = "kmeans"
)

fold_id_train2 <- extract_folds(spatial_folds_train2, nrow(train_val_sf2))

train_val_data2$fold <- fold_id_train2
table(train_val_data2$fold , train_val_data2$landslide)

set.seed(10)

spatial_folds_train3 <- spatial_clustering_cv(
  data = train_val_sf3,
  v = 4,  # 5 folds
  cluster_function = "kmeans"
)

fold_id_train3 <- extract_folds(spatial_folds_train3, nrow(train_val_sf3))

train_val_data3$fold <- fold_id_train3
table(train_val_data3$fold , train_val_data3$landslide)

set.seed(8)

spatial_folds_train4 <- spatial_clustering_cv(
  data = train_val_sf4,
  v = 4,  # 5 folds
  cluster_function = "kmeans"
)

fold_id_train4 <- extract_folds(spatial_folds_train4, nrow(train_val_sf4))

train_val_data4$fold <- fold_id_train4
table(train_val_data4$fold , train_val_data4$landslide)

spatial_folds_train5 <- spatial_clustering_cv(
  data = train_val_sf5,
  v = 4,  # 5 folds
  cluster_function = "kmeans"
)

fold_id_train5 <- extract_folds(spatial_folds_train5, nrow(train_val_sf5))

train_val_data5$fold <- fold_id_train5
table(train_val_data5$fold , train_val_data5$landslide)


# Convert to list format that gpb.grid.search.tune.parameters expects
# Each list element = indices of validation observations for that fold
folds_list1 <- list()
folds_list2 <- list()
folds_list3 <- list()
folds_list4 <- list()
folds_list5 <- list()

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list1[[i]] <- which(fold_id_train1 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list2[[i]] <- which(fold_id_train2 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list3[[i]] <- which(fold_id_train3 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list4[[i]] <- which(fold_id_train4 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list5[[i]] <- which(fold_id_train5 == i)
}


# parameter grid for tuning - same as full dataset
param_grid <- list(
  learning_rate = c(0.05,0.01), # needs to be low
  min_data_in_leaf = c(10, 20, 50), #10-100 is good for generalisation
  num_leaves = c(8,16,32), # 2-8 strong reg, 16-128 - balance
  lambda_l2 = c(10, 50, 100), # 0 - no reg, 1-10 mild, 100 strong but may underfit
  max_depth = c(-1)
) 


other_params <- list(
  objective = "binary",
  verbose = -1
)

# re instate model and data
X_train_all1 <- as.matrix(train_val_data1[, covars_forest])
X_train_all2 <- as.matrix(train_val_data2[, covars_forest])
X_train_all3 <- as.matrix(train_val_data3[, covars_forest])
X_train_all4 <- as.matrix(train_val_data4[, covars_forest])
X_train_all5 <- as.matrix(train_val_data5[, covars_forest])

y_train_all1 <- train_val_data1[, "landslide"]
y_train_all2 <- train_val_data2[, "landslide"]
y_train_all3 <- train_val_data3[, "landslide"]
y_train_all4 <- train_val_data4[, "landslide"]
y_train_all5 <- train_val_data5[, "landslide"]


coords_train_all1 <- as.matrix(train_val_data1[, c("x", "y")])
coords_train_all2 <- as.matrix(train_val_data2[, c("x", "y")])
coords_train_all3 <- as.matrix(train_val_data3[, c("x", "y")])
coords_train_all4 <- as.matrix(train_val_data4[, c("x", "y")])
coords_train_all5 <- as.matrix(train_val_data5[, c("x", "y")])

dtrain_all1 <- gpb.Dataset(data = X_train_all1,
                           label = y_train_all1,
                           categorical_feature = cat_feat_forest)
dtrain_all2 <- gpb.Dataset(data = X_train_all2,
                           label = y_train_all2,
                           categorical_feature = cat_feat_forest)
dtrain_all3 <- gpb.Dataset(data = X_train_all3,
                           label = y_train_all3,
                           categorical_feature = cat_feat_forest)
dtrain_all4 <- gpb.Dataset(data = X_train_all4,
                           label = y_train_all4,
                           categorical_feature = cat_feat_forest)
dtrain_all5 <- gpb.Dataset(data = X_train_all5,
                           label = y_train_all5,
                           categorical_feature = cat_feat_forest)

gp_model_final1 <- GPModel(gp_coords = coords_train_all1, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final2 <- GPModel(gp_coords = coords_train_all2, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final3 <- GPModel(gp_coords = coords_train_all3, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final4 <- GPModel(gp_coords = coords_train_all4, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final5 <- GPModel(gp_coords = coords_train_all5, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)

opt_params1 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list1,  
  data = dtrain_all1,
  gp_model = gp_model_final1,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params2 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list2,  
  data = dtrain_all2,
  gp_model = gp_model_final2,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params3 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list3,  
  data = dtrain_all3,
  gp_model = gp_model_final3,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params4 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list4,  
  data = dtrain_all4,
  gp_model = gp_model_final4,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params5 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list5,  
  data = dtrain_all5,
  gp_model = gp_model_final5,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)


# summarise optimum params and auc

cv_res = data.frame(Model = "Spatial_spatial_Forest",
                       learning_rate = median(c(opt_params1$best_params$learning_rate,
                                                opt_params2$best_params$learning_rate,
                                                opt_params3$best_params$learning_rate,
                                                opt_params4$best_params$learning_rate,
                                                opt_params5$best_params$learning_rate)),
                       min_data_in_leaf = median(c(opt_params1$best_params$min_data_in_leaf,
                                                   opt_params2$best_params$min_data_in_leaf,
                                                   opt_params3$best_params$min_data_in_leaf,
                                                   opt_params4$best_params$min_data_in_leaf,
                                                   opt_params5$best_params$min_data_in_leaf)),
                       num_leaves = median(c(opt_params1$best_params$num_leaves,
                                             opt_params2$best_params$num_leaves,
                                             opt_params3$best_params$num_leaves,
                                             opt_params4$best_params$num_leaves,
                                             opt_params5$best_params$num_leaves)),
                       lambda_l2 = median(c(opt_params1$best_params$lambda_l2,
                                            opt_params2$best_params$lambda_l2,
                                            opt_params3$best_params$lambda_l2,
                                            opt_params4$best_params$lambda_l2,
                                            opt_params5$best_params$lambda_l2)),
                       best_iter = mean(c(opt_params1$best_iter,
                                          opt_params2$best_iter,
                                          opt_params3$best_iter,
                                          opt_params4$best_iter,
                                          opt_params5$best_iter)),
                       mean_auc_hyp = mean(c(opt_params1$best_score,
                                             opt_params2$best_score,
                                             opt_params3$best_score,
                                             opt_params4$best_score,
                                             opt_params5$best_score)),
                       sd_auc_hyp =  sd(c(opt_params1$best_score,
                                          opt_params2$best_score,
                                          opt_params3$best_score,
                                          opt_params4$best_score,
                                          opt_params5$best_score))
)

# STEP 4: Retrain on ALL training data with best parameters x 5
# Now use ALL of train_val_data
X_train_all1 <- as.matrix(train_val_data1[, covars_forest])
y_train_all1 <- train_val_data1[, "landslide"]
coords_train_all1 <- as.matrix(train_val_data1[, c("x", "y")])

dtrain_all1 <- gpb.Dataset(data = X_train_all1,
                           label = y_train_all1,
                           categorical_feature = cat_feat_forest)

gp_model_final1 <- GPModel(gp_coords = coords_train_all1, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)

final_model1 <- gpboost(
  data = dtrain_all1,
  gp_model = gp_model_final1,
  nrounds = opt_params1$best_iter,
  learning_rate = opt_params1$best_params$learning_rate,
  num_leaves = opt_params1$best_params$num_leaves,
  min_data_in_leaf = opt_params1$best_params$min_data_in_leaf,
  lambda_l2 = opt_params1$best_params$lambda_l2,
  verbose = 1
)

pred_train1 <- predict(final_model1, 
                       data = as.matrix(train_val_data1[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data1[, c("x", "y")]),
                       predict_var = FALSE)

train_auc1 <- auc(roc(y_train_all1, pred_train1$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test1 <- as.matrix(test_data1[, covars_forest])
y_test1 <- test_data1[, "landslide"]
coords_test1 <- as.matrix(test_data1[, c("x", "y")])

test_pred1 <- predict(final_model1, 
                      data = X_test1, 
                      gp_coords_pred = coords_test1)

test_auc1 <- auc(roc(y_test1, test_pred1$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop1 = gp_model_final1$get_cov_pars()[1] / (gp_model_final1$get_cov_pars()[1]+1)

library(spdep)
library(sf)

# 1. Pearson residuals 
pearson_residuals1 <- (y_train_all1 - pred_train1$response_mean) / 
  sqrt(pred_train1$response_mean * (1 - pred_train1$response_mean))

# Create spatial neighbors 

coords1 <- cbind(train_val_data1[,"x"], train_val_data1[,"y"])
knn1 <- knearneigh(coords1, k = 12)  # adjust k based on your data
nb1 <- knn2nb(knn1)
weights1 <- nb2listw(nb1, style = "W")

# Test for spatial autocorrelation
moran_test1 <- moran.test(pearson_residuals1, weights1)
print(moran_test1)


# model 2
X_train_all2 <- as.matrix(train_val_data2[, covars_forest])
y_train_all2 <- train_val_data2[, "landslide"]
coords_train_all2 <- as.matrix(train_val_data2[, c("x", "y")])

dtrain_all2 <- gpb.Dataset(data = X_train_all2,
                           label = y_train_all2,
                           categorical_feature = cat_feat_forest)

gp_model_final2 <- GPModel(gp_coords = coords_train_all2, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model2 <- gpboost(
  data = dtrain_all2,
  gp_model = gp_model_final2,
  nrounds = opt_params2$best_iter,
  learning_rate = opt_params2$best_params$learning_rate,
  num_leaves = opt_params2$best_params$num_leaves,
  min_data_in_leaf = opt_params2$best_params$min_data_in_leaf,
  lambda_l2 = opt_params2$best_params$lambda_l2,
  verbose = 1
)

pred_train2 <- predict(final_model2, 
                       data = as.matrix(train_val_data2[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data2[, c("x", "y")]),
                       predict_var = FALSE)

train_auc2 <- auc(roc(y_train_all2, pred_train2$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test2 <- as.matrix(test_data2[, covars_forest])
y_test2 <- test_data2[, "landslide"]
coords_test2 <- as.matrix(test_data2[, c("x", "y")])

test_pred2 <- predict(final_model2, 
                      data = X_test2, 
                      gp_coords_pred = coords_test2)

test_auc2 <- auc(roc(y_test2, test_pred2$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop2 = gp_model_final2$get_cov_pars()[1] / (gp_model_final2$get_cov_pars()[1]+1)

# 2. Pearson residuals 
pearson_residuals2 <- (y_train_all2 - pred_train2$response_mean) / 
  sqrt(pred_train2$response_mean * (1 - pred_train2$response_mean))

# Create spatial neighbors 

coords2 <- cbind(train_val_data2[,"x"], train_val_data2[,"y"])
knn2 <- knearneigh(coords2, k = 12)  # adjust k based on your data
nb2 <- knn2nb(knn2)
weights2 <- nb2listw(nb2, style = "W")

# Test for spatial autocorrelation
moran_test2 <- moran.test(pearson_residuals2, weights2)
print(moran_test2)


# model 3
X_train_all3 <- as.matrix(train_val_data3[, covars_forest])
y_train_all3 <- train_val_data3[, "landslide"]
coords_train_all3 <- as.matrix(train_val_data3[, c("x", "y")])

dtrain_all3 <- gpb.Dataset(data = X_train_all3,
                           label = y_train_all3,
                           categorical_feature = cat_feat_forest)

gp_model_final3 <- GPModel(gp_coords = coords_train_all3, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model3 <- gpboost(
  data = dtrain_all3,
  gp_model = gp_model_final3,
  nrounds = opt_params3$best_iter,
  learning_rate = opt_params3$best_params$learning_rate,
  num_leaves = opt_params3$best_params$num_leaves,
  min_data_in_leaf = opt_params3$best_params$min_data_in_leaf,
  lambda_l2 = opt_params3$best_params$lambda_l2,
  verbose = 1
)

pred_train3 <- predict(final_model3, 
                       data = as.matrix(train_val_data3[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data3[, c("x", "y")]),
                       predict_var = FALSE)

train_auc3 <- auc(roc(y_train_all3, pred_train3$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test3 <- as.matrix(test_data3[, covars_forest])
y_test3 <- test_data3[, "landslide"]
coords_test3 <- as.matrix(test_data3[, c("x", "y")])

test_pred3 <- predict(final_model3, 
                      data = X_test3, 
                      gp_coords_pred = coords_test3)

test_auc3 <- auc(roc(y_test3, test_pred3$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop3 = gp_model_final3$get_cov_pars()[1] / (gp_model_final3$get_cov_pars()[1]+1)


# 3. Pearson residuals 
pearson_residuals3 <- (y_train_all3 - pred_train3$response_mean) / 
  sqrt(pred_train3$response_mean * (1 - pred_train3$response_mean))

# Create spatial neighbors 

coords3 <- cbind(train_val_data3[,"x"], train_val_data3[,"y"])
knn3 <- knearneigh(coords3, k = 12)  # adjust k based on your data
nb3 <- knn2nb(knn3)
weights3 <- nb2listw(nb3, style = "W")

# Test for spatial autocorrelation
moran_test3 <- moran.test(pearson_residuals3, weights3)
print(moran_test3)

# model 4
X_train_all4 <- as.matrix(train_val_data4[, covars_forest])
y_train_all4 <- train_val_data4[, "landslide"]
coords_train_all4 <- as.matrix(train_val_data4[, c("x", "y")])

dtrain_all4 <- gpb.Dataset(data = X_train_all4,
                           label = y_train_all4,
                           categorical_feature = cat_feat_forest)

gp_model_final4 <- GPModel(gp_coords = coords_train_all4, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model4 <- gpboost(
  data = dtrain_all4,
  gp_model = gp_model_final4,
  nrounds = opt_params4$best_iter,
  learning_rate = opt_params4$best_params$learning_rate,
  num_leaves = opt_params4$best_params$num_leaves,
  min_data_in_leaf = opt_params4$best_params$min_data_in_leaf,
  lambda_l2 = opt_params4$best_params$lambda_l2,
  verbose = 1
)

pred_train4 <- predict(final_model4, 
                       data = as.matrix(train_val_data4[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data4[, c("x", "y")]),
                       predict_var = FALSE)

train_auc4 <- auc(roc(y_train_all4, pred_train4$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test4 <- as.matrix(test_data4[, covars_forest])
y_test4 <- test_data4[, "landslide"]
coords_test4 <- as.matrix(test_data4[, c("x", "y")])

test_pred4 <- predict(final_model4, 
                      data = X_test4, 
                      gp_coords_pred = coords_test4)

test_auc4 <- auc(roc(y_test4, test_pred4$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop4 = gp_model_final4$get_cov_pars()[1] / (gp_model_final4$get_cov_pars()[1]+1)

# 4. Pearson residuals 
pearson_residuals4 <- (y_train_all4 - pred_train4$response_mean) / 
  sqrt(pred_train4$response_mean * (1 - pred_train4$response_mean))

# Create spatial neighbors 

coords4 <- cbind(train_val_data4[,"x"], train_val_data4[,"y"])
knn4 <- knearneigh(coords4, k = 12)  # adjust k based on your data
nb4 <- knn2nb(knn4)
weights4 <- nb2listw(nb4, style = "W")

# Test for spatial autocorrelation
moran_test4 <- moran.test(pearson_residuals4, weights4)
print(moran_test4)

# model 5
X_train_all5 <- as.matrix(train_val_data5[, covars_forest])
y_train_all5 <- train_val_data5[, "landslide"]
coords_train_all5 <- as.matrix(train_val_data5[, c("x", "y")])

dtrain_all5 <- gpb.Dataset(data = X_train_all5,
                           label = y_train_all5,
                           categorical_feature = cat_feat_forest)

gp_model_final5 <- GPModel(gp_coords = coords_train_all5, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model5 <- gpboost(
  data = dtrain_all5,
  gp_model = gp_model_final5,
  nrounds = opt_params5$best_iter,
  learning_rate = opt_params5$best_params$learning_rate,
  num_leaves = opt_params5$best_params$num_leaves,
  min_data_in_leaf = opt_params5$best_params$min_data_in_leaf,
  lambda_l2 = opt_params5$best_params$lambda_l2,
  verbose = 1
)

pred_train5 <- predict(final_model5, 
                       data = as.matrix(train_val_data5[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data5[, c("x", "y")]),
                       predict_var = FALSE)

train_auc5 <- auc(roc(y_train_all5, pred_train5$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test5 <- as.matrix(test_data5[, covars_forest])
y_test5 <- test_data5[, "landslide"]
coords_test5 <- as.matrix(test_data5[, c("x", "y")])

test_pred5 <- predict(final_model5, 
                      data = X_test5, 
                      gp_coords_pred = coords_test5)

test_auc5 <- auc(roc(y_test5, test_pred5$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop5 = gp_model_final5$get_cov_pars()[1] / (gp_model_final5$get_cov_pars()[1]+1)

# 5. Pearson residuals 
pearson_residuals5 <- (y_train_all5 - pred_train5$response_mean) / 
  sqrt(pred_train5$response_mean * (1 - pred_train5$response_mean))

# Create spatial neighbors 

coords5 <- cbind(train_val_data5[,"x"], train_val_data5[,"y"])
knn5 <- knearneigh(coords5, k = 12)  # adjust k based on your data
nb5 <- knn2nb(knn5)
weights5 <- nb2listw(nb5, style = "W")

# Test for spatial autocorrelation
moran_test5 <- moran.test(pearson_residuals5, weights5)
print(moran_test5)

cv_res = cv_res |> 
  mutate(mean_train_auc = mean(c(train_auc1,train_auc2,train_auc3,train_auc4,train_auc5)),
         sd_train_auc = sd(c(train_auc1,train_auc2,train_auc3,train_auc4,train_auc5)),
         mean_test_auc = mean(c(test_auc1, test_auc2, test_auc3, test_auc4, test_auc5)),
         sd_test_auc = sd(c(test_auc1, test_auc2, test_auc3, test_auc4, test_auc5)),
         mean_space = mean(c(space_prop1, space_prop2, space_prop3, space_prop4, space_prop5)),
         low_moran = min(c(moran_test1$estimate[[1]],
                           moran_test2$estimate[[1]],
                           moran_test3$estimate[[1]],
                           moran_test4$estimate[[1]],
                           moran_test5$estimate[[1]])),
         hi_moran = max(c(moran_test1$estimate[[1]],
                          moran_test2$estimate[[1]],
                          moran_test3$estimate[[1]],
                          moran_test4$estimate[[1]],
                          moran_test5$estimate[[1]])),
         low_p = min(c(moran_test1$p.value[[1]],
                       moran_test2$p.value[[1]],
                       moran_test3$p.value[[1]],
                       moran_test4$p.value[[1]],
                       moran_test5$p.value[[1]])),
         hi_p = max(c(moran_test1$p.value[[1]],
                      moran_test2$p.value[[1]],
                      moran_test3$p.value[[1]],
                      moran_test4$p.value[[1]],
                      moran_test5$p.value[[1]])))

## final global model on full data set with median hyperparameters

X_all <- as.matrix(Lands_forest[, covars_forest])
y_all <- Lands_forest[, "landslide"]
coords_all <- as.matrix(Lands_forest[, c("x", "y")])

d_all <- gpb.Dataset(data = X_all,
                     label = y_all,
                     categorical_feature = cat_feat_forest)

gp_model_final_all <- GPModel(gp_coords = coords_all, 
                              likelihood = "bernoulli_probit",
                              cov_function = "matern" ,
                              cov_fct_shape = 1.5)



# final model
final_model_all <- gpboost(
  data = d_all,
  gp_model = gp_model_final_all,
  nrounds = round(cv_res$best_iter,0),
  learning_rate = cv_res$learning_rate,
  num_leaves = cv_res$num_leaves,
  min_data_in_leaf = cv_res$min_data_in_leaf,
  lambda_l2 = cv_res$lambda_l2,
  verbose = 1
)

# final_model_all = readRDS.gpb.Booster("saved models/Sp_sp_forest_final.rds")

pred_all <- predict(final_model_all, 
                    data = as.matrix(Lands_forest[, covars_forest]),
                    gp_coords_pred = as.matrix(Lands_forest[, c("x", "y")]),
                    predict_var = FALSE)

all_auc <- auc(roc(y_all, pred_all$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop_all = gp_model_final_all$get_cov_pars()[1] / (gp_model_final_all$get_cov_pars()[1]+1)

# 5. Pearson residuals 
pearson_residuals_all <- (y_all - pred_all$response_mean) / 
  sqrt(pred_all$response_mean * (1 - pred_all$response_mean))

# Create spatial neighbors 

coords_all <- cbind(Lands_forest[,"x"], Lands_forest[,"y"])
knn_all <- knearneigh(coords_all, k = 12)  # adjust k based on your data
nb_all <- knn2nb(knn_all)
weights_all <- nb2listw(nb_all, style = "W")

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)


library(SHAPforxgboost)
shap_values_forest <- shap.values(xgb_model = final_model_all, 
                                X_train = X_all)
shap.plot.summary.wrap1(final_model_all, X = X_all) + ggtitle("SHAP values")
shap_long <- shap.prep(final_model_all, X_train = X_all)
fig_list <- lapply(names(shap_values_forest$mean_shap_score)[1:9], 
                   shap.plot.dependence, data_long = shap_long)
gridExtra::grid.arrange(grobs = fig_list, ncol = 3)


saveRDS.gpb.Booster(final_model_all, "saved models/Sp_sp_forest_final.rds")


## non spatial version of spatial clustering CV ####
# already have all the folds etc
opt_params1_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list1,  
  data = dtrain_all1,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params2_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list2,  
  data = dtrain_all2,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params3_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list3,  
  data = dtrain_all3,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params4_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list4,  
  data = dtrain_all4,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params5_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list5,  
  data = dtrain_all5,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

cv_res_ns = data.frame(Model = "Non_Spatial_spatial_Forest",
                          learning_rate = median(c(opt_params1_ns$best_params$learning_rate,
                                                   opt_params2_ns$best_params$learning_rate,
                                                   opt_params3_ns$best_params$learning_rate,
                                                   opt_params4_ns$best_params$learning_rate,
                                                   opt_params5_ns$best_params$learning_rate)),
                          min_data_in_leaf = median(c(opt_params1_ns$best_params$min_data_in_leaf,
                                                      opt_params2_ns$best_params$min_data_in_leaf,
                                                      opt_params3_ns$best_params$min_data_in_leaf,
                                                      opt_params4_ns$best_params$min_data_in_leaf,
                                                      opt_params5_ns$best_params$min_data_in_leaf)),
                          num_leaves = median(c(opt_params1_ns$best_params$num_leaves,
                                                opt_params2_ns$best_params$num_leaves,
                                                opt_params3_ns$best_params$num_leaves,
                                                opt_params4_ns$best_params$num_leaves,
                                                opt_params5_ns$best_params$num_leaves)),
                          lambda_l2 = median(c(opt_params1_ns$best_params$lambda_l2,
                                               opt_params2_ns$best_params$lambda_l2,
                                               opt_params3_ns$best_params$lambda_l2,
                                               opt_params4_ns$best_params$lambda_l2,
                                               opt_params5_ns$best_params$lambda_l2)),
                          best_iter = median(c(opt_params1_ns$best_iter,
                                               opt_params2_ns$best_iter,
                                               opt_params3_ns$best_iter,
                                               opt_params4_ns$best_iter,
                                               opt_params5_ns$best_iter)),
                          mean_auc_hyp = mean(c(opt_params1_ns$best_score,
                                                opt_params2_ns$best_score,
                                                opt_params3_ns$best_score,
                                                opt_params4_ns$best_score,
                                                opt_params5_ns$best_score)),
                          sd_auc_hyp =  sd(c(opt_params1_ns$best_score,
                                             opt_params2_ns$best_score,
                                             opt_params3_ns$best_score,
                                             opt_params4_ns$best_score,
                                             opt_params5_ns$best_score))
)

# fit models with best parameters - non_spatial ####
final_model1_ns <- gpboost(
  data = dtrain_all1,
  gp_model = NULL,
  nrounds = opt_params1_ns$best_iter,
  learning_rate = opt_params1_ns$best_params$learning_rate,
  num_leaves = opt_params1_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params1_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params1_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train1_ns <- predict(final_model1_ns, 
                          data = as.matrix(train_val_data1[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data1[, c("x", "y")]),
                          predict_var = FALSE)

train_auc1_ns <- auc(roc(y_train_all1, pred_train1_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred1_ns <- predict(final_model1_ns, 
                         data = X_test1, 
                         gp_coords_pred = coords_test1)

test_auc1_ns <- auc(roc(y_test1, test_pred1_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe1 <- pmax(pmin(pred_train1_ns, 1 - epsilon), epsilon)
pearson_residuals1_ns <- (y_train_all1 - y_pred_prob_safe1) / 
  sqrt(y_pred_prob_safe1 * (1 - y_pred_prob_safe1))

# Test for spatial autocorrelation
moran_test1_ns <- moran.test(pearson_residuals1_ns, weights1)
print(moran_test1_ns)


# model 2
final_model2_ns <- gpboost(
  data = dtrain_all2,
  gp_model = NULL,
  nrounds = opt_params2_ns$best_iter,
  learning_rate = opt_params2_ns$best_params$learning_rate,
  num_leaves = opt_params2_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params2_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params2_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train2_ns <- predict(final_model2_ns, 
                          data = as.matrix(train_val_data2[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data2[, c("x", "y")]),
                          predict_var = FALSE)

train_auc2_ns <- auc(roc(y_train_all2, pred_train2_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred2_ns <- predict(final_model2_ns, 
                         data = X_test2, 
                         gp_coords_pred = coords_test2)

test_auc2_ns <- auc(roc(y_test2, test_pred2_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe2 <- pmax(pmin(pred_train2_ns, 1 - epsilon), epsilon)
pearson_residuals2_ns <- (y_train_all2 - y_pred_prob_safe2) / 
  sqrt(y_pred_prob_safe2 * (1 - y_pred_prob_safe2))

# Test for spatial autocorrelation
moran_test2_ns <- moran.test(pearson_residuals2_ns, weights2)
print(moran_test2_ns)

# model 3
final_model3_ns <- gpboost(
  data = dtrain_all3,
  gp_model = NULL,
  nrounds = opt_params3_ns$best_iter,
  learning_rate = opt_params3_ns$best_params$learning_rate,
  num_leaves = opt_params3_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params3_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params3_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train3_ns <- predict(final_model3_ns, 
                          data = as.matrix(train_val_data3[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data3[, c("x", "y")]),
                          predict_var = FALSE)

train_auc3_ns <- auc(roc(y_train_all3, pred_train3_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred3_ns <- predict(final_model3_ns, 
                         data = X_test3, 
                         gp_coords_pred = coords_test3)

test_auc3_ns <- auc(roc(y_test3, test_pred3_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe3 <- pmax(pmin(pred_train3_ns, 1 - epsilon), epsilon)
pearson_residuals3_ns <- (y_train_all3 - y_pred_prob_safe3) / 
  sqrt(y_pred_prob_safe3 * (1 - y_pred_prob_safe3))

# Test for spatial autocorrelation
moran_test3_ns <- moran.test(pearson_residuals3_ns, weights3)
print(moran_test3_ns)

# model 4
final_model4_ns <- gpboost(
  data = dtrain_all4,
  gp_model = NULL,
  nrounds = opt_params4_ns$best_iter,
  learning_rate = opt_params4_ns$best_params$learning_rate,
  num_leaves = opt_params4_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params4_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params4_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train4_ns <- predict(final_model4_ns, 
                          data = as.matrix(train_val_data4[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data4[, c("x", "y")]),
                          predict_var = FALSE)

train_auc4_ns <- auc(roc(y_train_all4, pred_train4_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred4_ns <- predict(final_model4_ns, 
                         data = X_test4, 
                         gp_coords_pred = coords_test4)

test_auc4_ns <- auc(roc(y_test4, test_pred4_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe4 <- pmax(pmin(pred_train4_ns, 1 - epsilon), epsilon)
pearson_residuals4_ns <- (y_train_all4 - y_pred_prob_safe4) / 
  sqrt(y_pred_prob_safe4 * (1 - y_pred_prob_safe4))

# Test for spatial autocorrelation
moran_test4_ns <- moran.test(pearson_residuals4_ns, weights4)
print(moran_test4_ns)

# model 5
final_model5_ns <- gpboost(
  data = dtrain_all5,
  gp_model = NULL,
  nrounds = opt_params5_ns$best_iter,
  learning_rate = opt_params5_ns$best_params$learning_rate,
  num_leaves = opt_params5_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params5_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params5_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train5_ns <- predict(final_model5_ns, 
                          data = as.matrix(train_val_data5[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data5[, c("x", "y")]),
                          predict_var = FALSE)

train_auc5_ns <- auc(roc(y_train_all5, pred_train5_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred5_ns <- predict(final_model5_ns, 
                         data = X_test5, 
                         gp_coords_pred = coords_test5)

test_auc5_ns <- auc(roc(y_test5, test_pred5_ns))


# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe5 <- pmax(pmin(pred_train5_ns, 1 - epsilon), epsilon)
pearson_residuals5_ns <- (y_train_all5 - y_pred_prob_safe5) / 
  sqrt(y_pred_prob_safe5 * (1 - y_pred_prob_safe5))

# Test for spatial autocorrelation
moran_test5_ns <- moran.test(pearson_residuals5_ns, weights5)
print(moran_test5_ns)

cv_res_ns = cv_res_ns |> 
  mutate(mean_train_auc = mean(c(train_auc1_ns,train_auc2_ns,train_auc3_ns,train_auc4_ns,train_auc5_ns)),
         sd_train_auc = sd(c(train_auc1_ns,train_auc2_ns,train_auc3_ns,train_auc4_ns,train_auc5_ns)),
         mean_test_auc = mean(c(test_auc1_ns, test_auc2_ns, test_auc3_ns, test_auc4_ns, test_auc5_ns)),
         sd_test_auc = sd(c(test_auc1_ns, test_auc2_ns, test_auc3_ns, test_auc4_ns, test_auc5_ns)),
         mean_space = "NA",
         low_moran = min(c(moran_test1_ns$estimate[[1]],
                           moran_test2_ns$estimate[[1]],
                           moran_test3_ns$estimate[[1]],
                           moran_test4_ns$estimate[[1]],
                           moran_test5_ns$estimate[[1]])),
         hi_moran = max(c(moran_test1_ns$estimate[[1]],
                          moran_test2_ns$estimate[[1]],
                          moran_test3_ns$estimate[[1]],
                          moran_test4_ns$estimate[[1]],
                          moran_test5_ns$estimate[[1]])),
         low_p = min(c(moran_test1_ns$p.value[[1]],
                       moran_test2_ns$p.value[[1]],
                       moran_test3_ns$p.value[[1]],
                       moran_test4_ns$p.value[[1]],
                       moran_test5_ns$p.value[[1]])),
         hi_p = max(c(moran_test1_ns$p.value[[1]],
                      moran_test2_ns$p.value[[1]],
                      moran_test3_ns$p.value[[1]],
                      moran_test4_ns$p.value[[1]],
                      moran_test5_ns$p.value[[1]])))

## final final model on full data set with median hyperparameters

cv_res = rbind(cv_res,cv_res_ns)

write.csv(cv_res, "data/Nested_CV_results_forest.csv")

# load if needed
# cv_res = read.csv("Files from homePC/data/Nested_CV_results_forest.csv")

cv_res_v2 = cv_res[2,]

# final model
final_model_all <- gpboost(
  data = d_all,
  gp_model = NULL,
  nrounds = cv_res_v2$best_iter,
  learning_rate = cv_res_v2$learning_rate,
  num_leaves = cv_res_v2$num_leaves,
  min_data_in_leaf = cv_res_v2$min_data_in_leaf,
  lambda_l2 = cv_res_v2$lambda_l2,
  verbose = 1
)

saveRDS.gpb.Booster(final_model_all, "saved models/Non_Sp_sp_forest_final.rds")

pred_all <- predict(final_model_all, 
                    data = as.matrix(Lands_forest[, covars_forest]),
                    gp_coords_pred = as.matrix(Lands_forest[, c("x", "y")]),
                    predict_var = FALSE)

all_auc <- auc(roc(y_all, pred_all$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop_all = gp_model_final_all$get_cov_pars()[1] / (gp_model_final_all$get_cov_pars()[1]+1)

# 5. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe_all <- pmax(pmin(pred_all, 1 - epsilon), epsilon)
pearson_residuals_all <- (y_all - y_pred_prob_safe_all) / 
  sqrt(y_pred_prob_safe_all * (1 - y_pred_prob_safe_all))

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)

# Create spatial neighbors 

coords_all <- cbind(Lands_main[,"x"], Lands_main[,"y"])
knn_all <- knearneigh(coords_all, k = 12)  # adjust k based on your data
nb_all <- knn2nb(knn_all)
weights_all <- nb2listw(nb_all, style = "W")

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)


#### same again but with small block CV ####
library(blockCV)
library(sf)
set.seed(23)
block_cv_small <- spatial_block_cv(
  data = sf_Lands_forest,
  v= 4,
  n = c(8,8)
)

autoplot(block_cv_small)

fold_id <- extract_folds(block_cv_small, nrow(Lands_forest))
Lands_forest$fold <- fold_id

# NOW check balance
table(Lands_forest$fold, Lands_forest$landslide)


# spatial inner loop ####
# Assign one block as test set, one at a time
test_idx1 <- which(Lands_forest$fold == 1)  
test_idx2 <- which(Lands_forest$fold == 2)  
test_idx3 <- which(Lands_forest$fold == 3)  
test_idx4 <- which(Lands_forest$fold == 4)  
test_idx5 <- which(Lands_forest$fold == 5)  

train_val_idx1 <- which(Lands_forest$fold != 1)  
train_val_idx2 <- which(Lands_forest$fold != 2)  
train_val_idx3 <- which(Lands_forest$fold != 3)  
train_val_idx4 <- which(Lands_forest$fold != 4)  
train_val_idx5 <- which(Lands_forest$fold != 5)  

# Lock away the test set
test_data1 <- Lands_forest[test_idx1, ]
test_data2 <- Lands_forest[test_idx2, ]
test_data3 <- Lands_forest[test_idx3, ]
test_data4 <- Lands_forest[test_idx4, ]
test_data5 <- Lands_forest[test_idx5, ]


train_val_data1 <- Lands_forest[train_val_idx1, ]  
train_val_data2 <- Lands_forest[train_val_idx2, ]  
train_val_data3 <- Lands_forest[train_val_idx3, ]  
train_val_data4 <- Lands_forest[train_val_idx4, ]  
train_val_data5 <- Lands_forest[train_val_idx5, ]  

# STEP 2: Hyperparameter tuning with CV on training set ONLY
# Create NEW spatial folds within the training data
train_val_sf1 <- st_as_sf(train_val_data1, coords = c("x", "y"), crs = 32632)
train_val_sf2 <- st_as_sf(train_val_data2, coords = c("x", "y"), crs = 32632)
train_val_sf3 <- st_as_sf(train_val_data3, coords = c("x", "y"), crs = 32632)
train_val_sf4 <- st_as_sf(train_val_data4, coords = c("x", "y"), crs = 32632)
train_val_sf5 <- st_as_sf(train_val_data5, coords = c("x", "y"), crs = 32632)

# block cv small
set.seed(3)

block_cv_small_train1 <- spatial_block_cv(
  data = train_val_sf1,
  v= 4,
  n = c(8,8)
)

autoplot(block_cv_small_train1)

fold_id_train1 <- extract_folds(block_cv_small_train1, nrow(train_val_data1))
train_val_data1$fold <- fold_id_train1

# NOW check balance
table(train_val_data1$fold, train_val_data1$landslide)

set.seed(2)

block_cv_small_train2 <- spatial_block_cv(
  data = train_val_sf2,
  v= 4,
  n = c(8,8)
)

autoplot(block_cv_small_train2)

fold_id_train2 <- extract_folds(block_cv_small_train2, nrow(train_val_data2))
train_val_data2$fold <- fold_id_train2

# NOW check balance
table(train_val_data2$fold, train_val_data2$landslide)

set.seed(7)

block_cv_small_train3 <- spatial_block_cv(
  data = train_val_sf3,
  v= 4,
  n = c(8,8)
)

autoplot(block_cv_small_train3)

fold_id_train3 <- extract_folds(block_cv_small_train3, nrow(train_val_data3))
train_val_data3$fold <- fold_id_train3

# NOW check balance
table(train_val_data3$fold, train_val_data3$landslide)

set.seed(3)

block_cv_small_train4 <- spatial_block_cv(
  data = train_val_sf4,
  v= 4,
  n = c(8,8)
)

autoplot(block_cv_small_train4)

fold_id_train4 <- extract_folds(block_cv_small_train4, nrow(train_val_data4))
train_val_data4$fold <- fold_id_train4

# NOW check balance
table(train_val_data4$fold, train_val_data4$landslide)

block_cv_small_train5 <- spatial_block_cv(
  data = train_val_sf5,
  v= 4,
  n = c(8,8)
)

autoplot(block_cv_small_train5)

fold_id_train5 <- extract_folds(block_cv_small_train5, nrow(train_val_data5))
train_val_data5$fold <- fold_id_train5

# NOW check balance
table(train_val_data5$fold, train_val_data5$landslide)



# Convert to list format that gpb.grid.search.tune.parameters expects
# Each list element = indices of validation observations for that fold
folds_list1 <- list()
folds_list2 <- list()
folds_list3 <- list()
folds_list4 <- list()
folds_list5 <- list()

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list1[[i]] <- which(fold_id_train1 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list2[[i]] <- which(fold_id_train2 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list3[[i]] <- which(fold_id_train3 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list4[[i]] <- which(fold_id_train4 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list5[[i]] <- which(fold_id_train5 == i)
}



param_grid <- list(
  learning_rate = c(0.05, 0.01), # needs to be low
  min_data_in_leaf = c(10,20, 50), #10-100 is good for generalisation
  num_leaves = c(8,16,32), # 2-8 strong reg, 16-128 - balance
  lambda_l2 = c(10, 50, 100), # 0 - no reg, 1-10 mild, 100 strong but may underfit
  max_depth = c(-1)
) 



other_params <- list(
  objective = "binary",
  verbose = -1
)

# re instate model and data
X_train_all1 <- as.matrix(train_val_data1[, covars_forest])
X_train_all2 <- as.matrix(train_val_data2[, covars_forest])
X_train_all3 <- as.matrix(train_val_data3[, covars_forest])
X_train_all4 <- as.matrix(train_val_data4[, covars_forest])
X_train_all5 <- as.matrix(train_val_data5[, covars_forest])

y_train_all1 <- train_val_data1[, "landslide"]
y_train_all2 <- train_val_data2[, "landslide"]
y_train_all3 <- train_val_data3[, "landslide"]
y_train_all4 <- train_val_data4[, "landslide"]
y_train_all5 <- train_val_data5[, "landslide"]


coords_train_all1 <- as.matrix(train_val_data1[, c("x", "y")])
coords_train_all2 <- as.matrix(train_val_data2[, c("x", "y")])
coords_train_all3 <- as.matrix(train_val_data3[, c("x", "y")])
coords_train_all4 <- as.matrix(train_val_data4[, c("x", "y")])
coords_train_all5 <- as.matrix(train_val_data5[, c("x", "y")])

dtrain_all1 <- gpb.Dataset(data = X_train_all1,
                           label = y_train_all1,
                           categorical_feature = cat_feat_forest)
dtrain_all2 <- gpb.Dataset(data = X_train_all2,
                           label = y_train_all2,
                           categorical_feature = cat_feat_forest)
dtrain_all3 <- gpb.Dataset(data = X_train_all3,
                           label = y_train_all3,
                           categorical_feature = cat_feat_forest)
dtrain_all4 <- gpb.Dataset(data = X_train_all4,
                           label = y_train_all4,
                           categorical_feature = cat_feat_forest)
dtrain_all5 <- gpb.Dataset(data = X_train_all5,
                           label = y_train_all5,
                           categorical_feature = cat_feat_forest)

gp_model_final1 <- GPModel(gp_coords = coords_train_all1, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final2 <- GPModel(gp_coords = coords_train_all2, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final3 <- GPModel(gp_coords = coords_train_all3, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final4 <- GPModel(gp_coords = coords_train_all4, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final5 <- GPModel(gp_coords = coords_train_all5, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)

opt_params1 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list1,  
  data = dtrain_all1,
  gp_model = gp_model_final1,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params2 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list2,  
  data = dtrain_all2,
  gp_model = gp_model_final2,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params3 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list3,  
  data = dtrain_all3,
  gp_model = gp_model_final3,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params4 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list4,  
  data = dtrain_all4,
  gp_model = gp_model_final4,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params5 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list5,  
  data = dtrain_all5,
  gp_model = gp_model_final5,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)


# summarise optimum params and auc

cv_res_sb_v4 = data.frame(Model = "Spatial_smallB_forest_v3",
                       learning_rate = median(c(opt_params1$best_params$learning_rate,
                                                opt_params2$best_params$learning_rate,
                                                opt_params3$best_params$learning_rate,
                                                opt_params4$best_params$learning_rate)),
                                               # opt_params5$best_params$learning_rate)),
                       min_data_in_leaf = median(c(opt_params1$best_params$min_data_in_leaf,
                                                   opt_params2$best_params$min_data_in_leaf,
                                                   opt_params3$best_params$min_data_in_leaf,
                                                   opt_params4$best_params$min_data_in_leaf)),
                                                   #opt_params5$best_params$min_data_in_leaf)),
                       num_leaves = median(c(opt_params1$best_params$num_leaves,
                                             opt_params2$best_params$num_leaves,
                                             opt_params3$best_params$num_leaves,
                                             opt_params4$best_params$num_leaves)),
                                             #opt_params5$best_params$num_leaves)),
                       lambda_l2 = median(c(opt_params1$best_params$lambda_l2,
                                            opt_params2$best_params$lambda_l2,
                                            opt_params3$best_params$lambda_l2,
                                            opt_params4$best_params$lambda_l2)),
                                            #opt_params5$best_params$lambda_l2)),
                       best_iter = mean(c(opt_params1$best_iter,
                                          opt_params2$best_iter,
                                          opt_params3$best_iter,
                                          opt_params4$best_iter)),
                                          #opt_params5$best_iter)),
                       mean_auc_hyp = mean(c(opt_params1$best_score,
                                             opt_params2$best_score,
                                             opt_params3$best_score,
                                             opt_params4$best_score)),
                                             #opt_params5$best_score)),
                       sd_auc_hyp =  sd(c(opt_params1$best_score,
                                          opt_params2$best_score,
                                          opt_params3$best_score,
                                          opt_params4$best_score))
                                         # opt_params5$best_score))
)

# STEP 4: Retrain on ALL training data with best parameters x 5
# Now use ALL of train_val_data
X_train_all1 <- as.matrix(train_val_data1[, covars_forest])
y_train_all1 <- train_val_data1[, "landslide"]
coords_train_all1 <- as.matrix(train_val_data1[, c("x", "y")])

dtrain_all1 <- gpb.Dataset(data = X_train_all1,
                           label = y_train_all1,
                           categorical_feature = cat_feat_forest)

gp_model_final1 <- GPModel(gp_coords = coords_train_all1, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)

final_model1 <- gpboost(
  data = dtrain_all1,
  gp_model = gp_model_final1,
  nrounds = opt_params1$best_iter,
  learning_rate = opt_params1$best_params$learning_rate,
  num_leaves = opt_params1$best_params$num_leaves,
  min_data_in_leaf = opt_params1$best_params$min_data_in_leaf,
  lambda_l2 = opt_params1$best_params$lambda_l2,
  verbose = 1
)

pred_train1 <- predict(final_model1, 
                       data = as.matrix(train_val_data1[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data1[, c("x", "y")]),
                       predict_var = FALSE)

train_auc1 <- auc(roc(y_train_all1, pred_train1$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test1 <- as.matrix(test_data1[, covars_forest])
y_test1 <- test_data1[, "landslide"]
coords_test1 <- as.matrix(test_data1[, c("x", "y")])

test_pred1 <- predict(final_model1, 
                      data = X_test1, 
                      gp_coords_pred = coords_test1)

test_auc1 <- auc(roc(y_test1, test_pred1$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop1 = gp_model_final1$get_cov_pars()[1] / (gp_model_final1$get_cov_pars()[1]+1)


# 1. Pearson residuals 
pearson_residuals1 <- (y_train_all1 - pred_train1$response_mean) / 
  sqrt(pred_train1$response_mean * (1 - pred_train1$response_mean))

# Create spatial neighbors 

coords1 <- cbind(train_val_data1[,"x"], train_val_data1[,"y"])
knn1 <- knearneigh(coords1, k = 12)  # adjust k based on your data
nb1 <- knn2nb(knn1)
weights1 <- nb2listw(nb1, style = "W")

# Test for spatial autocorrelation
moran_test1 <- moran.test(pearson_residuals1, weights1)
print(moran_test1)


# model 2
X_train_all2 <- as.matrix(train_val_data2[, covars_forest])
y_train_all2 <- train_val_data2[, "landslide"]
coords_train_all2 <- as.matrix(train_val_data2[, c("x", "y")])

dtrain_all2 <- gpb.Dataset(data = X_train_all2,
                           label = y_train_all2,
                           categorical_feature = cat_feat_forest)

gp_model_final2 <- GPModel(gp_coords = coords_train_all2, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model2 <- gpboost(
  data = dtrain_all2,
  gp_model = gp_model_final2,
  nrounds = opt_params2$best_iter,
  learning_rate = opt_params2$best_params$learning_rate,
  num_leaves = opt_params2$best_params$num_leaves,
  min_data_in_leaf = opt_params2$best_params$min_data_in_leaf,
  lambda_l2 = opt_params2$best_params$lambda_l2,
  verbose = 1
)

pred_train2 <- predict(final_model2, 
                       data = as.matrix(train_val_data2[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data2[, c("x", "y")]),
                       predict_var = FALSE)

train_auc2 <- auc(roc(y_train_all2, pred_train2$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test2 <- as.matrix(test_data2[, covars_forest])
y_test2 <- test_data2[, "landslide"]
coords_test2 <- as.matrix(test_data2[, c("x", "y")])

test_pred2 <- predict(final_model2, 
                      data = X_test2, 
                      gp_coords_pred = coords_test2)

test_auc2 <- auc(roc(y_test2, test_pred2$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop2 = gp_model_final2$get_cov_pars()[1] / (gp_model_final2$get_cov_pars()[1]+1)

# 2. Pearson residuals 
pearson_residuals2 <- (y_train_all2 - pred_train2$response_mean) / 
  sqrt(pred_train2$response_mean * (1 - pred_train2$response_mean))

# Create spatial neighbors 

coords2 <- cbind(train_val_data2[,"x"], train_val_data2[,"y"])
knn2 <- knearneigh(coords2, k = 12)  # adjust k based on your data
nb2 <- knn2nb(knn2)
weights2 <- nb2listw(nb2, style = "W")

# Test for spatial autocorrelation
moran_test2 <- moran.test(pearson_residuals2, weights2)
print(moran_test2)


# model 3
X_train_all3 <- as.matrix(train_val_data3[, covars_forest])
y_train_all3 <- train_val_data3[, "landslide"]
coords_train_all3 <- as.matrix(train_val_data3[, c("x", "y")])

dtrain_all3 <- gpb.Dataset(data = X_train_all3,
                           label = y_train_all3,
                           categorical_feature = cat_feat_forest)

gp_model_final3 <- GPModel(gp_coords = coords_train_all3, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model3 <- gpboost(
  data = dtrain_all3,
  gp_model = gp_model_final3,
  nrounds = opt_params3$best_iter,
  learning_rate = opt_params3$best_params$learning_rate,
  num_leaves = opt_params3$best_params$num_leaves,
  min_data_in_leaf = opt_params3$best_params$min_data_in_leaf,
  lambda_l2 = opt_params3$best_params$lambda_l2,
  verbose = 1
)

pred_train3 <- predict(final_model3, 
                       data = as.matrix(train_val_data3[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data3[, c("x", "y")]),
                       predict_var = FALSE)

train_auc3 <- auc(roc(y_train_all3, pred_train3$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test3 <- as.matrix(test_data3[, covars_forest])
y_test3 <- test_data3[, "landslide"]
coords_test3 <- as.matrix(test_data3[, c("x", "y")])

test_pred3 <- predict(final_model3, 
                      data = X_test3, 
                      gp_coords_pred = coords_test3)

test_auc3 <- auc(roc(y_test3, test_pred3$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop3 = gp_model_final3$get_cov_pars()[1] / (gp_model_final3$get_cov_pars()[1]+1)


# 3. Pearson residuals 
pearson_residuals3 <- (y_train_all3 - pred_train3$response_mean) / 
  sqrt(pred_train3$response_mean * (1 - pred_train3$response_mean))

# Create spatial neighbors 

coords3 <- cbind(train_val_data3[,"x"], train_val_data3[,"y"])
knn3 <- knearneigh(coords3, k = 12)  # adjust k based on your data
nb3 <- knn2nb(knn3)
weights3 <- nb2listw(nb3, style = "W")

# Test for spatial autocorrelation
moran_test3 <- moran.test(pearson_residuals3, weights3)
print(moran_test3)

# model 4
X_train_all4 <- as.matrix(train_val_data4[, covars_forest])
y_train_all4 <- train_val_data4[, "landslide"]
coords_train_all4 <- as.matrix(train_val_data4[, c("x", "y")])

dtrain_all4 <- gpb.Dataset(data = X_train_all4,
                           label = y_train_all4,
                           categorical_feature = cat_feat_forest)

gp_model_final4 <- GPModel(gp_coords = coords_train_all4, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model4 <- gpboost(
  data = dtrain_all4,
  gp_model = gp_model_final4,
  nrounds = opt_params4$best_iter,
  learning_rate = opt_params4$best_params$learning_rate,
  num_leaves = opt_params4$best_params$num_leaves,
  min_data_in_leaf = opt_params4$best_params$min_data_in_leaf,
  lambda_l2 = opt_params4$best_params$lambda_l2,
  verbose = 1
)

pred_train4 <- predict(final_model4, 
                       data = as.matrix(train_val_data4[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data4[, c("x", "y")]),
                       predict_var = FALSE)

train_auc4 <- auc(roc(y_train_all4, pred_train4$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test4 <- as.matrix(test_data4[, covars_forest])
y_test4 <- test_data4[, "landslide"]
coords_test4 <- as.matrix(test_data4[, c("x", "y")])

test_pred4 <- predict(final_model4, 
                      data = X_test4, 
                      gp_coords_pred = coords_test4)

test_auc4 <- auc(roc(y_test4, test_pred4$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop4 = gp_model_final4$get_cov_pars()[1] / (gp_model_final4$get_cov_pars()[1]+1)

# 4. Pearson residuals 
pearson_residuals4 <- (y_train_all4 - pred_train4$response_mean) / 
  sqrt(pred_train4$response_mean * (1 - pred_train4$response_mean))

# Create spatial neighbors 

coords4 <- cbind(train_val_data4[,"x"], train_val_data4[,"y"])
knn4 <- knearneigh(coords4, k = 12)  # adjust k based on your data
nb4 <- knn2nb(knn4)
weights4 <- nb2listw(nb4, style = "W")

# Test for spatial autocorrelation
moran_test4 <- moran.test(pearson_residuals4, weights4)
print(moran_test4)

# model 5
X_train_all5 <- as.matrix(train_val_data5[, covars_forest])
y_train_all5 <- train_val_data5[, "landslide"]
coords_train_all5 <- as.matrix(train_val_data5[, c("x", "y")])

dtrain_all5 <- gpb.Dataset(data = X_train_all5,
                           label = y_train_all5,
                           categorical_feature = cat_feat_forest)

gp_model_final5 <- GPModel(gp_coords = coords_train_all5, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model5 <- gpboost(
  data = dtrain_all5,
  gp_model = gp_model_final5,
  nrounds = opt_params5$best_iter,
  learning_rate = opt_params5$best_params$learning_rate,
  num_leaves = opt_params5$best_params$num_leaves,
  min_data_in_leaf = opt_params5$best_params$min_data_in_leaf,
  lambda_l2 = opt_params5$best_params$lambda_l2,
  verbose = 1
)

pred_train5 <- predict(final_model5, 
                       data = as.matrix(train_val_data5[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data5[, c("x", "y")]),
                       predict_var = FALSE)

train_auc5 <- auc(roc(y_train_all5, pred_train5$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test5 <- as.matrix(test_data5[, covars_forest])
y_test5 <- test_data5[, "landslide"]
coords_test5 <- as.matrix(test_data5[, c("x", "y")])

test_pred5 <- predict(final_model5, 
                      data = X_test5, 
                      gp_coords_pred = coords_test5)

test_auc5 <- auc(roc(y_test5, test_pred5$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop5 = gp_model_final5$get_cov_pars()[1] / (gp_model_final5$get_cov_pars()[1]+1)

# 5. Pearson residuals 
pearson_residuals5 <- (y_train_all5 - pred_train5$response_mean) / 
  sqrt(pred_train5$response_mean * (1 - pred_train5$response_mean))

# Create spatial neighbors 

coords5 <- cbind(train_val_data5[,"x"], train_val_data5[,"y"])
knn5 <- knearneigh(coords5, k = 12)  # adjust k based on your data
nb5 <- knn2nb(knn5)
weights5 <- nb2listw(nb5, style = "W")

# Test for spatial autocorrelation
moran_test5 <- moran.test(pearson_residuals5, weights5)
print(moran_test5)

cv_res_sb_v4 = cv_res_sb_v4 |> 
  mutate(mean_train_auc = mean(c(train_auc1,train_auc2,train_auc3,train_auc4)),
         sd_train_auc = sd(c(train_auc1,train_auc2,train_auc3,train_auc4)),
         mean_test_auc = mean(c(test_auc1, test_auc2, test_auc3, test_auc4)),
         sd_test_auc = sd(c(test_auc1, test_auc2, test_auc3, test_auc4)),
         mean_space = mean(c(space_prop1, space_prop2, space_prop3, space_prop4)),
         low_moran = min(c(moran_test1$estimate[[1]],
                           moran_test2$estimate[[1]],
                           moran_test3$estimate[[1]],
                           moran_test4$estimate[[1]])),
                           #moran_test5$estimate[[1]])),
         hi_moran = max(c(moran_test1$estimate[[1]],
                          moran_test2$estimate[[1]],
                          moran_test3$estimate[[1]],
                          moran_test4$estimate[[1]])),
                          #moran_test5$estimate[[1]])),
         low_p = min(c(moran_test1$p.value[[1]],
                       moran_test2$p.value[[1]],
                       moran_test3$p.value[[1]],
                       moran_test4$p.value[[1]])),
                      # moran_test5$p.value[[1]])),
         hi_p = max(c(moran_test1$p.value[[1]],
                      moran_test2$p.value[[1]],
                      moran_test3$p.value[[1]],
                      moran_test4$p.value[[1]])))
                     # moran_test5$p.value[[1]])))

## final global model on full data set with median hyperparameters
X_all <- as.matrix(Lands_forest[, covars_forest])
y_all <- Lands_forest[, "landslide"]
coords_all <- as.matrix(Lands_forest[, c("x", "y")])

d_all <- gpb.Dataset(data = X_all,
                     label = y_all,
                     categorical_feature = cat_feat_forest)

gp_model_final_all <- GPModel(gp_coords = coords_all, 
                              likelihood = "bernoulli_probit",
                              cov_function = "matern" ,
                              cov_fct_shape = 1.5)


# final model
final_model_all <- gpboost(
  data = d_all,
  gp_model = gp_model_final_all,
  nrounds = round(cv_res_sb_v4$best_iter, 0),#cvbst$best_iter, #
  learning_rate = cv_res_sb_v4$learning_rate,
  min_data_in_leaf = cv_res_sb_v4$min_data_in_leaf,
  num_leaves = cv_res_sb_v4$num_leaves,
  lambda_l2 = cv_res_sb_v4$lambda_l2,
  verbose = 1
)

# load model if needed
#final_model_all = readRDS.gpb.Booster("saved models/Sp_SmallB_forest_final.rds")

pred_all <- predict(final_model_all, 
                    data = as.matrix(Lands_forest[, covars_forest]),
                    gp_coords_pred = as.matrix(Lands_forest[, c("x", "y")]),
                    predict_var = FALSE)

all_auc <- auc(roc(y_all, pred_all$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop_all = gp_model_final_all$get_cov_pars()[1] / (gp_model_final_all$get_cov_pars()[1]+1)

# 5. Pearson residuals 
pearson_residuals_all <- (y_all - pred_all$response_mean) / 
  sqrt(pred_all$response_mean * (1 - pred_all$response_mean))

# Create spatial neighbors 

coords_all <- cbind(Lands_forest[,"x"], Lands_forest[,"y"])
knn_all <- knearneigh(coords_all, k = 12)  # adjust k based on your data
nb_all <- knn2nb(knn_all)
weights_all <- nb2listw(nb_all, style = "W")

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)


library(SHAPforxgboost)
shap_values_forest <- shap.values(xgb_model = final_model_all, 
                                X_train = X_all)
shap.plot.summary.wrap1(final_model_all, X = X_all) + ggtitle("SHAP values")
shap_long <- shap.prep(final_model_all, X_train = X_all)
fig_list <- lapply(names(shap_values_forest$mean_shap_score)[1:9], 
                   shap.plot.dependence, data_long = shap_long)
gridExtra::grid.arrange(grobs = fig_list, ncol = 3)

saveRDS.gpb.Booster(final_model_all, "saved models/Sp_SmallB_forest_final.rds")


## non spatial version ####
# already have all the folds etc
opt_params1_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list1,  
  data = dtrain_all1,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params2_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list2,  
  data = dtrain_all2,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params3_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list3,  
  data = dtrain_all3,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params4_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list4,  
  data = dtrain_all4,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params5_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list5,  
  data = dtrain_all5,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

cv_res_sb_ns_v4 = data.frame(Model = "Non_Spatial_smallB_forest_v4",
                          learning_rate = median(c(opt_params1_ns$best_params$learning_rate,
                                                   opt_params2_ns$best_params$learning_rate,
                                                   opt_params3_ns$best_params$learning_rate,
                                                   opt_params4_ns$best_params$learning_rate)),
                                                   #opt_params5_ns$best_params$learning_rate)),
                          min_data_in_leaf = median(c(opt_params1_ns$best_params$min_data_in_leaf,
                                                      opt_params2_ns$best_params$min_data_in_leaf,
                                                      opt_params3_ns$best_params$min_data_in_leaf,
                                                      opt_params4_ns$best_params$min_data_in_leaf)),
                                                     # opt_params5_ns$best_params$min_data_in_leaf)),
                          num_leaves = median(c(opt_params1_ns$best_params$num_leaves,
                                                opt_params2_ns$best_params$num_leaves,
                                                opt_params3_ns$best_params$num_leaves,
                                                opt_params4_ns$best_params$num_leaves)),
                                                #opt_params5_ns$best_params$num_leaves)),
                          lambda_l2 = median(c(opt_params1_ns$best_params$lambda_l2,
                                               opt_params2_ns$best_params$lambda_l2,
                                               opt_params3_ns$best_params$lambda_l2,
                                               opt_params4_ns$best_params$lambda_l2)),
                                              # opt_params5_ns$best_params$lambda_l2)),
                          best_iter = mean(c(opt_params1_ns$best_iter,
                                             opt_params2_ns$best_iter,
                                             opt_params3_ns$best_iter,
                                             opt_params4_ns$best_iter)),
                                             #opt_params5_ns$best_iter)),
                          mean_auc_hyp = mean(c(opt_params1_ns$best_score,
                                                opt_params2_ns$best_score,
                                                opt_params3_ns$best_score,
                                                opt_params4_ns$best_score)),
                                               # opt_params5_ns$best_score)),
                          sd_auc_hyp =  sd(c(opt_params1_ns$best_score,
                                             opt_params2_ns$best_score,
                                             opt_params3_ns$best_score,
                                             opt_params4_ns$best_score))
                                            # opt_params5_ns$best_score))
)

# fit models with best parameters - non_spatial ####
final_model1_ns <- gpboost(
  data = dtrain_all1,
  gp_model = NULL,
  nrounds = opt_params1_ns$best_iter,
  learning_rate = opt_params1_ns$best_params$learning_rate,
  num_leaves = opt_params1_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params1_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params1_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train1_ns <- predict(final_model1_ns, 
                          data = as.matrix(train_val_data1[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data1[, c("x", "y")]),
                          predict_var = FALSE)

train_auc1_ns <- auc(roc(y_train_all1, pred_train1_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred1_ns <- predict(final_model1_ns, 
                         data = X_test1, 
                         gp_coords_pred = coords_test1)

test_auc1_ns <- auc(roc(y_test1, test_pred1_ns))

library(spdep)
library(sf)

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe1 <- pmax(pmin(pred_train1_ns, 1 - epsilon), epsilon)
pearson_residuals1_ns <- (y_train_all1 - y_pred_prob_safe1) / 
  sqrt(y_pred_prob_safe1 * (1 - y_pred_prob_safe1))

# Test for spatial autocorrelation
moran_test1_ns <- moran.test(pearson_residuals1_ns, weights1)
print(moran_test1_ns)


# model 2
final_model2_ns <- gpboost(
  data = dtrain_all2,
  gp_model = NULL,
  nrounds = opt_params2_ns$best_iter,
  learning_rate = opt_params2_ns$best_params$learning_rate,
  num_leaves = opt_params2_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params2_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params2_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train2_ns <- predict(final_model2_ns, 
                          data = as.matrix(train_val_data2[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data2[, c("x", "y")]),
                          predict_var = FALSE)

train_auc2_ns <- auc(roc(y_train_all2, pred_train2_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred2_ns <- predict(final_model2_ns, 
                         data = X_test2, 
                         gp_coords_pred = coords_test2)

test_auc2_ns <- auc(roc(y_test2, test_pred2_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe2 <- pmax(pmin(pred_train2_ns, 1 - epsilon), epsilon)
pearson_residuals2_ns <- (y_train_all2 - y_pred_prob_safe2) / 
  sqrt(y_pred_prob_safe2 * (1 - y_pred_prob_safe2))

# Test for spatial autocorrelation
moran_test2_ns <- moran.test(pearson_residuals2_ns, weights2)
print(moran_test2_ns)

# model 3
final_model3_ns <- gpboost(
  data = dtrain_all3,
  gp_model = NULL,
  nrounds = opt_params3_ns$best_iter,
  learning_rate = opt_params3_ns$best_params$learning_rate,
  num_leaves = opt_params3_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params3_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params3_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train3_ns <- predict(final_model3_ns, 
                          data = as.matrix(train_val_data3[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data3[, c("x", "y")]),
                          predict_var = FALSE)

train_auc3_ns <- auc(roc(y_train_all3, pred_train3_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred3_ns <- predict(final_model3_ns, 
                         data = X_test3, 
                         gp_coords_pred = coords_test3)

test_auc3_ns <- auc(roc(y_test3, test_pred3_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe3 <- pmax(pmin(pred_train3_ns, 1 - epsilon), epsilon)
pearson_residuals3_ns <- (y_train_all3 - y_pred_prob_safe3) / 
  sqrt(y_pred_prob_safe3 * (1 - y_pred_prob_safe3))

# Test for spatial autocorrelation
moran_test3_ns <- moran.test(pearson_residuals3_ns, weights3)
print(moran_test3_ns)

# model 4
final_model4_ns <- gpboost(
  data = dtrain_all4,
  gp_model = NULL,
  nrounds = opt_params4_ns$best_iter,
  learning_rate = opt_params4_ns$best_params$learning_rate,
  num_leaves = opt_params4_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params4_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params4_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train4_ns <- predict(final_model4_ns, 
                          data = as.matrix(train_val_data4[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data4[, c("x", "y")]),
                          predict_var = FALSE)

train_auc4_ns <- auc(roc(y_train_all4, pred_train4_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred4_ns <- predict(final_model4_ns, 
                         data = X_test4, 
                         gp_coords_pred = coords_test4)

test_auc4_ns <- auc(roc(y_test4, test_pred4_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe4 <- pmax(pmin(pred_train4_ns, 1 - epsilon), epsilon)
pearson_residuals4_ns <- (y_train_all4 - y_pred_prob_safe4) / 
  sqrt(y_pred_prob_safe4 * (1 - y_pred_prob_safe4))

# Test for spatial autocorrelation
moran_test4_ns <- moran.test(pearson_residuals4_ns, weights4)
print(moran_test4_ns)

# model 5
final_model5_ns <- gpboost(
  data = dtrain_all5,
  gp_model = NULL,
  nrounds = opt_params5_ns$best_iter,
  learning_rate = opt_params5_ns$best_params$learning_rate,
  num_leaves = opt_params5_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params5_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params5_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train5_ns <- predict(final_model5_ns, 
                          data = as.matrix(train_val_data5[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data5[, c("x", "y")]),
                          predict_var = FALSE)

train_auc5_ns <- auc(roc(y_train_all5, pred_train5_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred5_ns <- predict(final_model5_ns, 
                         data = X_test5, 
                         gp_coords_pred = coords_test5)

test_auc5_ns <- auc(roc(y_test5, test_pred5_ns))


# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe5 <- pmax(pmin(pred_train5_ns, 1 - epsilon), epsilon)
pearson_residuals5_ns <- (y_train_all5 - y_pred_prob_safe5) / 
  sqrt(y_pred_prob_safe5 * (1 - y_pred_prob_safe5))

# Test for spatial autocorrelation
moran_test5_ns <- moran.test(pearson_residuals5_ns, weights5)
print(moran_test5_ns)

cv_res_sb_ns_v4 = cv_res_sb_ns_v4 |> 
  mutate(mean_train_auc = mean(c(train_auc1_ns,train_auc2_ns,train_auc3_ns,train_auc4_ns)),
         sd_train_auc = sd(c(train_auc1_ns,train_auc2_ns,train_auc3_ns,train_auc4_ns)),
         mean_test_auc = mean(c(test_auc1_ns, test_auc2_ns, test_auc3_ns, test_auc4_ns)),
         sd_test_auc = sd(c(test_auc1_ns, test_auc2_ns, test_auc3_ns, test_auc4_ns)),
         mean_space = "NA",
         low_moran = min(c(moran_test1_ns$estimate[[1]],
                           moran_test2_ns$estimate[[1]],
                           moran_test3_ns$estimate[[1]],
                           moran_test4_ns$estimate[[1]])),
                           #moran_test5_ns$estimate[[1]])),
         hi_moran = max(c(moran_test1_ns$estimate[[1]],
                          moran_test2_ns$estimate[[1]],
                          moran_test3_ns$estimate[[1]],
                          moran_test4_ns$estimate[[1]])),
                          #moran_test5_ns$estimate[[1]])),
         low_p = min(c(moran_test1_ns$p.value[[1]],
                       moran_test2_ns$p.value[[1]],
                       moran_test3_ns$p.value[[1]],
                       moran_test4_ns$p.value[[1]])),
                      # moran_test5_ns$p.value[[1]])),
         hi_p = max(c(moran_test1_ns$p.value[[1]],
                      moran_test2_ns$p.value[[1]],
                      moran_test3_ns$p.value[[1]],
                      moran_test4_ns$p.value[[1]])))
                     # moran_test5_ns$p.value[[1]])))

cv_res = rbind(cv_res, cv_res_sb_ns_v4)

write.csv(cv_res, "data/Nested_CV_results_forest.csv")


# cv_res = read.csv("data/Nested_CV_results_forest.csv")

cv_res_v2 = cv_res[10,]

# final model
final_model_all <- gpboost(
  data = d_all,
  gp_model = NULL,
  nrounds = round(cv_res_v2$best_iter,0),
  learning_rate = cv_res_v2$learning_rate,
  num_leaves = cv_res_v2$num_leaves,
  min_data_in_leaf = cv_res_v2$min_data_in_leaf,
  lambda_l2 = cv_res_v2$lambda_l2,
  verbose = 1
)

saveRDS.gpb.Booster(final_model_all, "saved models/Non_Sp_SmallB_forest_final.rds")

pred_all <- predict(final_model_all, 
                    data = as.matrix(Lands_forest[, covars_forest]),
                    gp_coords_pred = as.matrix(Lands_forest[, c("x", "y")]),
                    predict_var = FALSE)

all_auc <- auc(roc(y_all, pred_all$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop_all = gp_model_final_all$get_cov_pars()[1] / (gp_model_final_all$get_cov_pars()[1]+1)

# 5. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe_all <- pmax(pmin(pred_all, 1 - epsilon), epsilon)
pearson_residuals_all <- (y_all - y_pred_prob_safe_all) / 
  sqrt(y_pred_prob_safe_all * (1 - y_pred_prob_safe_all))

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)

# Create spatial neighbors 

coords_all <- cbind(Lands_main[,"x"], Lands_main[,"y"])
knn_all <- knearneigh(coords_all, k = 12)  # adjust k based on your data
nb_all <- knn2nb(knn_all)
weights_all <- nb2listw(nb_all, style = "W")

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)

#### same again but with random CV ####
library(caret)

# Create random folds with caret
set.seed(123)
random_folds <- createFolds(sf_Lands_forest$landslide, 
                            k = 5, 
                            returnTrain = FALSE)

# Convert to blockCV format (fold_ids vector)
n <- nrow(sf_Lands_forest)
fold_id <- integer(n)  # Initialize vector

for (i in 1:length(random_folds)) {
  fold_id[random_folds[[i]]] <- i
}

# Now create a list object similar to blockCV output
test_blocks <- list(
  folds_id = fold_id,
  k = 5,
  records = table(fold_id)
)

# Verify it worked
print(table(fold_id))  # Should show roughly equal distribution across 5 folds

Lands_forest$fold <- fold_id

# Check balance 
for(fold in 1:5) {
  cat(paste("\nFold", fold, "balance:\n"))
  val_idx <- which(Lands_forest$fold == fold)
  print(table(Lands_forest$landslide[val_idx]))
}


# spatial inner loop ####
# Assign one block as test set, one at a time
test_idx1 <- which(Lands_forest$fold == 1)  
test_idx2 <- which(Lands_forest$fold == 2)  
test_idx3 <- which(Lands_forest$fold == 3)  
test_idx4 <- which(Lands_forest$fold == 4)  
test_idx5 <- which(Lands_forest$fold == 5)  

train_val_idx1 <- which(Lands_forest$fold != 1)  
train_val_idx2 <- which(Lands_forest$fold != 2)  
train_val_idx3 <- which(Lands_forest$fold != 3)  
train_val_idx4 <- which(Lands_forest$fold != 4)  
train_val_idx5 <- which(Lands_forest$fold != 5)  

# Lock away the test set
test_data1 <- Lands_forest[test_idx1, ]
test_data2 <- Lands_forest[test_idx2, ]
test_data3 <- Lands_forest[test_idx3, ]
test_data4 <- Lands_forest[test_idx4, ]
test_data5 <- Lands_forest[test_idx5, ]


train_val_data1 <- Lands_forest[train_val_idx1, ]  
train_val_data2 <- Lands_forest[train_val_idx2, ]  
train_val_data3 <- Lands_forest[train_val_idx3, ]  
train_val_data4 <- Lands_forest[train_val_idx4, ]  
train_val_data5 <- Lands_forest[train_val_idx5, ]  

# STEP 2: Hyperparameter tuning with CV on training set ONLY
# Create NEW spatial folds within the training data
train_val_sf1 <- st_as_sf(train_val_data1, coords = c("x", "y"), crs = 32632)
train_val_sf2 <- st_as_sf(train_val_data2, coords = c("x", "y"), crs = 32632)
train_val_sf3 <- st_as_sf(train_val_data3, coords = c("x", "y"), crs = 32632)
train_val_sf4 <- st_as_sf(train_val_data4, coords = c("x", "y"), crs = 32632)
train_val_sf5 <- st_as_sf(train_val_data5, coords = c("x", "y"), crs = 32632)

# random
set.seed(124)
random_cv_folds1 <- createFolds(train_val_sf1$landslide, 
                                k = 5, 
                                returnTrain = FALSE)

# Convert to blockCV format (fold_ids vector)
n1 <- nrow(train_val_sf1)
fold_id_train1 <- integer(n1)  # Initialize vector

for (i in 1:length(random_cv_folds1)) {
  fold_id_train1[random_cv_folds1[[i]]] <- i
}

# Now create a list object similar to blockCV output
cv_folds1 <- list(
  fold_id_train1 = fold_id_train1,
  k = 5,
  records = table(fold_id_train1)
)

# Verify it worked
print(table(fold_id_train1))  # Should show roughly equal distribution across 5 folds

train_val_data1$fold <- fold_id_train1

table(train_val_data1$fold, train_val_data1$landslide)

# fold 2
random_cv_folds2 <- createFolds(train_val_sf2$landslide, 
                                k = 5, 
                                returnTrain = FALSE)

# Convert to blockCV format (fold_ids vector)
n2 <- nrow(train_val_sf2)
fold_id_train2 <- integer(n2)  # Initialize vector

for (i in 1:length(random_cv_folds2)) {
  fold_id_train2[random_cv_folds2[[i]]] <- i
}

# Now create a list object similar to blockCV output
cv_folds2 <- list(
  fold_id_train2 = fold_id_train2,
  k = 5,
  records = table(fold_id_train2)
)

# Verify it worked
print(table(fold_id_train2))  # Should show roughly equal distribution across 5 folds

train_val_data2$fold <- fold_id_train2

table(train_val_data2$fold, train_val_data2$landslide)


# fold 3
random_cv_folds3 <- createFolds(train_val_sf3$landslide, 
                                k = 5, 
                                returnTrain = FALSE)

# Convert to blockCV format (fold_ids vector)
n3 <- nrow(train_val_sf3)
fold_id_train3 <- integer(n3)  # Initialize vector

for (i in 1:length(random_cv_folds3)) {
  fold_id_train3[random_cv_folds3[[i]]] <- i
}

# Now create a list object similar to blockCV output
cv_folds3 <- list(
  fold_id_train3 = fold_id_train3,
  k = 5,
  records = table(fold_id_train3)
)

# Verify it worked
print(table(fold_id_train3))  # Should show roughly equal distribution across 5 folds

train_val_data3$fold <- fold_id_train3

table(train_val_data3$fold, train_val_data3$landslide)


# fold 4
random_cv_folds4 <- createFolds(train_val_sf4$landslide, 
                                k = 5, 
                                returnTrain = FALSE)

# Convert to blockCV format (fold_ids vector)
n4 <- nrow(train_val_sf4)
fold_id_train4 <- integer(n4)  # Initialize vector

for (i in 1:length(random_cv_folds4)) {
  fold_id_train4[random_cv_folds4[[i]]] <- i
}

# Now create a list object similar to blockCV output
cv_folds4 <- list(
  fold_id_train4 = fold_id_train4,
  k = 5,
  records = table(fold_id_train4)
)

# Verify it worked
print(table(fold_id_train4))  # Should show roughly equal distribution across 5 folds

train_val_data4$fold <- fold_id_train4

table(train_val_data4$fold, train_val_data4$landslide)

# fold 5
random_cv_folds5 <- createFolds(train_val_sf5$landslide, 
                                k = 5, 
                                returnTrain = FALSE)

# Convert to blockCV format (fold_ids vector)
n5 <- nrow(train_val_sf5)
fold_id_train5 <- integer(n5)  # Initialize vector

for (i in 1:length(random_cv_folds5)) {
  fold_id_train5[random_cv_folds5[[i]]] <- i
}

# Now create a list object similar to blockCV output
cv_folds5 <- list(
  fold_id_train5 = fold_id_train5,
  k = 5,
  records = table(fold_id_train5)
)

# Verify it worked
print(table(fold_id_train5))  # Should show roughly equal distribution across 5 folds

train_val_data5$fold <- fold_id_train5

table(train_val_data5$fold, train_val_data5$landslide)


# Convert to list format that gpb.grid.search.tune.parameters expects
# Each list element = indices of validation observations for that fold
folds_list1 <- list()
folds_list2 <- list()
folds_list3 <- list()
folds_list4 <- list()
folds_list5 <- list()

for (i in 1:5) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list1[[i]] <- which(fold_id_train1 == i)
}

for (i in 1:5) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list2[[i]] <- which(fold_id_train2 == i)
}

for (i in 1:5) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list3[[i]] <- which(fold_id_train3 == i)
}

for (i in 1:5) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list4[[i]] <- which(fold_id_train4 == i)
}

for (i in 1:5) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list5[[i]] <- which(fold_id_train5 == i)
}



param_grid <- list(
  learning_rate = c(0.05,0.01), # needs to be low
  min_data_in_leaf = c(10, 20, 50), #10-100 is good for generalisation
  num_leaves = c(8,16,32), # 2-8 strong reg, 16-128 - balance
  lambda_l2 = c(10, 50, 100), # 0 - no reg, 1-10 mild, 100 strong but may underfit
  max_depth = c(-1)
) 



other_params <- list(
  objective = "binary",
  verbose = -1
)

# re instate model and data
X_train_all1 <- as.matrix(train_val_data1[, covars_forest])
X_train_all2 <- as.matrix(train_val_data2[, covars_forest])
X_train_all3 <- as.matrix(train_val_data3[, covars_forest])
X_train_all4 <- as.matrix(train_val_data4[, covars_forest])
X_train_all5 <- as.matrix(train_val_data5[, covars_forest])

y_train_all1 <- train_val_data1[, "landslide"]
y_train_all2 <- train_val_data2[, "landslide"]
y_train_all3 <- train_val_data3[, "landslide"]
y_train_all4 <- train_val_data4[, "landslide"]
y_train_all5 <- train_val_data5[, "landslide"]


coords_train_all1 <- as.matrix(train_val_data1[, c("x", "y")])
coords_train_all2 <- as.matrix(train_val_data2[, c("x", "y")])
coords_train_all3 <- as.matrix(train_val_data3[, c("x", "y")])
coords_train_all4 <- as.matrix(train_val_data4[, c("x", "y")])
coords_train_all5 <- as.matrix(train_val_data5[, c("x", "y")])

dtrain_all1 <- gpb.Dataset(data = X_train_all1,
                           label = y_train_all1,
                           categorical_feature = cat_feat_forest)
dtrain_all2 <- gpb.Dataset(data = X_train_all2,
                           label = y_train_all2,
                           categorical_feature = cat_feat_forest)
dtrain_all3 <- gpb.Dataset(data = X_train_all3,
                           label = y_train_all3,
                           categorical_feature = cat_feat_forest)
dtrain_all4 <- gpb.Dataset(data = X_train_all4,
                           label = y_train_all4,
                           categorical_feature = cat_feat_forest)
dtrain_all5 <- gpb.Dataset(data = X_train_all5,
                           label = y_train_all5,
                           categorical_feature = cat_feat_forest)

gp_model_final1 <- GPModel(gp_coords = coords_train_all1, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final2 <- GPModel(gp_coords = coords_train_all2, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final3 <- GPModel(gp_coords = coords_train_all3, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final4 <- GPModel(gp_coords = coords_train_all4, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final5 <- GPModel(gp_coords = coords_train_all5, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)

opt_params1 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list1,  
  data = dtrain_all1,
  gp_model = gp_model_final1,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = F 
)

opt_params2 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list2,  
  data = dtrain_all2,
  gp_model = gp_model_final2,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = F 
)

opt_params3 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list3,  
  data = dtrain_all3,
  gp_model = gp_model_final3,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = F 
)

opt_params4 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list4,  
  data = dtrain_all4,
  gp_model = gp_model_final4,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = F 
)

opt_params5 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list5,  
  data = dtrain_all5,
  gp_model = gp_model_final5,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = F 
)


# summarise optimum params and auc

cv_res_ran = data.frame(Model = "Spatial_random_forest",
                        learning_rate = median(c(opt_params1$best_params$learning_rate,
                                                 opt_params2$best_params$learning_rate,
                                                 opt_params3$best_params$learning_rate,
                                                 opt_params4$best_params$learning_rate,
                                                 opt_params5$best_params$learning_rate)),
                        min_data_in_leaf = median(c(opt_params1$best_params$min_data_in_leaf,
                                                    opt_params2$best_params$min_data_in_leaf,
                                                    opt_params3$best_params$min_data_in_leaf,
                                                    opt_params4$best_params$min_data_in_leaf,
                                                    opt_params5$best_params$min_data_in_leaf)),
                        num_leaves = median(c(opt_params1$best_params$num_leaves,
                                              opt_params2$best_params$num_leaves,
                                              opt_params3$best_params$num_leaves,
                                              opt_params4$best_params$num_leaves,
                                              opt_params5$best_params$num_leaves)),
                        lambda_l2 = median(c(opt_params1$best_params$lambda_l2,
                                             opt_params2$best_params$lambda_l2,
                                             opt_params3$best_params$lambda_l2,
                                             opt_params4$best_params$lambda_l2,
                                             opt_params5$best_params$lambda_l2)),
                        best_iter = mean(c(opt_params1$best_iter,
                                           opt_params2$best_iter,
                                           opt_params3$best_iter,
                                           opt_params4$best_iter,
                                           opt_params5$best_iter)),
                        mean_auc_hyp = mean(c(opt_params1$best_score,
                                              opt_params2$best_score,
                                              opt_params3$best_score,
                                              opt_params4$best_score,
                                              opt_params5$best_score)),
                        sd_auc_hyp =  sd(c(opt_params1$best_score,
                                           opt_params2$best_score,
                                           opt_params3$best_score,
                                           opt_params4$best_score,
                                           opt_params5$best_score))
)

# STEP 4: Retrain on ALL training data with best parameters x 5
# Now use ALL of train_val_data
X_train_all1 <- as.matrix(train_val_data1[, covars_forest])
y_train_all1 <- train_val_data1[, "landslide"]
coords_train_all1 <- as.matrix(train_val_data1[, c("x", "y")])

dtrain_all1 <- gpb.Dataset(data = X_train_all1,
                           label = y_train_all1,
                           categorical_feature = cat_feat_forest)

gp_model_final1 <- GPModel(gp_coords = coords_train_all1, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)

final_model1 <- gpboost(
  data = dtrain_all1,
  gp_model = gp_model_final1,
  nrounds = opt_params1$best_iter,
  learning_rate = opt_params1$best_params$learning_rate,
  num_leaves = opt_params1$best_params$num_leaves,
  min_data_in_leaf = opt_params1$best_params$min_data_in_leaf,
  lambda_l2 = opt_params1$best_params$lambda_l2,
  verbose = 1
)

pred_train1 <- predict(final_model1, 
                       data = as.matrix(train_val_data1[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data1[, c("x", "y")]),
                       predict_var = FALSE)

train_auc1 <- auc(roc(y_train_all1, pred_train1$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test1 <- as.matrix(test_data1[, covars_forest])
y_test1 <- test_data1[, "landslide"]
coords_test1 <- as.matrix(test_data1[, c("x", "y")])

test_pred1 <- predict(final_model1, 
                      data = X_test1, 
                      gp_coords_pred = coords_test1)

test_auc1 <- auc(roc(y_test1, test_pred1$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop1 = gp_model_final1$get_cov_pars()[1] / (gp_model_final1$get_cov_pars()[1]+1)


# 1. Pearson residuals 
pearson_residuals1 <- (y_train_all1 - pred_train1$response_mean) / 
  sqrt(pred_train1$response_mean * (1 - pred_train1$response_mean))

# Create spatial neighbors 

coords1 <- cbind(train_val_data1[,"x"], train_val_data1[,"y"])
knn1 <- knearneigh(coords1, k = 12)  # adjust k based on your data
nb1 <- knn2nb(knn1)
weights1 <- nb2listw(nb1, style = "W")

# Test for spatial autocorrelation
moran_test1 <- moran.test(pearson_residuals1, weights1)
print(moran_test1)


# model 2
X_train_all2 <- as.matrix(train_val_data2[, covars_forest])
y_train_all2 <- train_val_data2[, "landslide"]
coords_train_all2 <- as.matrix(train_val_data2[, c("x", "y")])

dtrain_all2 <- gpb.Dataset(data = X_train_all2,
                           label = y_train_all2,
                           categorical_feature = cat_feat_forest)

gp_model_final2 <- GPModel(gp_coords = coords_train_all2, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model2 <- gpboost(
  data = dtrain_all2,
  gp_model = gp_model_final2,
  nrounds = opt_params2$best_iter,
  learning_rate = opt_params2$best_params$learning_rate,
  num_leaves = opt_params2$best_params$num_leaves,
  min_data_in_leaf = opt_params2$best_params$min_data_in_leaf,
  lambda_l2 = opt_params2$best_params$lambda_l2,
  verbose = 1
)

pred_train2 <- predict(final_model2, 
                       data = as.matrix(train_val_data2[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data2[, c("x", "y")]),
                       predict_var = FALSE)

train_auc2 <- auc(roc(y_train_all2, pred_train2$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test2 <- as.matrix(test_data2[, covars_forest])
y_test2 <- test_data2[, "landslide"]
coords_test2 <- as.matrix(test_data2[, c("x", "y")])

test_pred2 <- predict(final_model2, 
                      data = X_test2, 
                      gp_coords_pred = coords_test2)

test_auc2 <- auc(roc(y_test2, test_pred2$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop2 = gp_model_final2$get_cov_pars()[1] / (gp_model_final2$get_cov_pars()[1]+1)

# 2. Pearson residuals 
pearson_residuals2 <- (y_train_all2 - pred_train2$response_mean) / 
  sqrt(pred_train2$response_mean * (1 - pred_train2$response_mean))

# Create spatial neighbors 

coords2 <- cbind(train_val_data2[,"x"], train_val_data2[,"y"])
knn2 <- knearneigh(coords2, k = 12)  # adjust k based on your data
nb2 <- knn2nb(knn2)
weights2 <- nb2listw(nb2, style = "W")

# Test for spatial autocorrelation
moran_test2 <- moran.test(pearson_residuals2, weights2)
print(moran_test2)


# model 3
X_train_all3 <- as.matrix(train_val_data3[, covars_forest])
y_train_all3 <- train_val_data3[, "landslide"]
coords_train_all3 <- as.matrix(train_val_data3[, c("x", "y")])

dtrain_all3 <- gpb.Dataset(data = X_train_all3,
                           label = y_train_all3,
                           categorical_feature = cat_feat_forest)

gp_model_final3 <- GPModel(gp_coords = coords_train_all3, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model3 <- gpboost(
  data = dtrain_all3,
  gp_model = gp_model_final3,
  nrounds = opt_params3$best_iter,
  learning_rate = opt_params3$best_params$learning_rate,
  num_leaves = opt_params3$best_params$num_leaves,
  min_data_in_leaf = opt_params3$best_params$min_data_in_leaf,
  lambda_l2 = opt_params3$best_params$lambda_l2,
  verbose = 1
)

pred_train3 <- predict(final_model3, 
                       data = as.matrix(train_val_data3[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data3[, c("x", "y")]),
                       predict_var = FALSE)

train_auc3 <- auc(roc(y_train_all3, pred_train3$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test3 <- as.matrix(test_data3[, covars_forest])
y_test3 <- test_data3[, "landslide"]
coords_test3 <- as.matrix(test_data3[, c("x", "y")])

test_pred3 <- predict(final_model3, 
                      data = X_test3, 
                      gp_coords_pred = coords_test3)

test_auc3 <- auc(roc(y_test3, test_pred3$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop3 = gp_model_final3$get_cov_pars()[1] / (gp_model_final3$get_cov_pars()[1]+1)


# 3. Pearson residuals 
pearson_residuals3 <- (y_train_all3 - pred_train3$response_mean) / 
  sqrt(pred_train3$response_mean * (1 - pred_train3$response_mean))

# Create spatial neighbors 

coords3 <- cbind(train_val_data3[,"x"], train_val_data3[,"y"])
knn3 <- knearneigh(coords3, k = 12)  # adjust k based on your data
nb3 <- knn2nb(knn3)
weights3 <- nb2listw(nb3, style = "W")

# Test for spatial autocorrelation
moran_test3 <- moran.test(pearson_residuals3, weights3)
print(moran_test3)

# model 4
X_train_all4 <- as.matrix(train_val_data4[, covars_forest])
y_train_all4 <- train_val_data4[, "landslide"]
coords_train_all4 <- as.matrix(train_val_data4[, c("x", "y")])

dtrain_all4 <- gpb.Dataset(data = X_train_all4,
                           label = y_train_all4,
                           categorical_feature = cat_feat_forest)

gp_model_final4 <- GPModel(gp_coords = coords_train_all4, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model4 <- gpboost(
  data = dtrain_all4,
  gp_model = gp_model_final4,
  nrounds = opt_params4$best_iter,
  learning_rate = opt_params4$best_params$learning_rate,
  num_leaves = opt_params4$best_params$num_leaves,
  min_data_in_leaf = opt_params4$best_params$min_data_in_leaf,
  lambda_l2 = opt_params4$best_params$lambda_l2,
  verbose = 1
)

pred_train4 <- predict(final_model4, 
                       data = as.matrix(train_val_data4[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data4[, c("x", "y")]),
                       predict_var = FALSE)

train_auc4 <- auc(roc(y_train_all4, pred_train4$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test4 <- as.matrix(test_data4[, covars_forest])
y_test4 <- test_data4[, "landslide"]
coords_test4 <- as.matrix(test_data4[, c("x", "y")])

test_pred4 <- predict(final_model4, 
                      data = X_test4, 
                      gp_coords_pred = coords_test4)

test_auc4 <- auc(roc(y_test4, test_pred4$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop4 = gp_model_final4$get_cov_pars()[1] / (gp_model_final4$get_cov_pars()[1]+1)

# 4. Pearson residuals 
pearson_residuals4 <- (y_train_all4 - pred_train4$response_mean) / 
  sqrt(pred_train4$response_mean * (1 - pred_train4$response_mean))

# Create spatial neighbors 

coords4 <- cbind(train_val_data4[,"x"], train_val_data4[,"y"])
knn4 <- knearneigh(coords4, k = 12)  # adjust k based on your data
nb4 <- knn2nb(knn4)
weights4 <- nb2listw(nb4, style = "W")

# Test for spatial autocorrelation
moran_test4 <- moran.test(pearson_residuals4, weights4)
print(moran_test4)

# model 5
X_train_all5 <- as.matrix(train_val_data5[, covars_forest])
y_train_all5 <- train_val_data5[, "landslide"]
coords_train_all5 <- as.matrix(train_val_data5[, c("x", "y")])

dtrain_all5 <- gpb.Dataset(data = X_train_all5,
                           label = y_train_all5,
                           categorical_feature = cat_feat_forest)

gp_model_final5 <- GPModel(gp_coords = coords_train_all5, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model5 <- gpboost(
  data = dtrain_all5,
  gp_model = gp_model_final5,
  nrounds = opt_params5$best_iter,
  learning_rate = opt_params5$best_params$learning_rate,
  num_leaves = opt_params5$best_params$num_leaves,
  min_data_in_leaf = opt_params5$best_params$min_data_in_leaf,
  lambda_l2 = opt_params5$best_params$lambda_l2,
  verbose = 1
)

pred_train5 <- predict(final_model5, 
                       data = as.matrix(train_val_data5[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data5[, c("x", "y")]),
                       predict_var = FALSE)

train_auc5 <- auc(roc(y_train_all5, pred_train5$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test5 <- as.matrix(test_data5[, covars_forest])
y_test5 <- test_data5[, "landslide"]
coords_test5 <- as.matrix(test_data5[, c("x", "y")])

test_pred5 <- predict(final_model5, 
                      data = X_test5, 
                      gp_coords_pred = coords_test5)

test_auc5 <- auc(roc(y_test5, test_pred5$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop5 = gp_model_final5$get_cov_pars()[1] / (gp_model_final5$get_cov_pars()[1]+1)

# 5. Pearson residuals 
pearson_residuals5 <- (y_train_all5 - pred_train5$response_mean) / 
  sqrt(pred_train5$response_mean * (1 - pred_train5$response_mean))

# Create spatial neighbors 

coords5 <- cbind(train_val_data5[,"x"], train_val_data5[,"y"])
knn5 <- knearneigh(coords5, k = 12)  # adjust k based on your data
nb5 <- knn2nb(knn5)
weights5 <- nb2listw(nb5, style = "W")

# Test for spatial autocorrelation
moran_test5 <- moran.test(pearson_residuals5, weights5)
print(moran_test5)

cv_res_ran = cv_res_ran |> 
  mutate(mean_train_auc = mean(c(train_auc1,train_auc2,train_auc3,train_auc4,train_auc5)),
         sd_train_auc = sd(c(train_auc1,train_auc2,train_auc3,train_auc4,train_auc5)),
         mean_test_auc = mean(c(test_auc1, test_auc2, test_auc3, test_auc4, test_auc5)),
         sd_test_auc = sd(c(test_auc1, test_auc2, test_auc3, test_auc4, test_auc5)),
         mean_space = mean(c(space_prop1, space_prop2, space_prop3, space_prop4, space_prop5)),
         low_moran = min(c(moran_test1$estimate[[1]],
                           moran_test2$estimate[[1]],
                           moran_test3$estimate[[1]],
                           moran_test4$estimate[[1]],
                           moran_test5$estimate[[1]])),
         hi_moran = max(c(moran_test1$estimate[[1]],
                          moran_test2$estimate[[1]],
                          moran_test3$estimate[[1]],
                          moran_test4$estimate[[1]],
                          moran_test5$estimate[[1]])),
         low_p = min(c(moran_test1$p.value[[1]],
                       moran_test2$p.value[[1]],
                       moran_test3$p.value[[1]],
                       moran_test4$p.value[[1]],
                       moran_test5$p.value[[1]])),
         hi_p = max(c(moran_test1$p.value[[1]],
                      moran_test2$p.value[[1]],
                      moran_test3$p.value[[1]],
                      moran_test4$p.value[[1]],
                      moran_test5$p.value[[1]])))


cv_res = read.csv("data/Nested_CV_results.csv")

cv_res_v2 = cv_res[5,]

# final model
final_model_all <- gpboost(
  data = d_all,
  gp_model = gp_model_final_all,
  nrounds = round(cv_res_v2$best_iter,0),
  learning_rate = cv_res_v2$learning_rate,
  num_leaves = cv_res_v2$num_leaves,
  min_data_in_leaf = cv_res_v2$min_data_in_leaf,
  lambda_l2 = cv_res_v2$lambda_l2,
  verbose = 1
)

saveRDS.gpb.Booster(final_model_all, "saved models/Sp_random_forest_final.rds")


pred_all <- predict(final_model_all, 
                    data = as.matrix(Lands_forest[, covars_forest]),
                    gp_coords_pred = as.matrix(Lands_forest[, c("x", "y")]),
                    predict_var = FALSE)

all_auc <- auc(roc(y_all, pred_all$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop_all = gp_model_final_all$get_cov_pars()[1] / (gp_model_final_all$get_cov_pars()[1]+1)

# 5. Pearson residuals 
pearson_residuals_all <- (y_all - pred_all$response_mean) / 
  sqrt(pred_all$response_mean * (1 - pred_all$response_mean))

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)

## non spatial version of random CV ####
# already have all the folds etc
opt_params1_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list1,  
  data = dtrain_all1,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params2_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list2,  
  data = dtrain_all2,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params3_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list3,  
  data = dtrain_all3,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params4_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list4,  
  data = dtrain_all4,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params5_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list5,  
  data = dtrain_all5,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

cv_res_ran_ns = data.frame(Model = "Non_Spatial_random_forest",
                           learning_rate = median(c(opt_params1_ns$best_params$learning_rate,
                                                    opt_params2_ns$best_params$learning_rate,
                                                    opt_params3_ns$best_params$learning_rate,
                                                    opt_params4_ns$best_params$learning_rate,
                                                    opt_params5_ns$best_params$learning_rate)),
                           min_data_in_leaf = median(c(opt_params1_ns$best_params$min_data_in_leaf,
                                                       opt_params2_ns$best_params$min_data_in_leaf,
                                                       opt_params3_ns$best_params$min_data_in_leaf,
                                                       opt_params4_ns$best_params$min_data_in_leaf,
                                                       opt_params5_ns$best_params$min_data_in_leaf)),
                           num_leaves = median(c(opt_params1_ns$best_params$num_leaves,
                                                 opt_params2_ns$best_params$num_leaves,
                                                 opt_params3_ns$best_params$num_leaves,
                                                 opt_params4_ns$best_params$num_leaves,
                                                 opt_params5_ns$best_params$num_leaves)),
                           lambda_l2 = median(c(opt_params1_ns$best_params$lambda_l2,
                                                opt_params2_ns$best_params$lambda_l2,
                                                opt_params3_ns$best_params$lambda_l2,
                                                opt_params4_ns$best_params$lambda_l2,
                                                opt_params5_ns$best_params$lambda_l2)),
                           best_iter = median(c(opt_params1_ns$best_iter,
                                                opt_params2_ns$best_iter,
                                                opt_params3_ns$best_iter,
                                                opt_params4_ns$best_iter,
                                                opt_params5_ns$best_iter)),
                           mean_auc_hyp = mean(c(opt_params1_ns$best_score,
                                                 opt_params2_ns$best_score,
                                                 opt_params3_ns$best_score,
                                                 opt_params4_ns$best_score,
                                                 opt_params5_ns$best_score)),
                           sd_auc_hyp =  sd(c(opt_params1_ns$best_score,
                                              opt_params2_ns$best_score,
                                              opt_params3_ns$best_score,
                                              opt_params4_ns$best_score,
                                              opt_params5_ns$best_score))
)

# fit models with best parameters - non_spatial ####
final_model1_ns <- gpboost(
  data = dtrain_all1,
  gp_model = NULL,
  nrounds = opt_params1_ns$best_iter,
  learning_rate = opt_params1_ns$best_params$learning_rate,
  num_leaves = opt_params1_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params1_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params1_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train1_ns <- predict(final_model1_ns, 
                          data = as.matrix(train_val_data1[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data1[, c("x", "y")]),
                          predict_var = FALSE)

train_auc1_ns <- auc(roc(y_train_all1, pred_train1_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred1_ns <- predict(final_model1_ns, 
                         data = X_test1, 
                         gp_coords_pred = coords_test1)

test_auc1_ns <- auc(roc(y_test1, test_pred1_ns))


# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe1 <- pmax(pmin(pred_train1_ns, 1 - epsilon), epsilon)
pearson_residuals1_ns <- (y_train_all1 - y_pred_prob_safe1) / 
  sqrt(y_pred_prob_safe1 * (1 - y_pred_prob_safe1))

# Test for spatial autocorrelation
moran_test1_ns <- moran.test(pearson_residuals1_ns, weights1)
print(moran_test1_ns)


# model 2
final_model2_ns <- gpboost(
  data = dtrain_all2,
  gp_model = NULL,
  nrounds = opt_params2_ns$best_iter,
  learning_rate = opt_params2_ns$best_params$learning_rate,
  num_leaves = opt_params2_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params2_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params2_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train2_ns <- predict(final_model2_ns, 
                          data = as.matrix(train_val_data2[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data2[, c("x", "y")]),
                          predict_var = FALSE)

train_auc2_ns <- auc(roc(y_train_all2, pred_train2_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred2_ns <- predict(final_model2_ns, 
                         data = X_test2, 
                         gp_coords_pred = coords_test2)

test_auc2_ns <- auc(roc(y_test2, test_pred2_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe2 <- pmax(pmin(pred_train2_ns, 1 - epsilon), epsilon)
pearson_residuals2_ns <- (y_train_all2 - y_pred_prob_safe2) / 
  sqrt(y_pred_prob_safe2 * (1 - y_pred_prob_safe2))

# Test for spatial autocorrelation
moran_test2_ns <- moran.test(pearson_residuals2_ns, weights2)
print(moran_test2_ns)

# model 3
final_model3_ns <- gpboost(
  data = dtrain_all3,
  gp_model = NULL,
  nrounds = opt_params3_ns$best_iter,
  learning_rate = opt_params3_ns$best_params$learning_rate,
  num_leaves = opt_params3_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params3_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params3_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train3_ns <- predict(final_model3_ns, 
                          data = as.matrix(train_val_data3[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data3[, c("x", "y")]),
                          predict_var = FALSE)

train_auc3_ns <- auc(roc(y_train_all3, pred_train3_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred3_ns <- predict(final_model3_ns, 
                         data = X_test3, 
                         gp_coords_pred = coords_test3)

test_auc3_ns <- auc(roc(y_test3, test_pred3_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe3 <- pmax(pmin(pred_train3_ns, 1 - epsilon), epsilon)
pearson_residuals3_ns <- (y_train_all3 - y_pred_prob_safe3) / 
  sqrt(y_pred_prob_safe3 * (1 - y_pred_prob_safe3))

# Test for spatial autocorrelation
moran_test3_ns <- moran.test(pearson_residuals3_ns, weights3)
print(moran_test3_ns)

# model 4
final_model4_ns <- gpboost(
  data = dtrain_all4,
  gp_model = NULL,
  nrounds = opt_params4_ns$best_iter,
  learning_rate = opt_params4_ns$best_params$learning_rate,
  num_leaves = opt_params4_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params4_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params4_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train4_ns <- predict(final_model4_ns, 
                          data = as.matrix(train_val_data4[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data4[, c("x", "y")]),
                          predict_var = FALSE)

train_auc4_ns <- auc(roc(y_train_all4, pred_train4_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred4_ns <- predict(final_model4_ns, 
                         data = X_test4, 
                         gp_coords_pred = coords_test4)

test_auc4_ns <- auc(roc(y_test4, test_pred4_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe4 <- pmax(pmin(pred_train4_ns, 1 - epsilon), epsilon)
pearson_residuals4_ns <- (y_train_all4 - y_pred_prob_safe4) / 
  sqrt(y_pred_prob_safe4 * (1 - y_pred_prob_safe4))

# Test for spatial autocorrelation
moran_test4_ns <- moran.test(pearson_residuals4_ns, weights4)
print(moran_test4_ns)

# model 5
final_model5_ns <- gpboost(
  data = dtrain_all5,
  gp_model = NULL,
  nrounds = opt_params5_ns$best_iter,
  learning_rate = opt_params5_ns$best_params$learning_rate,
  num_leaves = opt_params5_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params5_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params5_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train5_ns <- predict(final_model5_ns, 
                          data = as.matrix(train_val_data5[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data5[, c("x", "y")]),
                          predict_var = FALSE)

train_auc5_ns <- auc(roc(y_train_all5, pred_train5_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred5_ns <- predict(final_model5_ns, 
                         data = X_test5, 
                         gp_coords_pred = coords_test5)

test_auc5_ns <- auc(roc(y_test5, test_pred5_ns))


# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe5 <- pmax(pmin(pred_train5_ns, 1 - epsilon), epsilon)
pearson_residuals5_ns <- (y_train_all5 - y_pred_prob_safe5) / 
  sqrt(y_pred_prob_safe5 * (1 - y_pred_prob_safe5))

# Test for spatial autocorrelation
moran_test5_ns <- moran.test(pearson_residuals5_ns, weights5)
print(moran_test5_ns)

cv_res_ran_ns = cv_res_ran_ns |> 
  mutate(mean_train_auc = mean(c(train_auc1_ns,train_auc2_ns,train_auc3_ns,train_auc4_ns,train_auc5_ns)),
         sd_train_auc = sd(c(train_auc1_ns,train_auc2_ns,train_auc3_ns,train_auc4_ns,train_auc5_ns)),
         mean_test_auc = mean(c(test_auc1_ns, test_auc2_ns, test_auc3_ns, test_auc4_ns, test_auc5_ns)),
         sd_test_auc = sd(c(test_auc1_ns, test_auc2_ns, test_auc3_ns, test_auc4_ns, test_auc5_ns)),
         mean_space = "NA",
         low_moran = min(c(moran_test1_ns$estimate[[1]],
                           moran_test2_ns$estimate[[1]],
                           moran_test3_ns$estimate[[1]],
                           moran_test4_ns$estimate[[1]],
                           moran_test5_ns$estimate[[1]])),
         hi_moran = max(c(moran_test1_ns$estimate[[1]],
                          moran_test2_ns$estimate[[1]],
                          moran_test3_ns$estimate[[1]],
                          moran_test4_ns$estimate[[1]],
                          moran_test5_ns$estimate[[1]])),
         low_p = min(c(moran_test1_ns$p.value[[1]],
                       moran_test2_ns$p.value[[1]],
                       moran_test3_ns$p.value[[1]],
                       moran_test4_ns$p.value[[1]],
                       moran_test5_ns$p.value[[1]])),
         hi_p = max(c(moran_test1_ns$p.value[[1]],
                      moran_test2_ns$p.value[[1]],
                      moran_test3_ns$p.value[[1]],
                      moran_test4_ns$p.value[[1]],
                      moran_test5_ns$p.value[[1]])))

## final final model on full data set with median hyperparameters

cv_res = rbind(cv_res, cv_res_ran,cv_res_ran_ns)

write.csv(cv_res, "data/Nested_CV_results_forest.csv")

## final global model on full data set with median hyperparameters
X_all <- as.matrix(Lands_forest[, covars_forest])
y_all <- Lands_forest[, "landslide"]
coords_all <- as.matrix(Lands_forest[, c("x", "y")])

d_all <- gpb.Dataset(data = X_all,
                     label = y_all,
                     categorical_feature = cat_feat_forest)

gp_model_final_all <- GPModel(gp_coords = coords_all, 
                              likelihood = "bernoulli_probit",
                              cov_function = "matern" ,
                              cov_fct_shape = 1.5)


# final model - need to choose the right params from cv_res
final_model_all <- gpboost(
  data = d_all,
  gp_model = NULL,
  nrounds = cv_res_ran_ns$best_iter,
  learning_rate = cv_res_ran_ns$learning_rate,
  num_leaves = cv_res_ran_ns$num_leaves,
  min_data_in_leaf = cv_res_ran_ns$min_data_in_leaf,
  lambda_l2 = cv_res_ran_ns$lambda_l2,
  verbose = 1
)

# final_model_all = readRDS.gpb.Booster("saved models/Non_Sp_random_forest_final.rds")

pred_all <- predict(final_model_all, 
                    data = as.matrix(Lands_forest[, covars_forest]),
                    gp_coords_pred = as.matrix(Lands_forest[, c("x", "y")]),
                    predict_var = FALSE)

all_auc <- auc(roc(y_all, pred_all))

# Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe_all <- pmax(pmin(pred_all, 1 - epsilon), epsilon)
pearson_residuals_all <- (y_all - y_pred_prob_safe_all) / 
  sqrt(y_pred_prob_safe_all * (1 - y_pred_prob_safe_all))


# Create spatial neighbors 

coords_all <- cbind(Lands_forest[,"x"], Lands_forest[,"y"])
knn_all <- knearneigh(coords_all, k = 12)  # adjust k based on your data
nb_all <- knn2nb(knn_all)
weights_all <- nb2listw(nb_all, style = "W")

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)


library(SHAPforxgboost)
shap_values_forest <- shap.values(xgb_model = final_model_all, 
                                X_train = X_all)
shap.plot.summary.wrap1(final_model_all, X = X_all) + ggtitle("SHAP values")
shap_long <- shap.prep(final_model_all, X_train = X_all)
fig_list <- lapply(names(shap_values_forest$mean_shap_score)[1:9], 
                   shap.plot.dependence, data_long = shap_long)
gridExtra::grid.arrange(grobs = fig_list, ncol = 3)


saveRDS.gpb.Booster(final_model_all, "saved models/Non_Sp_random_forest_final.rds")



# same again with large block CV --------------------------------------------

library(blockCV)
block_cv_large <- spatial_block_cv(
  data = sf_Lands_forest,
  v= 3, # only used 3 blocks here as the balance was poor with 4 or 5
  n = c(1,3)
)

autoplot(block_cv_large)

fold_id <- extract_folds(block_cv_large, nrow(Lands_forest))
Lands_forest$fold <- fold_id

# NOW check balance
table(Lands_forest$fold, Lands_forest$landslide)



# spatial inner loop ####
# Assign one block as test set, one at a time
test_idx1 <- which(Lands_forest$fold == 1)  
test_idx2 <- which(Lands_forest$fold == 2)  
test_idx3 <- which(Lands_forest$fold == 3)  
  

train_val_idx1 <- which(Lands_forest$fold != 1)  
train_val_idx2 <- which(Lands_forest$fold != 2)  
train_val_idx3 <- which(Lands_forest$fold != 3)  

# Lock away the test set
test_data1 <- Lands_forest[test_idx1, ]
test_data2 <- Lands_forest[test_idx2, ]
test_data3 <- Lands_forest[test_idx3, ]


train_val_data1 <- Lands_forest[train_val_idx1, ]  
train_val_data2 <- Lands_forest[train_val_idx2, ]  
train_val_data3 <- Lands_forest[train_val_idx3, ]  

# STEP 2: Hyperparameter tuning with CV on training set ONLY
# Create NEW spatial folds within the training data
train_val_sf1 <- st_as_sf(train_val_data1, coords = c("x", "y"), crs = 32632)
train_val_sf2 <- st_as_sf(train_val_data2, coords = c("x", "y"), crs = 32632)
train_val_sf3 <- st_as_sf(train_val_data3, coords = c("x", "y"), crs = 32632)

# block cv large
set.seed(13)

block_cv_large_train1 <- spatial_block_cv(
  data = train_val_sf1,
  v= 4,
  n = c(1,4)
)

autoplot(block_cv_large_train1)

fold_id_train1 <- extract_folds(block_cv_large_train1, nrow(train_val_data1))
train_val_data1$fold <- fold_id_train1

# NOW check balance
table(train_val_data1$fold, train_val_data1$landslide)

block_cv_large_train2 <- spatial_block_cv(
  data = train_val_sf2,
  v= 4,
  n = c(1,4)
)

autoplot(block_cv_large_train2)

fold_id_train2 <- extract_folds(block_cv_large_train2, nrow(train_val_data2))
train_val_data2$fold <- fold_id_train2

# NOW check balance
table(train_val_data2$fold, train_val_data2$landslide)

block_cv_large_train3 <- spatial_block_cv(
  data = train_val_sf3,
  v= 4,
  n = c(1,4)
)

autoplot(block_cv_large_train3)

fold_id_train3 <- extract_folds(block_cv_large_train3, nrow(train_val_data3))
train_val_data3$fold <- fold_id_train3

# NOW check balance
table(train_val_data3$fold, train_val_data3$landslide)



# Convert to list format that gpb.grid.search.tune.parameters expects
# Each list element = indices of validation observations for that fold
folds_list1 <- list()
folds_list2 <- list()
folds_list3 <- list()


for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list1[[i]] <- which(fold_id_train1 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list2[[i]] <- which(fold_id_train2 == i)
}

for (i in 1:4) {
  # Get indices where fold == i (these are the VALIDATION indices)
  folds_list3[[i]] <- which(fold_id_train3 == i)
}



param_grid <- list(
  learning_rate = c(0.05,0.01), # needs to be low
  min_data_in_leaf = c(10, 20, 50), #10-100 is good for generalisation
  num_leaves = c(8,16,32), # 2-8 strong reg, 16-128 - balance
  lambda_l2 = c(10, 50, 100), # 0 - no reg, 1-10 mild, 100 strong but may underfit
  max_depth = c(-1)
) 



other_params <- list(
  objective = "binary",
  verbose = -1
)

# re instate model and data
X_train_all1 <- as.matrix(train_val_data1[, covars_forest])
X_train_all2 <- as.matrix(train_val_data2[, covars_forest])
X_train_all3 <- as.matrix(train_val_data3[, covars_forest])

y_train_all1 <- train_val_data1[, "landslide"]
y_train_all2 <- train_val_data2[, "landslide"]
y_train_all3 <- train_val_data3[, "landslide"]


coords_train_all1 <- as.matrix(train_val_data1[, c("x", "y")])
coords_train_all2 <- as.matrix(train_val_data2[, c("x", "y")])
coords_train_all3 <- as.matrix(train_val_data3[, c("x", "y")])


dtrain_all1 <- gpb.Dataset(data = X_train_all1,
                           label = y_train_all1,
                           categorical_feature = cat_feat_forest)
dtrain_all2 <- gpb.Dataset(data = X_train_all2,
                           label = y_train_all2,
                           categorical_feature = cat_feat_forest)
dtrain_all3 <- gpb.Dataset(data = X_train_all3,
                           label = y_train_all3,
                           categorical_feature = cat_feat_forest)


gp_model_final1 <- GPModel(gp_coords = coords_train_all1, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final2 <- GPModel(gp_coords = coords_train_all2, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)
gp_model_final3 <- GPModel(gp_coords = coords_train_all3, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)


opt_params1 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list1,  
  data = dtrain_all1,
  gp_model = gp_model_final1,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params2 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list2,  
  data = dtrain_all2,
  gp_model = gp_model_final2,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)

opt_params3 <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list3,  
  data = dtrain_all3,
  gp_model = gp_model_final3,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  use_gp_model_for_validation = TRUE,
  train_gp_model_cov_pars = TRUE,
  return_all_combinations = TRUE 
)


# summarise optimum params and auc

cv_res_Lb = data.frame(Model = "Spatial_largeB_forest",
                       learning_rate = median(c(opt_params1$best_params$learning_rate,
                                                opt_params2$best_params$learning_rate,
                                                opt_params3$best_params$learning_rate)),
                       min_data_in_leaf = median(c(opt_params1$best_params$min_data_in_leaf,
                                                   opt_params2$best_params$min_data_in_leaf,
                                                   opt_params3$best_params$min_data_in_leaf)),
                       num_leaves = median(c(opt_params1$best_params$num_leaves,
                                             opt_params2$best_params$num_leaves,
                                             opt_params3$best_params$num_leaves)),
                       lambda_l2 = median(c(opt_params1$best_params$lambda_l2,
                                            opt_params2$best_params$lambda_l2,
                                            opt_params3$best_params$lambda_l2)),
                       best_iter = mean(c(opt_params1$best_iter,
                                          opt_params2$best_iter,
                                          opt_params3$best_iter)),
                       mean_auc_hyp = mean(c(opt_params1$best_score,
                                             opt_params2$best_score,
                                             opt_params3$best_score)),
                       sd_auc_hyp =  sd(c(opt_params1$best_score,
                                          opt_params2$best_score,
                                          opt_params3$best_score))
)

# STEP 4: Retrain on ALL training data with best parameters x 5
# Now use ALL of train_val_data
X_train_all1 <- as.matrix(train_val_data1[, covars_forest])
y_train_all1 <- train_val_data1[, "landslide"]
coords_train_all1 <- as.matrix(train_val_data1[, c("x", "y")])

dtrain_all1 <- gpb.Dataset(data = X_train_all1,
                           label = y_train_all1,
                           categorical_feature = cat_feat_forest)

gp_model_final1 <- GPModel(gp_coords = coords_train_all1, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)

final_model1 <- gpboost(
  data = dtrain_all1,
  gp_model = gp_model_final1,
  nrounds = opt_params1$best_iter,
  learning_rate = opt_params1$best_params$learning_rate,
  num_leaves = opt_params1$best_params$num_leaves,
  min_data_in_leaf = opt_params1$best_params$min_data_in_leaf,
  lambda_l2 = opt_params1$best_params$lambda_l2,
  verbose = 1
)

pred_train1 <- predict(final_model1, 
                       data = as.matrix(train_val_data1[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data1[, c("x", "y")]),
                       predict_var = FALSE)

train_auc1 <- auc(roc(y_train_all1, pred_train1$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test1 <- as.matrix(test_data1[, covars_forest])
y_test1 <- test_data1[, "landslide"]
coords_test1 <- as.matrix(test_data1[, c("x", "y")])

test_pred1 <- predict(final_model1, 
                      data = X_test1, 
                      gp_coords_pred = coords_test1)

test_auc1 <- auc(roc(y_test1, test_pred1$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop1 = gp_model_final1$get_cov_pars()[1] / (gp_model_final1$get_cov_pars()[1]+1)


# 1. Pearson residuals 
pearson_residuals1 <- (y_train_all1 - pred_train1$response_mean) / 
  sqrt(pred_train1$response_mean * (1 - pred_train1$response_mean))

# Create spatial neighbors 

coords1 <- cbind(train_val_data1[,"x"], train_val_data1[,"y"])
knn1 <- knearneigh(coords1, k = 12)  # adjust k based on your data
nb1 <- knn2nb(knn1)
weights1 <- nb2listw(nb1, style = "W")

# Test for spatial autocorrelation
moran_test1 <- moran.test(pearson_residuals1, weights1)
print(moran_test1)


# model 2
X_train_all2 <- as.matrix(train_val_data2[, covars_forest])
y_train_all2 <- train_val_data2[, "landslide"]
coords_train_all2 <- as.matrix(train_val_data2[, c("x", "y")])

dtrain_all2 <- gpb.Dataset(data = X_train_all2,
                           label = y_train_all2,
                           categorical_feature = cat_feat_forest)

gp_model_final2 <- GPModel(gp_coords = coords_train_all2, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model2 <- gpboost(
  data = dtrain_all2,
  gp_model = gp_model_final2,
  nrounds = opt_params2$best_iter,
  learning_rate = opt_params2$best_params$learning_rate,
  num_leaves = opt_params2$best_params$num_leaves,
  min_data_in_leaf = opt_params2$best_params$min_data_in_leaf,
  lambda_l2 = opt_params2$best_params$lambda_l2,
  verbose = 1
)

pred_train2 <- predict(final_model2, 
                       data = as.matrix(train_val_data2[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data2[, c("x", "y")]),
                       predict_var = FALSE)

train_auc2 <- auc(roc(y_train_all2, pred_train2$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test2 <- as.matrix(test_data2[, covars_forest])
y_test2 <- test_data2[, "landslide"]
coords_test2 <- as.matrix(test_data2[, c("x", "y")])

test_pred2 <- predict(final_model2, 
                      data = X_test2, 
                      gp_coords_pred = coords_test2)

test_auc2 <- auc(roc(y_test2, test_pred2$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop2 = gp_model_final2$get_cov_pars()[1] / (gp_model_final2$get_cov_pars()[1]+1)

# 2. Pearson residuals 
pearson_residuals2 <- (y_train_all2 - pred_train2$response_mean) / 
  sqrt(pred_train2$response_mean * (1 - pred_train2$response_mean))

# Create spatial neighbors 

coords2 <- cbind(train_val_data2[,"x"], train_val_data2[,"y"])
knn2 <- knearneigh(coords2, k = 12)  # adjust k based on your data
nb2 <- knn2nb(knn2)
weights2 <- nb2listw(nb2, style = "W")

# Test for spatial autocorrelation
moran_test2 <- moran.test(pearson_residuals2, weights2)
print(moran_test2)


# model 3
X_train_all3 <- as.matrix(train_val_data3[, covars_forest])
y_train_all3 <- train_val_data3[, "landslide"]
coords_train_all3 <- as.matrix(train_val_data3[, c("x", "y")])

dtrain_all3 <- gpb.Dataset(data = X_train_all3,
                           label = y_train_all3,
                           categorical_feature = cat_feat_forest)

gp_model_final3 <- GPModel(gp_coords = coords_train_all3, 
                           likelihood = "bernoulli_probit",
                           cov_function = "matern" ,
                           cov_fct_shape = 1.5)



final_model3 <- gpboost(
  data = dtrain_all3,
  gp_model = gp_model_final3,
  nrounds = opt_params3$best_iter,
  learning_rate = opt_params3$best_params$learning_rate,
  num_leaves = opt_params3$best_params$num_leaves,
  min_data_in_leaf = opt_params3$best_params$min_data_in_leaf,
  lambda_l2 = opt_params3$best_params$lambda_l2,
  verbose = 1
)

pred_train3 <- predict(final_model3, 
                       data = as.matrix(train_val_data3[, covars_forest]),
                       gp_coords_pred = as.matrix(train_val_data3[, c("x", "y")]),
                       predict_var = FALSE)

train_auc3 <- auc(roc(y_train_all3, pred_train3$response_mean))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
X_test3 <- as.matrix(test_data3[, covars_forest])
y_test3 <- test_data3[, "landslide"]
coords_test3 <- as.matrix(test_data3[, c("x", "y")])

test_pred3 <- predict(final_model3, 
                      data = X_test3, 
                      gp_coords_pred = coords_test3)

test_auc3 <- auc(roc(y_test3, test_pred3$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop3 = gp_model_final3$get_cov_pars()[1] / (gp_model_final3$get_cov_pars()[1]+1)


# 3. Pearson residuals 
pearson_residuals3 <- (y_train_all3 - pred_train3$response_mean) / 
  sqrt(pred_train3$response_mean * (1 - pred_train3$response_mean))

# Create spatial neighbors 

coords3 <- cbind(train_val_data3[,"x"], train_val_data3[,"y"])
knn3 <- knearneigh(coords3, k = 12)  # adjust k based on your data
nb3 <- knn2nb(knn3)
weights3 <- nb2listw(nb3, style = "W")

# Test for spatial autocorrelation
moran_test3 <- moran.test(pearson_residuals3, weights3)
print(moran_test3)



cv_res_Lb = cv_res_Lb |> 
  mutate(mean_train_auc = mean(c(train_auc1,train_auc2,train_auc3)),
         sd_train_auc = sd(c(train_auc1,train_auc2,train_auc3)),
         mean_test_auc = mean(c(test_auc1, test_auc2, test_auc3)),
         sd_test_auc = sd(c(test_auc1, test_auc2, test_auc3)),
         mean_space = mean(c(space_prop1, space_prop2, space_prop3)),
         low_moran = min(c(moran_test1$estimate[[1]],
                           moran_test2$estimate[[1]],
                           moran_test3$estimate[[1]])),
         hi_moran = max(c(moran_test1$estimate[[1]],
                          moran_test2$estimate[[1]],
                          moran_test3$estimate[[1]])),
         low_p = min(c(moran_test1$p.value[[1]],
                       moran_test2$p.value[[1]],
                       moran_test3$p.value[[1]])),
         hi_p = max(c(moran_test1$p.value[[1]],
                      moran_test2$p.value[[1]],
                      moran_test3$p.value[[1]])))

## final global model on full data set with median hyperparameters
X_all <- as.matrix(Lands_forest[, covars_forest])
y_all <- Lands_forest[, "landslide"]
coords_all <- as.matrix(Lands_forest[, c("x", "y")])

d_all <- gpb.Dataset(data = X_all,
                     label = y_all,
                     categorical_feature = cat_feat_forest)

gp_model_final_all <- GPModel(gp_coords = coords_all, 
                              likelihood = "bernoulli_probit",
                              cov_function = "matern" ,
                              cov_fct_shape = 1.5)

# cv_res = read.csv("Files from homePC/data/Nested_CV_results_forest.csv")

cv_res_Lb = cv_res[7,]

# final model
final_model_all <- gpboost(
  data = d_all,
  gp_model = gp_model_final_all,
  nrounds = round(cv_res_Lb$best_iter, 0),
  learning_rate = cv_res_Lb$learning_rate,
  min_data_in_leaf = cv_res_Lb$min_data_in_leaf,
  num_leaves = cv_res_Lb$num_leaves,
  lambda_l2 = cv_res_Lb$lambda_l2,
  verbose = 1
)

# final_model_all = readRDS.gpb.Booster("saved models/Sp_LargeB_forest_final.rds")

pred_all <- predict(final_model_all, 
                    data = as.matrix(Lands_forest[, covars_forest]),
                    gp_coords_pred = as.matrix(Lands_forest[, c("x", "y")]),
                    predict_var = FALSE)

all_auc <- auc(roc(y_all, pred_all$response_mean))

# Calculate the proportion of variance explained by spatial structure
space_prop_all = gp_model_final_all$get_cov_pars()[1] / (gp_model_final_all$get_cov_pars()[1]+1)

# 5. Pearson residuals 
pearson_residuals_all <- (y_all - pred_all$response_mean) / 
  sqrt(pred_all$response_mean * (1 - pred_all$response_mean))

# Create spatial neighbors 

coords_all <- cbind(Lands_forest[,"x"], Lands_forest[,"y"])
knn_all <- knearneigh(coords_all, k = 12)  # adjust k based on your data
nb_all <- knn2nb(knn_all)
weights_all <- nb2listw(nb_all, style = "W")

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)


library(SHAPforxgboost)
shap_values_forest <- shap.values(xgb_model = final_model_all, 
                                X_train = X_all)
shap.plot.summary.wrap1(final_model_all, X = X_all) + ggtitle("SHAP values")
shap_long <- shap.prep(final_model_all, X_train = X_all)
fig_list <- lapply(names(shap_values_forest$mean_shap_score)[1:9], 
                   shap.plot.dependence, data_long = shap_long)
gridExtra::grid.arrange(grobs = fig_list, ncol = 3)

saveRDS.gpb.Booster(final_model_all, "saved models/Sp_LargeB_forest_final.rds")


## non spatial version of Large Block CV ####
# already have all the folds etc
opt_params1_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list1,  
  data = dtrain_all1,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params2_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list2,  
  data = dtrain_all2,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)

opt_params3_ns <- gpb.grid.search.tune.parameters(
  param_grid = param_grid,
  params = other_params,
  folds = folds_list3,  
  data = dtrain_all3,
  gp_model = NULL,
  nrounds = 500,
  early_stopping_rounds = 10,
  metric = "auc",  
  verbose_eval = 1,
  return_all_combinations = TRUE 
)



cv_res_Lb_ns = data.frame(Model = "Non_Spatial_LargeB_forest",
                          learning_rate = median(c(opt_params1_ns$best_params$learning_rate,
                                                   opt_params2_ns$best_params$learning_rate,
                                                   opt_params3_ns$best_params$learning_rate)),
                          min_data_in_leaf = median(c(opt_params1_ns$best_params$min_data_in_leaf,
                                                      opt_params2_ns$best_params$min_data_in_leaf,
                                                      opt_params3_ns$best_params$min_data_in_leaf)),
                          num_leaves = median(c(opt_params1_ns$best_params$num_leaves,
                                                opt_params2_ns$best_params$num_leaves,
                                                opt_params3_ns$best_params$num_leaves)),
                          lambda_l2 = median(c(opt_params1_ns$best_params$lambda_l2,
                                               opt_params2_ns$best_params$lambda_l2,
                                               opt_params3_ns$best_params$lambda_l2)),
                          best_iter = mean(c(opt_params1_ns$best_iter,
                                             opt_params2_ns$best_iter,
                                             opt_params3_ns$best_iter)),
                          mean_auc_hyp = mean(c(opt_params1_ns$best_score,
                                                opt_params2_ns$best_score,
                                                opt_params3_ns$best_score)),
                          sd_auc_hyp =  sd(c(opt_params1_ns$best_score,
                                             opt_params2_ns$best_score,
                                             opt_params3_ns$best_score))
)

# fit models with best parameters - non_spatial ####
final_model1_ns <- gpboost(
  data = dtrain_all1,
  gp_model = NULL,
  nrounds = opt_params1_ns$best_iter,
  learning_rate = opt_params1_ns$best_params$learning_rate,
  num_leaves = opt_params1_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params1_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params1_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train1_ns <- predict(final_model1_ns, 
                          data = as.matrix(train_val_data1[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data1[, c("x", "y")]),
                          predict_var = FALSE)

train_auc1_ns <- auc(roc(y_train_all1, pred_train1_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred1_ns <- predict(final_model1_ns, 
                         data = X_test1, 
                         gp_coords_pred = coords_test1)

test_auc1_ns <- auc(roc(y_test1, test_pred1_ns))

library(spdep)
library(sf)

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe1 <- pmax(pmin(pred_train1_ns, 1 - epsilon), epsilon)
pearson_residuals1_ns <- (y_train_all1 - y_pred_prob_safe1) / 
  sqrt(y_pred_prob_safe1 * (1 - y_pred_prob_safe1))

# Test for spatial autocorrelation
moran_test1_ns <- moran.test(pearson_residuals1_ns, weights1)
print(moran_test1_ns)


# model 2
final_model2_ns <- gpboost(
  data = dtrain_all2,
  gp_model = NULL,
  nrounds = opt_params2_ns$best_iter,
  learning_rate = opt_params2_ns$best_params$learning_rate,
  num_leaves = opt_params2_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params2_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params2_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train2_ns <- predict(final_model2_ns, 
                          data = as.matrix(train_val_data2[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data2[, c("x", "y")]),
                          predict_var = FALSE)

train_auc2_ns <- auc(roc(y_train_all2, pred_train2_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred2_ns <- predict(final_model2_ns, 
                         data = X_test2, 
                         gp_coords_pred = coords_test2)

test_auc2_ns <- auc(roc(y_test2, test_pred2_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe2 <- pmax(pmin(pred_train2_ns, 1 - epsilon), epsilon)
pearson_residuals2_ns <- (y_train_all2 - y_pred_prob_safe2) / 
  sqrt(y_pred_prob_safe2 * (1 - y_pred_prob_safe2))

# Test for spatial autocorrelation
moran_test2_ns <- moran.test(pearson_residuals2_ns, weights2)
print(moran_test2_ns)

# model 3
final_model3_ns <- gpboost(
  data = dtrain_all3,
  gp_model = NULL,
  nrounds = opt_params3_ns$best_iter,
  learning_rate = opt_params3_ns$best_params$learning_rate,
  num_leaves = opt_params3_ns$best_params$num_leaves,
  min_data_in_leaf = opt_params3_ns$best_params$min_data_in_leaf,
  lambda_l2 = opt_params3_ns$best_params$lambda_l2,
  verbose = 1
)

pred_train3_ns <- predict(final_model3_ns, 
                          data = as.matrix(train_val_data3[, covars_forest]),
                          gp_coords_pred = as.matrix(train_val_data3[, c("x", "y")]),
                          predict_var = FALSE)

train_auc3_ns <- auc(roc(y_train_all3, pred_train3_ns))

# STEP 5: NOW evaluate on test set (ONLY NOW!)
test_pred3_ns <- predict(final_model3_ns, 
                         data = X_test3, 
                         gp_coords_pred = coords_test3)

test_auc3_ns <- auc(roc(y_test3, test_pred3_ns))

# 1. Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe3 <- pmax(pmin(pred_train3_ns, 1 - epsilon), epsilon)
pearson_residuals3_ns <- (y_train_all3 - y_pred_prob_safe3) / 
  sqrt(y_pred_prob_safe3 * (1 - y_pred_prob_safe3))

# Test for spatial autocorrelation
moran_test3_ns <- moran.test(pearson_residuals3_ns, weights3)
print(moran_test3_ns)


cv_res_Lb_ns = cv_res_Lb_ns |> 
  mutate(mean_train_auc = mean(c(train_auc1_ns,train_auc2_ns,train_auc3_ns)),
         sd_train_auc = sd(c(train_auc1_ns,train_auc2_ns,train_auc3_ns)),
         mean_test_auc = mean(c(test_auc1_ns, test_auc2_ns, test_auc3_ns)),
         sd_test_auc = sd(c(test_auc1_ns, test_auc2_ns, test_auc3_ns)),
         mean_space = "NA",
         low_moran = min(c(moran_test1_ns$estimate[[1]],
                           moran_test2_ns$estimate[[1]],
                           moran_test3_ns$estimate[[1]])),
         hi_moran = max(c(moran_test1_ns$estimate[[1]],
                          moran_test2_ns$estimate[[1]],
                          moran_test3_ns$estimate[[1]])),
         low_p = min(c(moran_test1_ns$p.value[[1]],
                       moran_test2_ns$p.value[[1]],
                       moran_test3_ns$p.value[[1]])),
         hi_p = max(c(moran_test1_ns$p.value[[1]],
                      moran_test2_ns$p.value[[1]],
                      moran_test3_ns$p.value[[1]])))

cv_res = rbind(cv_res, cv_res_Lb,cv_res_Lb_ns)

write.csv(cv_res, "data/Nested_CV_results_forest.csv")
# cv_res = read.csv("data/Nested_CV_results_forest.csv")

cv_res_ran_ns = cv_res[8,]

# final model - need to choose the right params from cv_res
final_model_all <- gpboost(
  data = d_all,
  gp_model = NULL,
  nrounds = round(cv_res_ran_ns$best_iter,0),
  learning_rate = cv_res_ran_ns$learning_rate,
  num_leaves = cv_res_ran_ns$num_leaves,
  min_data_in_leaf = cv_res_ran_ns$min_data_in_leaf,
  lambda_l2 = cv_res_ran_ns$lambda_l2,
  verbose = 1
)

saveRDS.gpb.Booster(final_model_all, "saved models/Non_Sp_LargeB_forest_final.rds")

pred_all <- predict(final_model_all, 
                    data = as.matrix(Lands_forest[, covars_forest]),
                    gp_coords_pred = as.matrix(Lands_forest[, c("x", "y")]),
                    predict_var = FALSE)

all_auc <- auc(roc(y_all, pred_all))

# Pearson residuals 
epsilon <- 1e-10
y_pred_prob_safe_all <- pmax(pmin(pred_all, 1 - epsilon), epsilon)
pearson_residuals_all <- (y_all - y_pred_prob_safe_all) / 
  sqrt(y_pred_prob_safe_all * (1 - y_pred_prob_safe_all))

# Create spatial neighbors 

coords_all <- cbind(Lands_forest[,"x"], Lands_forest[,"y"])
knn_all <- knearneigh(coords_all, k = 12)  # adjust k based on your data
nb_all <- knn2nb(knn_all)
weights_all <- nb2listw(nb_all, style = "W")

# Test for spatial autocorrelation
moran_test_all <- moran.test(pearson_residuals_all, weights_all)
print(moran_test_all)
