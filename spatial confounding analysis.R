# SUPPLEMENTARY ANALYSIS: Spatial confounding, CV leakage, and variable
# importance sensitivity
# Written by Mark Gillespie with help from Claude AI Sonnet 5 for writing 
# functions and loops

library(tidyverse)
library(gpboost)
library(ggplot2)
library(gridExtra)
library(sf)
library(spatialsample)
library(spdep)
library(gstat)
library(sp)
library(FNN)
library(corrplot)

# load in needed items
# dataset
Lands_main = read.table("data/Landslides_main_update.txt", header=T, sep="\t", stringsAsFactors = T)

# transform precip intensity
Lands_main$max_precip_3h = sqrt(Lands_main$max_precip_3h)

# rename variables (optional)
Lands_main = Lands_main |> 
  rename(Elevation = elevation,
         Slope= slope,
         `Relative precipitation`= relprecip ,
         `Profile curvature`  =profcurv ,
         `Planar curvature` =  plancurv,
         Easterness  = aspect_sin ,
         Northerness =  aspect_cos,
         Groundwater = groundwater,
         Precipitation =  precip ,
         `Precipitation intensity` =  max_precip_3h,
         `Deposit thickness` = deposit_thickness,
         Bedrock   = bedrock2 ,
         Landform   = landforms2,
         `Deposit class`    = deposit2 ,
         `Flow accumulation (log)`   =  LogFlow )

# make a vector of covariate names and categorical variables
covars_main = names(Lands_main)[c(5:19)]
cat_feat_main <- colnames(Lands_main)[which(colnames(Lands_main) %in% c("Landform", "Bedrock",
                                                                        "Deposit thickness", "Deposit class"))]

# also needed - the Fold ids from the four different schemes
# spatial:
sf_Lands_main <- st_as_sf(Lands_main, 
                          coords = c("x", "y"), crs = 32632)

set.seed(13)
spatial_folds <- spatial_clustering_cv(
  data = sf_Lands_main,
  v = 5,  # 5 folds
  cluster_function = "kmeans"
)

extract_folds = function(spatial_cv_object, n_data) {
  fold_ids = rep(NA, n_data)
  for(i in 1:length(spatial_cv_object$splits)) {
    ass_idx = setdiff(1:n_data, spatial_cv_object$splits[[i]]$in_id)
    fold_ids[ass_idx] = i 
  }
  return(fold_ids)
}

spatial_fold_id <- extract_folds(spatial_folds, nrow(Lands_main))


# large block CV
set.seed(123)

block_cv_large <- spatial_block_cv(
  data = sf_Lands_main,
  v= 5,
  n = c(1,5)
)

large_fold_id <- extract_folds(block_cv_large, nrow(Lands_main))

# small block CV
set.seed(25)
block_cv_small <- spatial_block_cv(
  data = sf_Lands_main,
  v= 5,
  n = c(7,7)
)
autoplot(block_cv_small)

small_fold_id <- extract_folds(block_cv_small, nrow(Lands_main))

# random CV
library(caret)

# Create random folds with caret
set.seed(123)
random_folds <- createFolds(sf_Lands_main$landslide, 
                            k = 5, 
                            returnTrain = FALSE)

# Convert to blockCV format (fold_ids vector)
n <- nrow(sf_Lands_main)
random_fold_id <- integer(n)  # Initialize vector

for (i in 1:length(random_folds)) {
  random_fold_id[random_folds[[i]]] <- i
}

# combine all the fold ids
cv_scheme_folds <- list(
  random             = random_fold_id,
  small_block        = small_fold_id,
  large_block        = large_fold_id,
  spatial_clustering = spatial_fold_id
)

# convert to list format
cv_scheme_fold_lists <- lapply(cv_scheme_folds, function(fid) {
  v <- length(unique(fid))
  lapply(1:v, function(i) which(fid == i))
})

#### also needed is the data converted into formats needed by GPBoost
# actually, probably only X_all and coords_all is needed
X_all <- as.matrix(Lands_main[, covars_main])
y_all <- Lands_main[, "landslide"]
coords_all <- as.matrix(Lands_main[, c("x", "y")])

d_all <- gpb.Dataset(data = X_all,
                     label = y_all,
                     categorical_feature = cat_feat_main)

gp_model_final_all <- GPModel(gp_coords = coords_all, 
                              likelihood = "bernoulli_probit",
                              cov_function = "matern" ,
                              cov_fct_shape = 1.5)

#===============================================================================
# 1. CORRELATION MATRIX (Appendix C)
#===============================================================================

# As there is a mixture of continuous, nominal and ordinal variables, 
# the correlation matrix should be a mixture of Spearmans rank coefficients
# Cramers V and eta from Anova results

# save variable names as appropriate
ordinal_vars <- c("Deposit thickness")
nominal_vars <- c("Bedrock", "Landform", "Deposit class")  
continuous_vars <- setdiff(covars_main, c(ordinal_vars, nominal_vars))

spearman_vars <- c(continuous_vars, ordinal_vars)


# --- Spearman: continuous + ordinal ---
spearman_mat <- cor(Lands_main[, spearman_vars], method = "spearman", use = "pairwise.complete.obs")

# --- Cramer's V calculation: nominal-nominal pairs ---
cramers_v <- function(x, y) {
  tbl <- table(x, y)
  chi2 <- suppressWarnings(chisq.test(tbl, correct = FALSE)$statistic)
  n <- sum(tbl); k <- min(nrow(tbl), ncol(tbl))
  sqrt(as.numeric(chi2) / (n * (k - 1)))
}

nominal_nominal_pairs <- if (length(nominal_vars) > 1) {
  t(combn(nominal_vars, 2))
} else {
  matrix(nrow = 0, ncol = 2)
}

cramers_results <- if (nrow(nominal_nominal_pairs) > 0) {
  data.frame(v1 = nominal_nominal_pairs[,1], v2 = nominal_nominal_pairs[,2]) %>%
    rowwise() %>%
    mutate(estimate = cramers_v(Lands_main[[v1]], Lands_main[[v2]]),
           method = "cramers_v", p_value = NA_real_) %>%
    ungroup()
} else {
  tibble(v1 = character(), v2 = character(), estimate = numeric(),
         method = character(), p_value = numeric())
}

# --- Eta (correlation ratio): nominal vs continuous/ordinal pairs ---
eta_squared <- function(continuous_var, nominal_var) {
  fit <- aov(continuous_var ~ as.factor(nominal_var))
  ss <- summary(fit)[[1]][["Sum Sq"]]
  p  <- summary(fit)[[1]][["Pr(>F)"]][1]
  list(eta = sqrt(ss[1] / sum(ss)), p_value = p)
}

nominal_continuous_pairs <- expand.grid(v1 = nominal_vars, v2 = spearman_vars,
                                        stringsAsFactors = FALSE)

eta_results <- nominal_continuous_pairs %>%
  rowwise() %>%
  mutate(res = list(eta_squared(Lands_main[[v2]], Lands_main[[v1]]))) %>%
  ungroup() %>%
  mutate(estimate = purrr::map_dbl(res, "eta"),
         p_value  = purrr::map_dbl(res, "p_value"),
         method = "eta") %>%
  select(-res)

# --- Combine everything for the Appendix C table  ---
spearman_long <- as.data.frame(as.table(spearman_mat)) %>%
  rename(v1 = Var1, v2 = Var2, estimate = Freq) %>%
  mutate(method = "spearman", p_value = NA_real_) %>%
  filter(v1 != v2)

correlation_appendix_C <- bind_rows(spearman_long, cramers_results, eta_results) %>%
  mutate(v1 = as.character(v1), v2 = as.character(v2)) %>%
  distinct(v1, v2, .keep_all = TRUE) %>%
  arrange(desc(abs(estimate)))

print(head(correlation_appendix_C, 15))

# create a p-value matrix for the corrplot to only print values for 
# significant correlations
p_matrix <- cor.mtest(Lands_main[, spearman_vars], method = "spearman", conf.level = 0.95)$p

# --- Plot 1: corrplot for the Spearman matrix only ---
corrplot(spearman_mat, method = "circle", type = "upper", tl.col = "black",
         tl.srt = 45, 
         col = colorRampPalette(c("#3B4CC0", "white", "#B40426"))(200),
         title = "Spearman correlations (continuous & ordinal predictors)",
         mar = c(0,0,2,0), addCoef.col = "black", number.cex = 0.7,
         p.mat = p_matrix,        # Supplies the p-value matrix
         sig.level = 0.05,        # Defines the alpha threshold for significance
         insig = "blank" , diag = F)


# --- Plot 2: UNSIGNED magnitude heatmap for anything involving nominal variables ---
# Deliberately kept separate from Plot 1: Cramer's V and eta are bounded 0-1 with
# no meaningful sign, so putting them on the same diverging red/blue scale as
# Spearman would visually imply a "direction" that doesn't exist for these pairs.
nominal_assoc_plot_data <- bind_rows(cramers_results, eta_results) %>%
  mutate(pair = paste(v1, v2, sep = " - "))

nominal_assoc_plot <- ggplot(nominal_assoc_plot_data,
                             aes(x = v2, y = v1, fill = estimate)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", estimate)), size = 3) +
  scale_fill_gradient(low = "white", high ="#B40426", guide = "colourbar") +
  labs(x = NULL, y = NULL,
       title = "Associations involving nominal predictors (unsigned, 0-1)",
       fill = "Association\n(Cramer's V / eta)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

nominal_assoc_plot

#===============================================================================
# 2. SPATIAL AUTOCORRELATION RANGE (variograms) FOR ALL PREDICTORS
#===============================================================================
# Compare the effective spatial range of each top predictor to your spatial
# CV block size, to test whether "smooth" predictors have
# autocorrelation ranges comparable to (or larger than) your CV block scale
# -- which is the mechanism that would let them leak under random CV.

# This function returns the full fit.variogram() object (not just $range) so we can later
# evaluate correlation AT a specific distance, and flag unstable fits.
get_variogram_fit <- function(varname, data, coords) {
  df <- data.frame(coords, z = as.numeric(data[[varname]]))
  coordinates(df) <- ~x + y
  vgm_emp <- variogram(z ~ 1, df)
  fit <- tryCatch(fit.variogram(vgm_emp, vgm("Sph")), error = function(e) NULL)
  list(fit = fit, empirical = vgm_emp)
}

variogram_fits_raw <- lapply(covars_main, get_variogram_fit,
                             data = Lands_main, coords = coords_all)
names(variogram_fits_raw) <- covars_main
variogram_fits <- lapply(variogram_fits_raw, `[[`, "fit")  # named list of fit.variogram objects, used in section 3b

# example of a fit
variogram_fits$Elevation

# --- Stability diagnostic: nugget:sill ratio ---
# A high nugget relative to total sill means most variance is unstructured
# noise with little spatial structure - the model has little real plateau to
# lock onto, and a large fitted "range" in that situation is often a fitting
# artifact rather than genuine long-range smoothness. Flag anything > 0.6 for
# visual inspection before reporting its range/ratio anywhere.
variogram_summary <- purrr::map_dfr(names(variogram_fits), function(v) {
  fit <- variogram_fits[[v]]
  if (is.null(fit) || nrow(fit) < 2) {
    return(data.frame(variable = v, effective_range = NA_real_,
                      nugget = NA_real_, partial_sill = NA_real_,
                      nugget_to_sill_ratio = NA_real_, unstable_fit = TRUE))
  }
  nugget <- fit$psill[fit$model == "Nug"]
  nugget <- if (length(nugget) == 0) 0 else nugget
  partial_sill <- sum(fit$psill[fit$model != "Nug"])
  total_sill <- nugget + partial_sill
  data.frame(variable = v,
             effective_range = fit$range[fit$model != "Nug"][1],
             nugget = nugget, partial_sill = partial_sill,
             nugget_to_sill_ratio = nugget / total_sill,
             unstable_fit = (nugget / total_sill) > 0.6)
})


print(variogram_summary)

if (any(variogram_summary$unstable_fit, na.rm = TRUE)) {
  message("Flagged as unstable (nugget:sill > 0.6) - inspect empirical variograms before trusting their range: ",
          paste(variogram_summary$variable[which(variogram_summary$unstable_fit)], collapse = ", "))
}

# Quick visual check for any flagged variable, e.g.:
plot(variogram_fits_raw[["Planar curvature"]]$empirical,
      variogram_fits_raw[["Planar curvature"]]$fit)

plot(variogram_fits_raw[["Relative precipitation"]]$empirical,
     variogram_fits_raw[["Relative precipitation"]]$fit)

plot(variogram_fits_raw[["Flow accumulation (log)"]]$empirical,
     variogram_fits_raw[["Flow accumulation (log)"]]$fit)

#===============================================================================
# 3. TRAIN-TEST SPATIAL LEAKAGE ACROSS ALL 4 CV SCHEMES
#===============================================================================
nn_distances_for_scheme <- function(fold_assignment, coords) {
  v <- length(unique(fold_assignment))
  purrr::map_dfr(1:v, function(f) {
    train_idx <- which(fold_assignment != f)
    test_idx  <- which(fold_assignment == f)
    nn <- get.knnx(coords[train_idx, , drop = FALSE], coords[test_idx, , drop = FALSE], k = 1)
    data.frame(fold = f, nn_dist = nn$nn.dist[, 1])
  })
}

leakage_all <- purrr::imap_dfr(cv_scheme_folds, function(fid, scheme_name) {
  nn_distances_for_scheme(fid, coords_all) %>% mutate(cv_scheme = scheme_name)
})

leakage_summary <- leakage_all %>%
  group_by(cv_scheme) %>%
  summarise(median_nn_dist = median(nn_dist),
            q10 = quantile(nn_dist, 0.1),
            q90 = quantile(nn_dist, 0.9)) %>%
  arrange(median_nn_dist) 

print(leakage_summary)

leakage_plot <- ggplot(leakage_all, aes(x = nn_dist, fill = cv_scheme)) +
  geom_density(alpha = 0.4) +
  labs(x = "Distance to nearest training point", y = "Density",
       title = "Train-test spatial proximity across CV schemes") +
  theme_bw()


#===============================================================================
# 3b. RESIDUAL CORRELATION AT THE HELD-OUT DISTANCE (replaces raw range ratio)
#===============================================================================
# The nominal "range" is where the variogram plateaus (correlation ~ 0), but
# two predictors can share a range while decaying at very different rates
# beforehand. What actually matters for leakage risk is how much spatial
# correlation is STILL PRESENT specifically at the distance each CV scheme
# holds test points out to - so evaluate the fitted variogram AT median_nn_dist
# and convert to a correlation, rather than comparing ranges directly.

get_correlation_at_distance <- function(fit, d) {
  if (is.null(fit) || is.na(d)) return(NA_real_)
  gamma_d <- gstat::variogramLine(fit, dist_vector = d)$gamma
  total_sill <- sum(fit$psill)
  1 - gamma_d / total_sill
}

range_vs_leakage <- expand.grid(variable = variogram_summary$variable,
                                cv_scheme = names(cv_scheme_folds),
                                stringsAsFactors = FALSE) %>%
  left_join(variogram_summary, by = "variable") %>%
  left_join(leakage_summary %>% select(cv_scheme, median_nn_dist), by = "cv_scheme") %>%
  rowwise() %>%
  mutate(rho_at_nndist = get_correlation_at_distance(variogram_fits[[variable]], median_nn_dist),
         risk_band = case_when(
           unstable_fit                ~ "unstable fit - inspect before interpreting",
           is.na(rho_at_nndist)        ~ NA_character_,
           rho_at_nndist < 0.2         ~ "low (effectively decorrelated)",
           rho_at_nndist < 0.5         ~ "moderate",
           TRUE                        ~ "high (substantial residual structure)"
         )) %>%
  ungroup()

print(range_vs_leakage %>% select(variable, cv_scheme, median_nn_dist, rho_at_nndist, risk_band))

#===============================================================================
# 4. FACTORIAL IMPORTANCE: 4 CV SCHEMES x 2 STRUCTURES (8 CELLS)
#===============================================================================
# If you already have saved final models from your original 8-model analysis,
# populate saved_model_paths below and this will load rather than refit.
# Any cell left as NA will be fit fresh (slow — gpb.cv + gpboost per cell).

saved_model_paths <- list(
  random__no_GP             = "saved models/Non_Sp_random_full_final.rds",
  random__GP                = "saved models/Sp_random_full_final.rds",
  small_block__no_GP        = "saved models/Non_Sp_SmallB_full_final.rds",
  small_block__GP           = "saved models/Sp_SmallB_full_final.rds",
  large_block__no_GP        = "saved models/Non_Sp_LargeB_full_final.rds",
  large_block__GP           = "saved models/Sp_LargeB_full_final.rds",
  spatial_clustering__no_GP = "saved models/Non_Sp_sp_full_final.rds",
  spatial_clustering__GP    = "saved models/Sp_sp_full_final.rds"  
)

#Fallback only: each cell's own already-selected hyperparameters (from your
# original tuning), used solely if saved_model_paths[[cell]] is missing/not found.
cell_hyperparams <- list(
  random__no_GP             = list(learning_rate = NA, min_data_in_leaf = NA, num_leaves = NA, lambda_l2 = NA, max_depth = -1),
  random__GP                = list(learning_rate = NA, min_data_in_leaf = NA, num_leaves = NA, lambda_l2 = NA, max_depth = -1),
  small_block__no_GP        = list(learning_rate = NA, min_data_in_leaf = NA, num_leaves = NA, lambda_l2 = NA, max_depth = -1),
  small_block__GP           = list(learning_rate = NA, min_data_in_leaf = NA, num_leaves = NA, lambda_l2 = NA, max_depth = -1),
  large_block__no_GP        = list(learning_rate = NA, min_data_in_leaf = NA, num_leaves = NA, lambda_l2 = NA, max_depth = -1),
  large_block__GP           = list(learning_rate = NA, min_data_in_leaf = NA, num_leaves = NA, lambda_l2 = NA, max_depth = -1),
  spatial_clustering__no_GP = list(learning_rate = NA, min_data_in_leaf = NA, num_leaves = NA, lambda_l2 = NA, max_depth = -1),
  spatial_clustering__GP    = list(learning_rate = NA, min_data_in_leaf = NA, num_leaves = NA, lambda_l2 = NA, max_depth = -1)
)
 
fit_cell <- function(folds_list, use_gp_model, saved_path, cell_params, cell_label) {
  if (!is.na(saved_path) && file.exists(saved_path)) {
    return(readRDS.gpb.Booster(saved_path))
  }
  message("No saved model found for ", cell_label, " - fitting fresh using this cell's own hyperparameters.")
  if (any(is.na(unlist(cell_params[c("learning_rate","min_data_in_leaf","num_leaves","lambda_l2")])))) {
    stop("cell_hyperparams for '", cell_label, "' is incomplete (still NA) - either supply saved_model_paths ",
         "or fill in this cell's tuned hyperparameters before refitting.")
  }
  gp_arg <- if (use_gp_model) gp_model_final_all else NULL
  cv_res <- gpb.cv(params = cell_params, data = d_all, gp_model = gp_arg,
                   folds = folds_list, nrounds = 1000, eval = "auc",
                   early_stopping_rounds = 20, use_gp_model_for_validation = use_gp_model)
  gpboost(data = d_all, gp_model = gp_arg, nrounds = cv_res$best_iter,
          learning_rate = cell_params$learning_rate,
          min_data_in_leaf = cell_params$min_data_in_leaf,
          num_leaves = cell_params$num_leaves,
          lambda_l2 = cell_params$lambda_l2, verbose = 0)
}

cell_grid <- expand.grid(cv_scheme = names(cv_scheme_folds),
                         structure = c("no_GP", "GP"), stringsAsFactors = FALSE) %>%
  mutate(cell_label = paste(cv_scheme, structure, sep = "__"))

library(SHAPforxgboost)

# extract importance as absolute mean shap score
get_shap_importance <- function(model, X) {

  shap_res <- shap.values(xgb_model = model, X_train = X)
  list(
    summary = data.frame(Feature = names(shap_res$mean_shap_score),
                         mean_abs_shap = as.numeric(shap_res$mean_shap_score)),
    raw = shap_res$shap_score  # per-observation matrix, kept for bootstrap comparison below
  )
}

factorial_shap_raw <- list()  # cell_label -> raw per-observation SHAP matrix

factorial_importance <- purrr::pmap_dfr(cell_grid, function(cv_scheme, structure, cell_label) {
  model <- fit_cell(cv_scheme_fold_lists[[cv_scheme]],
                    use_gp_model = (structure == "GP"),
                    saved_path = saved_model_paths[[cell_label]],
                    cell_params = cell_hyperparams[[cell_label]],
                    cell_label = cell_label)
  res <- get_shap_importance(model, X_all)
  factorial_shap_raw[[cell_label]] <<- res$raw
  res$summary %>% mutate(cv_scheme = cv_scheme, structure = structure)
})


factorial_plot_data <- factorial_importance %>%
  mutate(cv_scheme = factor(cv_scheme, levels = c("random", "small_block",
                                                  "large_block", "spatial_clustering")),
         structure = factor(structure, levels = c("no_GP", "GP"),
                            labels = c("No spatial GP term", "Spatial GP term")))

factorial_plot_data <- factorial_plot_data |>
  mutate(
    Feature = as.character(Feature),
    Feature = recode(Feature,
                     "elevation" = "Elevation",
                     "slope" = "Slope",
                     "relprecip" = "Relative precipitation" ,
                     "profcurv" = "Profile curvature",
                     "plancurv" = "Planar curvature",
                     "aspect_sin" = "Easterness",
                     "aspect_cos" = "Northerness",
                     "groundwater" = "Groundwater",
                     "precip" = "Precipitation",
                     "max_precip_3h" = "Precipitation intensity",
                     "deposit_thickness" = "Deposit thickness" ,
                     "bedrock2" = "Bedrock",
                     "landforms2" = "Landform",
                     "deposit2" = "Deposit class",
                     "LogFlow" = "Flow accumulation (log)"
    ),
    Feature = factor(Feature)
  )

levels(factor(factorial_plot_data$Feature))

factorial_plot <- ggplot(factorial_plot_data, aes(x = Feature, y = mean_abs_shap, fill = cv_scheme)) +
  geom_col(position = "dodge") +
  facet_wrap(~ structure) +
  coord_flip() +
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9", "#009E73"))+
  labs(x = NULL, y = "Gain (%)", fill = "CV scheme",
       title = "Variable importance across the full CV x structure design") +
  theme_bw()

factorial_plot
