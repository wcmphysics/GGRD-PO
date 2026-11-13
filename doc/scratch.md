# === Stage A: System and Config ===
# Step 1: import and set up working directories
# Step 2: Configuration setup (with YAML files)

# === Stage B: Data and Features ===
# Step 1: Pseudo-data generation / Data reading
# Step 2: Data pre-processing [eg: normalization/outlier removal]
# Step 3: Feature engineering (with dynamic function for later)
#   Accumulated physical features (eg: total thermal budget)
#   Features from Physical equations/intuition
#   Features from process knowledge
#   PCA and SISSO (need permutation test)
#   Feature correlation and removal [add flag for drop or not by correlation metrics, eg: drop_pearson]
# Step 4: Model-dedicated data re-packing
# Step 5: Train/val/test data splitting [check how to do cross validation without data leakage in this architecture]
# Step 6: Dataset and data loader build-up

# === Stage C: Models and Cost ===
# Step 1: Cost function definition for base models
#   MSE
#   Physical-law violation penalty
# Step 2: Base model definition (in order of priority)
#   baseline = mean or median
#   GP
#   Elastic Net with/without polynomial features
#   K-nearest neighbors
#   Tree-based (Random forest, XGBoost, LightGBM, CatBoost)
#   SVM
#   Type-Specific Projector + GRU
#   Transformer
#   DGP
#   Graph-based models
# Step 3: Hyper-paramter search (HPS) definition
# Step 4: Cost function definition for meta models
# Step 5: Meta model definition
#   baseline = mean or median
#   Elastic Net (with non-negative/sum-to-one weight)
#   XGBoost/LightGBM with strong regularization [depth<=3, small branching]

# === Stage D: Training ===
# Step 1: Base model training wrapped inside HPS
# Step 2: Meta model training 

# === Stage E: Reverse Search ===
# Step 1: Score defnition
# Step 2: Search