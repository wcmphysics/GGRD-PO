# Run the wafer pipeline

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe src/main.py --config src/config/config.yaml --headless
```

Edit `src/config/config.yaml` to set `enabled: true` for the stages you want. Each stage's `config` points to its settings file. The CLI runs enabled stages in the order below. Paths are relative to the repository root. A disabled stage's settings file is skipped.

| Stage | Settings and inputs |
|---|---|
| `generate` | `src/config/pseudo_data_config.yaml` writes the synthetic history CSV. Enable this stage if the demo CSV is missing. |
| `train` | `src/config/train_demo.yaml` reads the CSV and writes evaluation results and a model bundle. |
| `predict` | `src/config/predict_demo.yaml` reads a saved bundle and a JSON file of process sequences. |
| `optimize` | `src/config/optimize_free_demo.yaml` searches for a recipe using a saved bundle. Select `src/config/optimize_fixed_demo.yaml` to run with pinned recipe settings. |

To run the demo from generation through optimization, enable all four stages. The demo's prediction and search settings already point to the bundle written by training. To use a previously fitted model, disable training and set `bundle_directory` in the enabled consumer settings. If you change the training output directory and want to use the new model in the same run, update the consumer's bundle path to match. Training, prediction, and search each require a fresh output directory.

The stage calls are in `src/main.py`. A direct `--config src/config/pseudo_data_config.yaml` invocation still runs generation alone.

## Base model settings

For the expanded stack, install the CPU dependencies listed in `requirements.txt` and set the `train` stage's `config` to `src/config/train_all_models.yaml`. That profile uses a separate training output directory. Update `bundle_directory` in predict and optimize settings if those stages should consume the expanded stack. The order of `models` determines the order of columns in the stack. Choose at least two models and give each a unique name.

| `estimator` | Required entries under `parameters` | Notes |
|---|---|---|
| `ridge` | `alpha` | Nonnegative penalty. |
| `random_forest` | `n_estimators`, `min_samples_leaf`, `n_jobs` | Use `n_jobs: 1`. Search can vary the other two settings. |
| `elastic_net` | `alpha`, `l1_ratio`, `max_iter`, `tol`, `selection` | Use a positive `alpha` and `0 < l1_ratio <= 1`. `selection` is `cyclic` or `random`. |
| `gaussian_process` | `kernel`, `alpha`, `optimizer`, `n_restarts_optimizer`, `normalize_y` | `kernel` declares `kind: rbf` or `kind: matern`, positive `length_scale`, and `length_scale_bounds` as two increasing positive numbers or `fixed`. Matérn requires `nu` of `0.5`, `1.5`, or `2.5`. Choose `optimizer: fmin_l_bfgs_b` or `null`. Use zero restarts with a fixed kernel or disabled optimizer. The grid may vary only `alpha`, `n_restarts_optimizer`, or `normalize_y`. This base supplies point predictions to the stack. |
| `knn` | `n_neighbors`, `weights`, `p` | `weights` is `uniform` or `distance`. Use `p: 1` for Manhattan or `p: 2` for Euclidean distance. Every evaluated neighbor count must fit inside the smallest applicable training subset, including grouped tuning folds. |
| `svr` | `C`, `epsilon`, `kernel`, `gamma`, `degree`, `coef0` | `kernel` is `rbf`, `linear`, `poly`, or `sigmoid`. `gamma` is `scale`, `auto`, or a positive number. `degree` applies to `poly`; `coef0` applies to `poly` and `sigmoid`. |
| `lightgbm` | `n_estimators`, `learning_rate`, `reg_alpha`, `reg_lambda`, `num_leaves`, `min_child_samples`, `max_depth` | Use `max_depth: -1` for unrestricted depth. |
| `catboost` | `iterations`, `learning_rate`, `depth`, `l2_leaf_reg` | Use depth `1` through `16`. |
| `xgboost` | `n_estimators`, `learning_rate`, `reg_alpha`, `reg_lambda`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree` | Sampling fractions are in `(0, 1]`. |

All bases use fold-local imputation and categorical encoding. Numeric scaling also applies to Ridge, Elastic Net, Gaussian process, KNN, and SVR. The booster adapters use one CPU thread with no early stopping based on validation labels. Their configured iteration count is an upper bound. All of these models use point predictions in the same target unit.

Set `enable_search: false` and `search_grid: null` for fixed settings. To run grouped grid search, set `enable_search: true` and specify one or more allowed parameter names, each with a nonempty list of candidates. Other parameters retain their fixed values. For example, use `search_grid: {C: [0.5, 2.0], epsilon: [0.05, 0.1]}` for SVR. Each search runs inside its own allowed fit rows.

## Training diagnostics

Set `analysis.convergence` in the training YAML to control convergence reporting. `analysis.convergence_every` controls the saved step interval and retains the first and last point. The run saves points to `analysis/convergence.csv`, fit availability to `analysis/convergence_fits.csv`, and plots to `figures/convergence_*.png`.

The figures show each individual fit and the median among curves with a recorded point at that step. Inner OOF fits, outer refits, and final fits have separate panels. The plots cover the primary feature set. With grouped tuning, they show the selected model's refit in each fit context. Validation curves score fitted checkpoints on the corresponding grouped holdout after training and never control stopping or selection. Final full-data refits have no validation curve.

| Estimator | Meaning of a step and plotted loss |
|---|---|
| Random Forest | One additional fitted tree. MSE evaluates prefixes of the ensemble. |
| LightGBM, CatBoost, XGBoost | One fitted boosting round. MSE is recomputed from each round's cumulative prediction. |
| Elastic Net | One coordinate descent sweep. Shows the penalized training objective and prediction MSE. The traceable dense solver uses the Elastic Net objective and a duality gap stopping check, with its own random coordinate sequence when `selection: random`. |
| Gaussian process | One accepted L-BFGS-B iteration. Shows the negative log marginal likelihood for each optimizer restart on the fit rows. A fixed kernel or disabled optimizer has no such trace. |
| Meta-model | One major SLSQP iteration. Shows the MSE plus weight penalty on the inner OOF prediction matrix used to fit the weights. |
| Ridge, KNN, SVR | The configured Ridge and SVR APIs do not expose intermediate costs, and KNN has no iterative training objective. The fit availability report lists these cases. |

`figures/held_out_parity.png` shows the same held-out predictions in a full-range panel and a percentile-zoomed panel. The zoom reports how many points fall inside its limits. Scores still use all eligible held-out rows.
