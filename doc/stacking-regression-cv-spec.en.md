# Regression stacking: implementation specification

Version 2.2 · English

This specification defines a Python implementation of regression stacking for approximately 100 labeled samples with a required grouping category. Inner cross-validation produces base predictions for training the meta-regressor. Repeated outer cross-validation compares the stack with simpler predictors on held-out groups. After evaluation, train a single deployment stack on all labeled data.

## 1. Inputs, defaults, and split policy

| Input or setting | Requirement |
|---|---|
| `X`, `y` | One feature row and one finite numeric target per sample, in the same order. |
| `groups` or `group_col` | Required grouping information. Supply a one-dimensional `groups` vector aligned with `X` or identify the grouping column in the source DataFrame with `group_col`. Equal labels form one group. Labels can repeat and must be non-missing. Exclude the grouping column from model features by default. |
| `row_id` | Optional source identifier for exports and joins back to the source data. If supplied, it must be unique, non-null, and aligned with `X`. Otherwise use the original input position as the reported `row_id`. Exclude audit IDs and the target from model features. |
| `base_models` | An ordered list of at least two named base-regressor configurations, with feature selections, preprocessing, and optional search settings. Every base model must be able to predict every required row. |
| `meta_model` | A supplied regression configuration and any preprocessing for the base prediction columns. The meta-model receives only those columns, in the declared base-model order. A fixed Ridge configuration is a reasonable starting point if a meta-model has yet to be chosen. |
| `outer_folds`, `inner_folds`, `outer_repeats` | Defaults are `5`, `5`, and `5`. Fold counts must be at least 2 and repeats at least 1. They are configurable. |
| `root_seed` | Required or set by a documented default. Derive reproducible seeds for each outer repeat, each inner split plan, and the final inner split plan. Where supported, assign a reproducible seed to each model configuration and keep it fixed across fits by default. Record the actual seeds. |
| Metrics | MAE is primary. RMSE and MAPE are secondary. MAE and RMSE use the original target units. MAPE is reported in percent on eligible rows. |
| `mape_min_abs_y` | Finite nonnegative threshold in original target units, default `0.0`. A row is eligible when `abs(y_true) > mape_min_abs_y`. Set the threshold before evaluation. |
| Base hyperparameter search | Disabled by default. Enable per base model with a predeclared search space and budget. Set `search_folds` to at least 2 (start with 3 if feasible). See §4.1. |
| Python dependencies | Implement in Python with NumPy, pandas for DataFrame input and result tables, and scikit-learn `>=1.6` for shuffled `GroupKFold`. Record the installed versions. |

Construct the inputs in one aligned order. For a DataFrame, extract features, target, groups, and optional IDs from the same rows. For separate arrays, validate the dimensions and lengths. The caller must supply the same order for all arrays; equal lengths cannot detect a permutation. Reject missing group labels before splitting. If labels need encoding, map each distinct label to a unique integer code and retain the mapping for auditing.

### 1.1. Row position, source ID, and group label

The ordered inputs already provide a position for each observation. Assign an internal `row_pos = np.arange(len(X))` once, then carry those original positions through all subsets. A unique source `row_id` only helps with external auditing and joins across row reordering or dataset versions. A group label tells the splitter which rows must move together. Two rows can share a group label or even have identical feature values while still occupying distinct positions. Use positional indexing (`.iloc` for pandas, integer-array indexing for NumPy) rather than treating a DataFrame index label as its position. If using generated IDs, record the input dataset version and row order because those IDs are local to that ordering.

### 1.2. Grouped split plan

Use `GroupKFold(n_splits=..., shuffle=True, random_state=seed)` for each outer repeat and for the inner plan inside each outer training subset. Use the same splitter for the final inner OOF plan on all labeled data. Within every train/holdout split, the sets of group labels must be disjoint. Each complete plan assigns every available group, and therefore every row, to exactly one holdout fold. Share each outer plan across all comparison candidates and each inner plan across all base models. A group may be held out in one fold and used for training in other folds, and the same rule applies separately to each repeat.

`GroupShuffleSplit` also keeps groups intact within an individual split. Its independently sampled holdouts may overlap or leave some groups untested, so use a grouped K-fold partition for the complete-coverage outer repeats and inner OOF matrix. One group shuffle split is suitable for an optional early-stopping holdout inside the rows already allowed for that fit. [GroupKFold reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) · [GroupShuffleSplit reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)

```python
import numpy as np
from sklearn.model_selection import GroupKFold


def build_group_plan(X, groups, *, n_splits, seed):
    # groups is a validated, one-dimensional array aligned with X.
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape[0] != len(X):
        raise ValueError("groups must contain one label per row of X")
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"n_splits={n_splits} requires at least that many "
                         f"distinct groups; found {n_groups}")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X, groups=groups))
```

The helper returns positions relative to the `X` passed to it. For example, with pandas `X` and a NumPy `groups` array, call `build_group_plan(X, groups, n_splits=outer_folds, seed=outer_seed)` for one outer repeat. For each returned `outer_train_idx`, call `build_group_plan(X.iloc[outer_train_idx], groups[outer_train_idx], n_splits=inner_folds, seed=inner_seed)`. Build the final inner plan from the full inputs with `n_splits=inner_folds` and a separate seed. A new seed is used for each outer repeat. With few groups, different seeds may yield the same outer partition. Record split membership and report the number of distinct outer partitions ignoring fold order. Identical outer partitions can still yield different scores if their inner partitions differ.

Preflight the outer, inner, and final split plans before fitting, including search splits (§4.1) or early-stopping splits when configured. The full dataset needs at least `outer_folds` distinct groups and each actual outer training subset must have at least `inner_folds`. The full dataset also needs at least `inner_folds` for final OOF. For example, five total groups allow outer 5-fold, but leave just four groups in each outer training subset, so inner 5-fold is impossible. Validate actual row and group counts, including model minimum training sizes. Fail with the repeat, fold, and observed counts if a plan is infeasible. Adjust the fold settings explicitly and keep group isolation. Unequal group sizes can lead to unequal row counts across folds.

Handle missing features within each model's train-fitted pipeline. Fit any imputer only on the training rows permitted for that call. Fail with a diagnostic if any model cannot produce a finite prediction for an eligible row.

## 2. Candidates and the outer-fold comparison

Evaluate the following candidates on **exactly the same outer test rows**, using the same full outer-training fits for the constituent base models. Each prediction must use the original target units, even if a model fits a transformed target internally.

| Candidate ID | Prediction for a held-out row | Question it answers |
|---|---|---|
| `base:<name>` for each configured base model | The named base model's prediction after fitting on the full outer training set. | Does combining predictions improve on an individual model? |
| `equal_mean` | The arithmetic mean of the base predictions in that row, with weight `1 / number_of_base_models` per model. | Does learning how to combine predictions improve on a fixed, equal-weight rule? |
| `stack` | The meta-regressor's prediction from the same ordered vector of base predictions. | Does the relationship learned from inner OOF predictions generalize to the outer test set? |

Reuse the same fitted full outer-training base pipelines across all candidates, including the fitted winners of any base searches. This isolates how their predictions are combined. Equal averaging has no weights to learn. A candidate that cannot predict a required row invalidates the complete comparison for that run. Surface the failure with its repeat, fold, and model name.

## 3. Train one stack

Implement `fit_stack(train_data, config, fit_context)` for outer evaluation and final training. `train_data` carries aligned `X`, `y`, `groups`, `row_pos`, and exported IDs. The context carries the fit identifier, seeds, and preflighted inner plan for that exact row order. Use `sklearn.base.clone` for scikit-learn models and preprocessing pipelines, or a factory for other estimators.

1. **Use the inner group plan.** In every inner fold, partition `train_data` into `inner_train` and `inner_holdout` with disjoint groups. Allocate an OOF matrix `Z_oof` of shape `[n_train, n_base]` in training row order and declared base model order. Track writes to detect duplicates and omissions.
2. **Generate base OOF predictions.** Fit a fresh base pipeline on `inner_train`. Predict `inner_holdout.X` and write the values at the local holdout positions and base model column. Every row's OOF prediction comes from a base fit that excluded its entire group. This lets the meta-model learn from predictions made on unseen groups.
3. **Validate and fit the meta-regressor.** Confirm every OOF cell was written exactly once, all entries are finite, and rows are in the same order as `train_data.y`. Fit a fresh meta pipeline, including any meta feature scaling. Inner OOF metrics may be saved as base model diagnostics. Scoring the meta-model on the rows it just fit measures training error.
4. **Refit each base model on all of `train_data`.** Use fresh model and preprocessing instances with the configured training procedure. Retain the fitted meta-regressor from step 3. The full-data base fits can use more training rows at inference. Their in-sample predictions are excluded from meta-model training because those predictions would allow the base model's fit on a row to influence its input for that row.
5. **Return the fitted stack.** Include the ordered refitted base pipelines, the fitted meta pipeline, and the input feature and base prediction schemas. Return diagnostics separately if supported.

```text
inner_plan = fit_context.inner_plan
Z_oof = empty_matrix(rows=len(train_data.X), columns=config.base_names)
write_count = zero_matrix_like(Z_oof)

for inner_fold_id, (train_idx, holdout_idx) in enumerate(inner_plan):
    inner_train = train_data.take_positions(train_idx)
    inner_holdout = train_data.take_positions(holdout_idx)
    assert_disjoint_groups(inner_train.groups, inner_holdout.groups)
    for base in config.base_models:
        pipeline = fit_fresh_base(base, inner_train, fit_context, inner_fold_id)
        pred = predict_in_target_units(pipeline, inner_holdout.X)
        validate_predictions(pred, expected_rows=len(holdout_idx))
        write_oof(Z_oof, write_count, holdout_idx, base.name, pred)

assert_all_cells_written_once(Z_oof, write_count)
meta = fit_fresh_meta(config.meta_model, Z_oof, train_data.y, fit_context)
final_bases = [fit_fresh_base(base, train_data, fit_context, "refit")
               for base in config.base_models]
return fitted_stack(final_bases, meta, schemas)
```

Outer indices refer to the full dataset. Inner indices refer to the rows of the current training subset in the order supplied to the splitter. For example, `outer_train_idx[inner_holdout_idx]` maps inner holdout positions to the full dataset. Apply inner indices to the already sliced subset, or map them first if indexing the full dataset. Applying local indices directly to full data can bring outer test rows into training. Preserve prediction batch order when filling OOF cells. Export IDs separately.


If whole-group assignments happen to balance the row counts, 100 rows and five outer folds yield approximately 80 training rows and 20 test rows per fold. Five balanced inner folds would use about 64 rows to predict 16. Use actual counts to allocate OOF rows. Inner base fits use a subset of the training data, and each base refit uses all of it. This size difference is part of the procedure evaluated by outer CV and also applies to final training.

## 4. Fit boundaries and prediction interface

All state learned from data belongs inside the fit that uses it. This includes imputation, scaling, feature selection, encoders, dimensionality reduction, and target transforms. Fit them using the training subset for that call. Transform and predict on holdouts using the fitted state. Predictive features must be available when a new prediction is made. Features that use targets from other rows require their own leakage-safe construction within the relevant training split.

Freeze the meta configuration and the base configurations or search policies before evaluation. Use inner holdout labels only for diagnostics after prediction. For early stopping outside the search workflow, split the current permitted training set by group, for example with `GroupShuffleSplit(n_splits=1, test_size=..., random_state=seed)` and `.split(..., groups=...)`. The size parameter counts groups. Check group disjointness and row counts. Fit trial preprocessing on the fit subset, select the stopping iteration on the validation subset, then fit fresh preprocessing and model on all permitted rows at that fixed count. The permitted rows are `inner_train` for an OOF fit and `train_data` for a full base refit. Use a fixed iteration budget if the model can only create a row-wise internal validation split.

To score a future meta-model or architecture selection procedure on an outer holdout, make the choice wholly inside outer training. Rebuild the stack within each grouped selection split. A sliced, precomputed OOF matrix may contain training rows whose base predictions were influenced by the selection holdout.

Use a shared prediction path. `predict_base_matrix(fitted_stack, X_new)` validates the input schema and returns one finite prediction per row and base, in training column order. `predict_meta(fitted_stack, Z_new)` checks the base column order and returns one finite value per row. `predict_stack(fitted_stack, X_new)` calls both in sequence. For outer evaluation, compute the base matrix once and use it for all candidates. For deployment, preserve the input row order and accept unlabeled features through `predict_stack`.

### 4.1. Optional grouped hyperparameter search for base models

Implement search inside `fit_fresh_base` for the models that enable it. The outer CV measures stack performance, the inner CV produces base OOF predictions, and a separate grouped search CV chooses base parameters within each permitted training subset.

| Base fit context | Data available for search and winner refit | What the winner predicts |
|---|---|---|
| OOF during an outer fold | That OOF split's `inner_train` | That split's `inner_holdout` |
| Refit before scoring an outer fold | All rows in that fold's `outer_train` | `outer_test` |
| OOF for the final meta-model | That final OOF split's `inner_train` | That final split's `inner_holdout` |
| Refit for deployment | All labeled rows | Future data |

Suppose group G is held out for an inner OOF prediction. Tuning once on all of `outer_train` would let G's targets influence the selected parameters. A subsequent base fit that excludes G would still carry information from G through those settings. The outer test remains isolated in this example, but the OOF guarantee is lost. Search separately inside every `inner_train` and search again on the larger training subset for each full base refit. Winners may differ across fits. The outer CV evaluates this same search-and-fit policy, and the final stack uses it again.

1. Build a `GroupKFold` plan on the allowed subset for each search. Preflight all tuned fit contexts, including final training. Check that the actual subset has at least `search_folds` distinct groups and that every split has enough rows for the model. Report the context and counts if infeasible and adjust the configuration explicitly.
2. Search the entire pipeline. Each candidate fits preprocessing on its search-training fold only. If a target transform is used, wrap it so that `predict` returns original target units. Use `GridSearchCV` for a small grid or `RandomizedSearchCV` with a fixed budget (`n_iter`) and seed. Keep candidate order reproducible and use the first candidate among exact score ties.
3. Rank by `neg_mean_absolute_error` and use `best_estimator_` with `refit=True`. The ranking uses the **unweighted mean of search-fold MAEs**, which may differ from the row-weighted MAE used for outer reporting when folds have different row counts. The winner is refitted on all rows allowed for this base fit. Search scores are diagnostics and outer test scores evaluate the full workflow.
4. In the basic search workflow, disable early stopping and use fixed iteration counts. Iteration count can itself be a grid parameter. If adding early stopping later, use an additional group-disjoint validation subset within each search-training fold and within the allowed data for the winner refit. The search validation fold is only for scoring the candidate.

The example passes precomputed, group-safe local indices to `cv` and fits on the same subset in the same order. These splits already incorporate the group labels. [GridSearchCV reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) · [RandomizedSearchCV reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

```python
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV


def search_base(pipeline, param_grid, X_fit, y_fit, groups_fit, *,
                search_folds, seed):
    cv = build_group_plan(X_fit, groups_fit,
                          n_splits=search_folds, seed=seed)
    search = GridSearchCV(
        clone(pipeline), param_grid, cv=cv,
        scoring="neg_mean_absolute_error", refit=True,
        error_score="raise", n_jobs=1,
    )
    search.fit(X_fit, y_fit)
    return search.best_estimator_, search.best_params_, -search.best_score_
```

Return the fitted winner to the existing prediction path. Log the split plan, seed, selected parameters, and search MAE for each search. In an outer fold, share the full outer-training winner's predictions across the individual base, equal mean, and stack.

## 5. Run repeated outer cross-validation

In this document a **repeat** is one complete pass over all outer folds. For every repeat and fold, construct a fresh stack using the outer training subset. The outer test features enter only the prediction path. Join the outer test targets for scoring after generating the predictions.

```text
data = validate_and_align_inputs(X, y, groups, row_id=optional_row_id)
outer_plan, final_context = preflight_group_plans(data, config)

for repeat in outer_plan:
    for fold in repeat.folds:
        outer_train = data.take_positions(fold.train_idx)
        outer_test = data.take_positions(fold.test_idx)
        assert_disjoint_groups(outer_train.groups, outer_test.groups)
        fitted = fit_stack(outer_train, config, fold.fit_context)
        Z_test = predict_base_matrix(fitted, outer_test.X)

        predictions = {}
        for base_name in config.base_names:
            predictions["base:" + base_name] = Z_test.column(base_name)
        predictions["equal_mean"] = row_mean(Z_test)
        predictions["stack"] = predict_meta(fitted, Z_test)

        append_predictions(repeat.id, fold.id, outer_test.row_id,
                           outer_test.groups, outer_test.y, predictions)

    validate_complete_row_and_group_coverage(repeat.id)
    score_each_candidate_on_complete_repeat(repeat.id, config.mape_min_abs_y)
    compute_paired_deltas(repeat.id)

write_evaluation_outputs()
final_stack = fit_stack(data, config, final_context)
save_final_stack(final_stack)
```

With the defaults this runs 25 outer fits of the complete stack, then one final stack fit. With fixed-iteration models and no extra tuning or early stopping, each complete stack fit trains each base model five times for inner OOF and once for refitting. That is 156 fits per base model and 26 meta fits in total. The baseline comparison uses those same fits.

For a tuned base model, each of those 156 fit contexts evaluates `H` settings across `T = search_folds` folds and refits the winner, for `156 × (H × T + 1)` underlying fits without early stopping. Five settings and three search folds give 2,496 fits per tuned base model. Start with a small search on the bases that need it.

## 6. Score and compare

For each candidate and repeat, pair predictions with targets in their original row order, or join exported records by `row_id`. Each candidate must have the same set of test IDs, exactly once per repeat. Let `e_i = y_pred_i - y_true_i`, where `i` ranges over that repeat's complete set of held-out rows:

- `MAE = mean(abs(e_i))`
- `RMSE = sqrt(mean(e_i^2))`
- `MAPE (%) = 100 * mean(abs(e_i) / abs(y_true_i))` over rows with `abs(y_true_i) > mape_min_abs_y`.

Build the MAPE eligibility mask from the true targets once per complete repeat and share it across all candidates. The default threshold excludes exactly zero, for which percentage error is undefined. Small nonzero targets can dominate MAPE. If the target has a meaningful minimum magnitude, set the threshold in original units before evaluation and document it, since it changes which rows the MAPE describes. If no rows qualify, return null (unavailable). Export the eligible and excluded counts. All test rows still contribute to MAE and RMSE. Compute the ratio directly after masking. scikit-learn's MAPE function returns a fraction and substitutes a small denominator at zero. [scikit-learn MAPE reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html)

Compute RMSE from all squared residuals in the repeat. Taking the mean of fold RMSEs applies the square root too early and can give a different result. The default metrics weight rows equally, so groups with more rows contribute more. If each group should count equally, a separate group-macro MAE can be calculated as the mean of each group's MAE. Across repeats, report the arithmetic mean and sample standard deviation of the **per-repeat** metric values. With one repeat, report the standard deviation as unavailable. Take these summaries after scoring each repeat separately. Averaging predictions for the same row across repeats would score a different predictor.

Also compute paired differences in each repeat for `stack` against `equal_mean` and against every `base:<name>`, for each metric. Keep each named base as the same candidate throughout the run. Taking the best base in each test fold would use knowledge of the test targets to switch models.

```text
delta = metric(baseline) - metric(stack)
```

A positive delta favors stacking because lower error is better. Report the deltas per repeat, their mean and sample standard deviation, and the count of repeats with a positive delta out of the total. For example, MAE of 4.0 for `equal_mean` and 3.6 for `stack` gives a delta of +0.4 target units. Pairing on the same splits makes the differences easier to interpret because the candidates face the same test rows. A small gain that changes sign across repeats is weak evidence of an improvement on this dataset. The score spread and win count describe sensitivity to resplitting the same data. Repeats share observations, so these summaries cannot be used as if they came from independent datasets.

MAPE differences are in percentage points. If a repeat has unavailable MAPE, its MAPE delta is also unavailable. Summarize only available values and report the number of valid repeats alongside the total. With zero valid repeats the mean and standard deviation are null. With one valid repeat only the standard deviation is null.

Treat this comparison as an evaluation report. The default training job still produces one final stack. If the report informs a later choice of which candidate to deploy, the winning score has also participated in model selection and may be optimistic. To estimate the performance of an automatic selection policy, predefine the policy and run it only on each outer training subset before scoring the selected candidate on that fold's outer test set. Another option is to evaluate the chosen predictor on newly collected independent data.

## 7. Refit on all data and package for deployment

After evaluation, call the same `fit_stack` with all labeled rows and its own grouped inner plan. Each OOF fit predicts an entire fold of groups excluded from that fit. This creates a **new** OOF matrix with one row per labeled sample, fits a **new** meta-model, and fits each final base model on all labeled rows. Outer fitted models can be discarded after their predictions have been recorded. Deployment prediction uses only the declared feature columns of `X_new`. Split metadata and audit IDs are unnecessary at prediction time.

Save the fitted base and meta pipelines, their order, the feature schema, the model configuration, and training metadata sufficient to reload and check compatibility. Test reload and prediction on unlabeled features. Outer CV estimates performance on held-out groups given each fold's actual training group and row counts. The final fit trains on all groups. Its own independent test score is unavailable in this workflow.

## 8. Required outputs

The implementation may choose storage formats. Use the following fields or document an equivalent schema.

| Output | Required content |
|---|---|
| Run manifest and split records | Data/configuration and package versions, model names and order, feature schema, effective seeds, group-label mapping if encoded, source or generated ID policy, and membership by `row_id` and `group_id` for each outer, inner, final and optional early-stopping split. Record actual train and holdout row and group counts and the number of distinct outer partitions. Identify each inner plan by its parent outer repeat and fold, or the final fit. |
| Outer predictions | `repeat_id`, `outer_fold_id`, `row_id`, `group_id`, `candidate_id`, `y_true`, `y_pred`. Unique key `(repeat_id, row_id, candidate_id)`. |
| Per-repeat metrics | `repeat_id`, `candidate_id`, `n_scored`, `mae`, `rmse`, `mape_pct`, `n_mape_eligible`, `n_mape_excluded`. Record `mape_min_abs_y` in the run config. |
| Metric summary | `candidate_id`, `metric`, `n_repeats`, `n_valid_repeats`, `mean`, `sample_sd`. |
| Paired comparisons | `repeat_id`, `baseline_id`, `metric`, `delta`, plus grouped `mean_delta`, `sample_sd_delta`, `n_positive`, `n_valid_repeats`, `n_repeats`. |
| Search logs (when enabled) | Base model and fit context (outer repeat/fold and inner fold or refit, or final training), search seed, grouped split membership, candidates tried, winning parameters, and mean search-fold MAE. Retain the final winning parameters in deployment metadata. |
| Deployment package | Final fitted stack with metadata and a callable prediction entry point. |

## 9. Acceptance tests for the implementation

1. **Split isolation and feasibility.** Use multiple rows per group. Check train/holdout group disjointness and exactly-once holdout coverage of rows and groups in each complete outer or inner plan, including final OOF. Check disjoint groups in any early-stopping split. Use a fit spy to ensure outer test groups and the corresponding inner holdout groups never enter the relevant models or preprocessing. On a fixed outer split, changing only a test target may change its score and must leave the prediction unchanged. Verify that missing group labels and infeasible inner folds, such as 5-by-5 nested CV with five total groups, produce a clear error before fitting.
2. **Alignment and fresh state.** Test without source IDs on a DataFrame with a nonconsecutive or duplicate index. Test with optional supplied IDs separately. Verify local-to-global position mapping and one OOF value per training row and base column. Make duplicate or missing OOF writes fail. Confirm meta target alignment, inference column order, and fresh estimator and preprocessing instances for each fit.
3. **Paired predictions.** Check that every candidate covers the same rows once per repeat. On a fixture with base predictions 10, 14, and 18 for one row, `equal_mean` must be 14. Verify that the stack and baseline calculations consume the same base matrix.
4. **Scoring.** Use known residuals and uneven fold sizes to distinguish row-level metrics from an unweighted mean of fold metrics. Recompute metrics and paired deltas from the exported rows. Check that positive deltas favor the stack and that `sample_sd` is unavailable with one repeat.
5. **Reproducible split plan.** With the same input version, ordering, and seed, reproduce split membership. If estimator predictions are sensitive to nondeterministic operations or the execution environment, distinguish that from split reproducibility.
6. **Final fit and reload.** Verify that the final call builds its own inner OOF matrix, refits all base models on the full input, and saves the fitted stack. Compare predictions before and after reload with a suitable numeric tolerance and with inputs that contain features only.
7. **MAPE edge cases.** With `y_true = [100, -50, 0, 0.01]`, `y_pred = [110, -40, 3, 1]`, and threshold `0.01`, assert `mape_pct = 15`, two eligible rows and two excluded rows. Verify that MAE and RMSE use all four rows, each candidate uses the same eligibility mask, and an empty mask produces a null MAPE and null delta.
8. **Search isolation.** Verify group disjointness in each search split and that winner refits use exactly the allowed rows. On a fixed split with fixed seeds, changing an inner OOF holdout target cannot change the winning parameters or base OOF predictions for that fit. Changing an outer test target cannot change any search, refit, or prediction for that outer fold. An infeasible search plan must fail before fitting.
