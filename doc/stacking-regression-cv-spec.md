# 回歸 Stacking 實作規格

第 2.2 版 · 繁體中文

本規格以 Python 實作約 100 筆有標籤、必須依指定類別分組切分的回歸 stacking。內層交叉驗證產生 base models 的折外預測（out-of-fold，OOF），用來訓練 meta-model。重複外層交叉驗證在留出的群組上比較 stacking 與基準。評估後，再用全資料訓練一套正式部署的 stack。

## 1. 輸入、預設值與切分原則

| 輸入或設定 | 規格 |
|---|---|
| `X`、`y` | 每筆樣本有一列特徵及一個有限數值目標，兩者的列順序一致。 |
| `groups` 或 `group_col` | 必填的分組資訊。傳入與 `X` 對齊的一維 `groups`，或用 `group_col` 指定來源 DataFrame 的分組欄。相同值屬於同一群組。群組標籤可以重複，且每列都須有值。預設從模型特徵排除分組欄。 |
| `row_id` | 選填的來源識別碼，供匯出及回查原始資料。若有提供，必須唯一、非空且與 `X` 對齊。省略時，用原始輸入的列位置當作輸出中的 `row_id`。模型特徵排除稽核用 ID 與目標欄。 |
| `base_models` | 至少設定兩個有唯一名稱的 base regressors。必要時指定其特徵欄位、前處理及選用的搜尋設定。清單順序固定，每個模型都要能預測所有需要預測的資料列。 |
| `meta_model` | 傳入回歸模型設定，以及需要的前處理。輸入是按照 base model 清單排列的預測欄位。若尚未選定 meta-model，可先用設定固定的 Ridge 回歸。 |
| `outer_folds`、`inner_folds`、`outer_repeats` | 預設依序為 `5`、`5`、`5`，都可調整。Fold 數至少為 2，重複次數至少為 1。 |
| `root_seed` | 由呼叫者指定或提供有紀錄的預設值。由它產生各輪外層、各次內層及最終內層切分所用的種子。模型若支援隨機種子，預設為每個模型設定指定一個可重現的種子，並在各次 fit 維持固定。保存實際使用的種子。 |
| 指標 | MAE 為主要指標，RMSE 和 MAPE 為輔助指標。MAE、RMSE 使用目標的原始單位。MAPE 對符合條件的列計算並以百分比呈現。 |
| `mape_min_abs_y` | 以目標原始單位表示的有限非負門檻，預設 `0.0`。只有 `abs(y_true) > mape_min_abs_y` 的列會納入 MAPE。評估前先固定門檻。 |
| Base model 超參數搜尋 | 預設關閉，可逐一為 base model 啟用。事先指定搜尋空間與預算。`search_folds` 至少為 2，可行時可先試 3 折。見 §4.1。 |
| Python 套件 | 使用 Python、NumPy、pandas（DataFrame 輸入與輸出表格）以及 scikit-learn `>=1.6`（支援打散的 `GroupKFold`）。記錄實際套件版本。 |

從 DataFrame 讀取時，在同一批列上取出特徵、目標、群組和選填的 ID。若分開傳入陣列，驗證各陣列的維度與長度，且呼叫者須維持同一列順序。長度相同無法檢出某個陣列是否曾被重排。群組標籤有缺值時，在切分前直接報錯。若要編碼，為每種群組標籤分配唯一整數，並保存對照表供回查。

### 1.1. 列位置、來源 ID 與群組標籤

輸入資料已經有列順序。在載入資料時用 `row_pos = np.arange(len(X))` 建立一次內部列位置，再把這些原始位置帶進每個子集。來源 `row_id` 只在匯出後回查、跨資料版本對照或資料重排時有額外用途。群組標籤告訴切分器哪些列必須一起分到同一側。兩列即使屬於同一個群組，或特徵值完全相同，仍各自有一個列位置。切分器回傳整數位置，pandas 要用 `.iloc`，NumPy 則用整數陣列取列。若 `row_id` 是用原始位置產生，應保存輸入版本與列順序，因為這些 ID 只對該順序有效。

### 1.2. 群組切分

每輪外層 CV，以及每次在外層訓練子集上產生內層 OOF，都使用 `GroupKFold(n_splits=..., shuffle=True, random_state=seed)`。最終在全資料產生內層 OOF 時也使用同一種切分器。每個 split 的訓練端與留出端，其群組集合交集必須為空。每輪完整切分讓每個群組及其中所有資料列恰好留出一次。所有候選共用同一份外層切分，同一次 stack 訓練中的 base models 共用內層切分。同一個群組可以在其中一折當留出資料，在其他折當訓練資料。各輪都獨立遵守這個規則。

`GroupShuffleSplit` 也能保證單次切分不拆開群組。多次抽樣的留出群組可能重疊，也可能有些群組從未被留出。外層整輪評分與內層 OOF 都需要恰好一次的覆蓋，因此這兩處採用 `GroupKFold`。如果模型需要 early stopping，可以在該次允許使用的訓練資料內，用一次 `GroupShuffleSplit` 留出驗證群組。 [GroupKFold 文件](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) · [GroupShuffleSplit 文件](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)

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

這個函式傳回的是相對於所傳入 `X` 的列位置。例如 `X` 是 pandas DataFrame、`groups` 是 NumPy 陣列時，一輪外層呼叫 `build_group_plan(X, groups, n_splits=outer_folds, seed=outer_seed)`。對每個回傳的 `outer_train_idx`，內層呼叫 `build_group_plan(X.iloc[outer_train_idx], groups[outer_train_idx], n_splits=inner_folds, seed=inner_seed)`。最終內層用全資料、`n_splits=inner_folds` 和另一個種子。外層每輪用不同種子，但群組數少時也可能產生相同的外層切分。保存成員資訊，並回報忽略 fold 順序後有幾種不同的外層切分。如果外層切分相同、內層切分不同，分數仍可能有差異。

訓練前先檢查所有外層、內層及最終切分，包括已設定的搜尋切分（§4.1）或 early-stopping 切分。全資料的不同群組數至少要有 `outer_folds`，**每個實際切出的外層訓練集**中也至少要有 `inner_folds` 組。全資料亦須滿足最終內層的折數要求。例如若總共只有 5 個群組，外層可做 5 folds，但每個外層訓練集只剩 4 組，無法再做內層 5 folds。檢查各折的實際列數與群組數，包含模型最小訓練樣本數。若無法成立，回報輪次、fold 與實際數量，明確調整折數後再跑，維持群組隔離。群組大小不均時，各折列數也可能不同。

缺失的特徵值由模型的前處理處理，補值參數只能用當次允許的訓練資料估計。若模型無法對該預測的列產生有限數值，回報原因並中止該次執行。

## 2. 外層要比較的預測方式

所有候選都預測**同一批外層留出的樣本**。三種方式共用在完整外層訓練集上重訓的 base models。若模型訓練時轉換過目標值，輸出預測前須轉回原始單位。

| 候選 ID | 對單筆留出樣本的預測 | 要回答的問題 |
|---|---|---|
| 每個 base model 各有一個 `base:<name>` | 取該 base model 在完整外層訓練集上 fit 後的預測。 | 組合多個模型是否優於個別模型？ |
| `equal_mean` | 把該筆樣本的所有 base 預測取算術平均，每個模型的權重是 `1 / base 模型數`。 | 讓模型學習組合方式，是否優於固定的等權平均？ |
| `stack` | 按固定順序把同一組 base 預測交給 meta-model。 | 用內層 OOF 學到的組合方式，在外層未見資料上是否有效？ |

所有候選共用在完整外層訓練集 fit 的同一批 base pipelines。若 base model 有調參，就共用搜尋勝出後重訓的模型。這樣可直接比較預測組合方式的差異。等權平均使用固定權重。若任一候選缺少應有的預測，該次執行無法完成可比較的評估，錯誤訊息須指出輪次、fold 與模型名稱。

## 3. 訓練一套 stack

外層評估與最終訓練共用 `fit_stack(train_data, config, fit_context)`。`train_data` 包含已對齊的 `X`、`y`、`groups`、`row_pos` 與匯出用 ID。`fit_context` 包含本次 fit 識別、種子及已按該列順序驗證的內層切分。用 `sklearn.base.clone` 建立全新的 scikit-learn 模型及前處理 pipeline，其他模型可用 factory 建立實例。

1. **使用內層群組切分。** 每折把 `train_data` 切成群組互斥的 `inner_train` 和 `inner_holdout`。依訓練資料列順序及固定模型欄位，配置形狀為 `[n_train, n_base]` 的 OOF 矩陣 `Z_oof`，並追蹤每格寫入次數。
2. **產生 base OOF。** 用 `inner_train` fit 全新的 base pipeline。預測 `inner_holdout.X`，依留出資料的局部列位置和模型欄位寫回預測。每筆 OOF 預測都來自排除了該筆所屬**整個群組**的 base fit，讓 meta-model 從 base models 對未見群組的預測學習。
3. **驗證矩陣並訓練 meta-model。** 檢查每格恰好寫入一次，所有預測都是有限數值，且列順序與 `train_data.y` 一致。以這兩份資料 fit 全新的 meta pipeline，包括所需的 meta 特徵標準化。base OOF 分數可供診斷。在同一份矩陣上評估剛 fit 的 meta-model，得到的是訓練誤差。
4. **用完整 `train_data` 重訓 base models。** 建立全新的 base model 與前處理實例，按已設定的訓練方式各 fit 一次。沿用步驟 3 的 meta-model。這讓推論時的 base model 使用更多合法的訓練資料。如果回頭用重訓後的 base models 預測 `train_data` 來訓練 meta-model，base model 對某筆資料的 fit 就會影響該筆的 meta 輸入，破壞 OOF 的隔離。
5. **回傳已訓練的 stack。** 包含依序排列的 base pipelines、meta pipeline、輸入特徵規格和 base 預測欄位規格。診斷資料如有輸出，可另外回傳。

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

外層索引相對於全資料，內層索引是當次訓練子集的局部列位置。例如用 `outer_train_idx[inner_holdout_idx]` 可將內層留出位置轉成全資料位置。可以直接用內層索引取已切好的子集，若要取全資料，須先轉換。如果把局部位置直接拿來取全資料，可能誤將外層測試樣本放進訓練。寫回 OOF 時保留預測批次的原列順序，報表另外保存識別碼。


若分組後各折筆數恰好接近，100 筆資料、外層 5 folds 時，每折約有 80 筆訓練、20 筆測試。內層 5 folds 也均衡時，每折約用 64 筆訓練、預測 16 筆。OOF 矩陣以各折的實際訓練筆數配置。內層 base fit 只用訓練資料的子集，重訓則用當次完整訓練集。外層 CV 會評估這個設計，最終重訓也沿用它。

## 4. 訓練資料界線與預測介面

所有會從資料估計參數的步驟，都使用當次 fit 可用的訓練資料。這包括補值、標準化、特徵選擇、編碼、降維與目標轉換。對留出資料使用已 fit 的前處理，再產生預測。特徵必須能在實際推論時取得。若特徵會用到其他樣本的標籤，必須另外定義如何在相應的訓練切分內產生該特徵，避免把留出資料的資訊帶入模型。

評估前，先固定 meta-model 的設定，以及各 base model 的設定或調參規則。當次預測產生後，內層留出標籤只供診斷評分。若只做 early stopping 而沒有啟用搜尋，在當次允許的訓練資料內按群組切分，例如用 `GroupShuffleSplit(n_splits=1, test_size=..., random_state=seed)` 並傳入 `.split(..., groups=...)`。`test_size` 指群組數或比例。檢查兩邊群組互斥、列數足夠。試訓的前處理只在訓練子集 fit，由驗證子集決定迭代次數，再固定次數，使用當次允許的全部訓練資料重新 fit 前處理和模型。對 OOF fit，可用範圍是 `inner_train`。對完整 base refit，可用範圍是 `train_data`。若模型只能在內部逐列切驗證集，改用預先指定的迭代次數。

若日後要用外層留出資料衡量包含 meta-model 或架構選擇的流程，選擇程序須完全放在外層訓練集內。每個分組選模 split 都要重建 stack。直接切開先算好的 OOF 矩陣，可能讓選模訓練列的 base 預測受選模留出資料影響。

評估與部署共用預測邏輯。`predict_base_matrix(fitted_stack, X_new)` 驗證輸入欄位，並為每筆資料及每個 base model 傳回一個有限數值，欄位順序與訓練一致。`predict_meta(fitted_stack, Z_new)` 檢查 base 欄位順序，再為每列傳回一個有限數值。`predict_stack(fitted_stack, X_new)` 依序呼叫兩者。外層評估每折只計算一次 base 預測矩陣，供所有候選共用。部署時傳入無標籤的特徵，並按輸入的列順序回傳預測。

### 4.1. 選用的 base model 分組超參數搜尋

對啟用調參的 base model，把搜尋接到既有的 `fit_fresh_base`。外層 CV 衡量整套 stack，內層 CV 產生 base OOF 預測。每次 base fit 另外在自己允許的訓練範圍內，以分組 CV 比較超參數。

| Base fit 的用途 | 搜尋及重訓可用的資料 | 勝出模型接著預測 |
|---|---|---|
| 在一個外層 fold 內產生 OOF | 該 OOF split 的 `inner_train` | 該 split 的 `inner_holdout` |
| 重訓以便評估一個外層 fold | 該 fold 的全部 `outer_train` | `outer_test` |
| 為最終 meta-model 產生 OOF | 該最終 OOF split 的 `inner_train` | 該最終 split 的 `inner_holdout` |
| 重訓以供部署 | 全部有標籤資料 | 未來資料 |

例如某次內層 OOF 要預測群組 G。若先用完整 `outer_train` 找一組參數，G 的真值就可能影響參數選擇。後續的 base fit 即使排除 G，也會沿用受 G 影響的參數。這時外層測試仍隔離，但內層 OOF 已失去隔離。應在每個 `inner_train` 內各自搜尋，完整重訓時再用該次較大的合法訓練集搜尋。不同 fit 可以選到不同參數，外層 CV 衡量的是「搜尋後再 fit」的流程，部署時也遵循同一流程。

1. 在每次允許的訓練子集上建立 `GroupKFold` 搜尋切分。訓練前檢查所有啟用搜尋的 fit，包括最終重訓。每次可用的相異群組數至少要有 `search_folds` 組，各折實際筆數也要符合模型需求。若無法成立，回報是哪一次 fit 與實際數量，明確調整設定後再執行。
2. 將整個 pipeline 放入搜尋，讓各個參數候選只用當次搜尋訓練折 fit 前處理。若有目標轉換，封裝後的 `predict` 須以原始目標單位輸出。小型網格可用 `GridSearchCV`，也可以用固定預算 `n_iter` 與種子的 `RandomizedSearchCV`。固定候選順序，分數完全相同時選第一個。
3. 用 `neg_mean_absolute_error` 排名，以 `refit=True` 取得 `best_estimator_`。這裡使用**各搜尋折 MAE 的等權平均**。當各折筆數不同時，結果可能與外層報表的逐筆等權 MAE 不同。勝出模型會用這次允許的全部列重訓。搜尋分數供診斷，整套流程的表現以外層測試分數衡量。
4. 基本搜尋流程關閉 early stopping 並使用固定迭代次數。也可把迭代次數本身列入搜尋。若日後同時使用兩者，各候選須在搜尋訓練折內另切群組互斥的 early-stopping 驗證子集，勝出模型重訓時也須在當次允許的資料內執行同樣程序。搜尋驗證折只用來替候選打分。

以下範例先建立群組互斥的局部索引再傳給 `cv`。`search.fit` 用相同子集與列順序。群組資訊已反映在這些切分中。 [GridSearchCV 文件](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) · [RandomizedSearchCV 文件](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

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

把 fit 完成的勝出模型交回既有預測流程，並保存搜尋切分、種子、選出的參數與搜尋 MAE。對同一個外層 fold，個別 base、等權平均及 stack 共用完整外層訓練集上勝出模型的預測。

## 5. 執行重複外層交叉驗證

本文的**一輪**是跑完所有 outer folds。每輪的每一折都用外層訓練集建立一套新的 stack。外層測試特徵只從預測介面進入，完成預測後才對齊外層真實值以進行評分。

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

依預設設定，評估要 fit 25 套 stack，最後再 fit 一套正式 stack。若模型使用固定迭代次數，且沒有額外調參或 early stopping，每套 stack 內的每個 base model 會為內層 OOF fit 5 次、完整重訓 1 次。因此每個 base model 總共 fit 156 次，meta-model 總共 fit 26 次。基準比較共用這些 fit 的結果。

啟用搜尋的 base model 會在上述 156 個 fit 情境中，各評估 `H` 組參數、跑 `T = search_folds` 折，再重訓勝出者。若沒有額外的 early stopping，總 fit 次數為 `156 × (H × T + 1)`。例如 5 組參數、3 折搜尋，一個 base model 就要 fit 2,496 次。可先針對需要調參的模型做小範圍搜尋。

## 6. 計算分數與配對差值

每輪對每個候選按原始列順序對齊預測與真實值，或依 `row_id` 合併已匯出的紀錄。候選的外層測試 ID 集合須相同，每個 ID 恰好出現一次。令 `e_i = y_pred_i - y_true_i`，其中 `i` 涵蓋該輪全部留出樣本。

- `MAE = mean(abs(e_i))`，即逐筆絕對誤差的平均。
- `RMSE = sqrt(mean(e_i^2))`，即逐筆平方誤差先平均，再開根號。
- `MAPE (%) = 100 * mean(abs(e_i) / abs(y_true_i))`，只納入 `abs(y_true_i) > mape_min_abs_y` 的列。

每輪依真值建立一次 MAPE 遮罩，所有候選使用同一份遮罩。預設只排除真值恰好為零的列，因為除以零的百分比誤差沒有定義。非零但很接近零的真值也可能主導 MAPE。如果目標在業務上有具意義的最小量級，請在評估前用原始單位指定門檻並記錄，因為這會改變 MAPE 所描述的資料範圍。若沒有符合條件的列，記為 `null`（不適用）。匯出納入與排除筆數。MAE、RMSE 仍使用全部測試列。先套用遮罩再直接計算百分比。scikit-learn 的 MAPE 函式回傳比例值，並會對真值為零的列使用極小的分母。 [scikit-learn MAPE 文件](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html)

先合併一輪中各折的預測，再用整輪平方誤差計算 RMSE。直接平均各折的 RMSE 會提早開根號，結果可能不同。預設每筆樣本等權，所以資料較多的群組對指標影響較大。若希望每個群組等權，可額外計算各組的 MAE，再把各組 MAE 取平均。跨輪彙整時，對同一指標的**各輪分數**計算算術平均與樣本標準差。重複次數為 1 時，標準差記為不適用。若先平均同一樣本在不同輪的預測再評分，衡量的會是另一個預測器。

對每個指標，在每輪內比較 `stack` 與 `equal_mean`，也逐一比較 `stack` 與各 `base:<name>`。整個實驗中，各個 base 的名稱固定對應同一個候選。若每折都用測試分數挑出最好的 base，就等於用測試資料的真值決定哪個模型負責預測。

```text
delta = metric(baseline) - metric(stack)
```

指標越低越好，因此差值為正代表 stacking 的誤差較小。保存每輪差值、跨輪差值平均和樣本標準差，以及差值為正的輪數與總輪數。例如等權平均 MAE 為 4.0、stacking MAE 為 3.6，差值就是 +0.4 個目標單位。由於兩者面對相同的測試樣本，這項配對比較能減少測試資料不同帶來的干擾。如果差值很小，而且各輪有正有負，就只有有限的證據顯示 stacking 改善了預測。輪與輪之間共用同一批樣本，因此分數波動與勝出輪數只描述對切分的敏感度，無法當作來自獨立資料集的結果。

MAPE 的差值以百分點表示。某輪的 MAPE 不適用時，該輪的 MAPE 差值也不適用。彙整時只使用有效輪次，並回報有效輪數和總輪數。沒有有效輪次時，平均與標準差都記為 `null`。只有一輪有效時，只有標準差記為 `null`。

這項比較用於評估報告，預設訓練流程仍產出一套正式 stack。若之後根據這份報告決定部署哪個候選，勝出者的外層分數已參與選擇，可能偏樂觀。若要評估自動選模規則的表現，先定義規則，然後在每次外層訓練集內完成選擇，再用該折的外層測試資料打分。也可以用新收集的獨立資料評估已選出的模型。

## 7. 用全資料重訓並打包部署

評估完成後，用所有有標籤資料與新建立的最終內層分組切分，再呼叫一次相同的 `fit_stack`。每次 OOF fit 預測整個留出 fold 的群組。將預測拼成涵蓋全資料的**新 OOF 矩陣**，訓練一個**新的 meta-model**，再各用所有有標籤資料 fit 每個最終 base model。外層評估用的模型在記錄預測後即可釋放。部署只需傳入 `X_new` 中宣告的模型特徵。群組標籤與稽核 ID 用於切分和報表。

保存已 fit 的 base 與 meta pipelines、模型順序、特徵規格、模型設定和相容性資訊。以無標籤特徵測試載入與預測。外層 CV 分數估計的是在每折實際可用的群組數與列數下訓練，再對未見群組預測的表現。最終模型用所有群組訓練，這個流程沒有提供該模型自己的獨立測試分數。

## 8. 必要輸出

儲存格式可由實作者選擇。使用以下欄位，或記錄等效的資料規格。

| 輸出 | 必要內容 |
|---|---|
| 執行與切分紀錄 | 資料、設定與套件版本、模型名稱與順序、特徵規格、實際種子、群組編碼表（若有編碼）、ID 來自來源或列位置的規則。保存外層、內層、最終與選用的 early-stopping 切分之 `row_id`、`group_id` 成員和兩端實際列數、群組數，以及互不相同的外層切分數。各次內層切分須標記所屬外層輪次與 fold，或標記屬於最終重訓。 |
| 外層逐筆預測 | `repeat_id`、`outer_fold_id`、`row_id`、`group_id`、`candidate_id`、`y_true`、`y_pred`。唯一鍵為 `(repeat_id, row_id, candidate_id)`。 |
| 每輪指標 | `repeat_id`、`candidate_id`、`n_scored`、`mae`、`rmse`、`mape_pct`、`n_mape_eligible`、`n_mape_excluded`。在執行設定中保存 `mape_min_abs_y`。 |
| 指標彙整 | `candidate_id`、`metric`、`n_repeats`、`n_valid_repeats`、`mean`、`sample_sd`。 |
| 配對比較 | `repeat_id`、`baseline_id`、`metric`、`delta`，以及分組彙整的 `mean_delta`、`sample_sd_delta`、`n_positive`、`n_valid_repeats`、`n_repeats`。 |
| 搜尋紀錄（啟用時） | Base model 名稱及 fit 情境（外層輪次、外層 fold、內層 fold 或重訓，或最終訓練）、搜尋種子、分組切分成員、嘗試過的候選、勝出參數與各搜尋折 MAE 的平均。在部署模型資訊中保存最終勝出的參數。 |
| 部署套件 | 最終 fit 的 stack、模型資訊與可呼叫的預測介面。 |

## 9. 程式驗收測試

1. **群組隔離與可行性。** 測試資料中的每組要有多筆樣本。驗證每折訓練端與留出端的群組互斥，且各輪外層、各次內層和最終 OOF 切分都讓每筆樣本與每個群組恰好被留出一次。early-stopping 切分也須群組互斥。可用測試模型記錄實際 fit 的資料，確認對應的外層或內層留出群組未進入模型或前處理。在固定外層切分下，改動外層測試標籤可以改變分數，但預測必須相同。另測缺失群組標籤與不可行的折數，例如只有五組卻要求外層和內層都五折，應在 fit 前清楚報錯。
2. **列欄對齊與全新實例。** 用索引不連續或有重複索引的 DataFrame 測試未提供來源 ID 的情況，另測選填的來源 ID。驗證局部位置轉回全資料位置的對應，以及每筆訓練資料的各個 base 欄位都有一個 OOF 值。重複或漏寫會報錯。確認 meta 標籤按列對齊、推論欄順序固定、每次 fit 都使用新的模型及前處理實例。
3. **候選配對。** 驗證每輪所有候選都對同一組樣本各預測一次。如果某筆資料的 base 預測為 10、14、18，`equal_mean` 必須是 14。確認 stacking 和基準都源於同一份 base 預測矩陣。
4. **公式與彙整。** 用已知誤差且 fold 大小不等的測試資料，區分逐筆計算與對各 fold 指標取等權平均。用匯出的逐筆資料重算指標與配對差值。確認正差值代表 stacking 較好，只有一輪時 `sample_sd` 為不適用。
5. **重現切分。** 相同資料版本、列順序與種子可得到相同切分成員。若模型預測受非決定性運算或執行環境影響，須與切分可重現性分開檢查。
6. **最終重訓與載入。** 檢查最終訓練會重建自己的內層 OOF 矩陣，用全資料重訓每個 base model，並保存已 fit 的 stack。使用只有特徵的資料，在適當的數值容許誤差內比較保存前與重新載入後的預測。
7. **MAPE 邊界情況。** 令 `y_true = [100, -50, 0, 0.01]`、`y_pred = [110, -40, 3, 1]`，門檻為 `0.01`。驗證 `mape_pct = 15`，納入與排除各兩列。驗證 MAE 和 RMSE 使用全部四列，各候選共用 MAPE 遮罩，且沒有符合條件的列時，MAPE 與其配對差值皆為 `null`。
8. **搜尋隔離。** 驗證搜尋各折的群組互斥，勝出模型只使用當次允許的列重訓。固定切分與種子後，改動內層 OOF 留出標籤，不能影響該次搜尋的最佳參數或 base OOF 預測。改動某個外層測試標籤，不能影響該外層折的搜尋、重訓或預測。搜尋折數不可行時，須在訓練前報錯。
