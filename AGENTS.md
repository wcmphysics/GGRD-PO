# GGRD-PO — Agent Guide


## Project Overview
This project aiming to first predict the property of a wafer by its operation history and second use the predictive power to search for the best process sequence that leads to a specified taget value of that property.

The main model is:
- an stacking model (meta model) composed of several trained base models for better prediction (all base models have uniform interface)
- evaluated reliably by repeated K-fold cross-validation
- with constrained weight (like positive, sum to 1, no bias for ridge regression) to constrain its extrapolation 

The raw data:
- are operation history of wafers, namely an ordered sequence of deposition, treatment, and measurement of wafer property
- have operation history of varying number of operations (up to `N_step` steps), each with distinct or shared features
- are stored as long table (see `doc/data_example.md` for raw data examples) and may need transformation to wide table for training 
- the transition between two operations is characterized by vacuum break: whether the wafer is exposed to atmohphere between operations
- there is only one type of measurement (default is the stress of the wafer) and it is always carried out at the last step 

Process search:
- target is a scalar value composed of various weighted components (mean-squared error, process cost, physics-violation penalty, etc.)
- two search modes in process search: fixed mode (sequence structure is fixed to certain extent) and free mode (no constaint on sequence structure)
- every candidate sequence is represented as up to `N_step_search` slots (each slot having is_active or operation ∈ {none, deposition, treatment} plus its parameters and vacuum_break).
- fixed mode: simply pins selected slots/fields in config (e.g., fix `N_step_search` = 2, step1.operation = deposition, step1.material = TiN) and optimizes only the remaining unpinned continuous/categorical variables.
- free mode: leave is_active / operation / material / recipe unpinned across all `N_step_search` steps.


## Repository Layout
`doc/future_model_candidates.md` contains a list of candidate models to implement in the future. Ignore `doc/scratch.md`.
| Path | Contents or Purpose |
|------|----------|
| `src/main.py` | End-to-end pipeline orchestration |
| `src/config/` | holds the primary pipeline configuration file `config.yaml` |
| `src/dataset/` | for bundling data into datasaet for training|
| `src/inference/` | makes prediction by trained models |
| `src/model/` | model and cost definition |
| `src/feature/` | feature engineering |
| `src/test/` | test suite (`test_*.py`) |
| `src/trainer/` | trains models |
| `src/utility/` | general utility functions supporting basic operation |
| `src/visual/` | fucntions to plot and save figures |
| `src/optimize/` | for process target definition and process optimization functions |
| `src/analysis/` | analyzes feature importance or reliability of optimized processes |
| `data/` | raw data storage |
| `doc/` | documentation |
| `output/` | stores figures, predictions, and logs |


## Preferred Tools:
Other tools are allowed if they fit tasks better.
- Python 3
- PyTorch (CPU-only)
- scikit-learn
- SciPy
- Seaborn/Matplotlib
- PyYAML
- Ax/BoTorch


## Pipeline Stages
General stages are as follows:
- Load config
- Pseudo-data generation or data reading
- Data exploration
- Data preprocessing
- Feature engineering
- Define models
- Train-validation-test split (K-fold cross-validation)
- Data bundling for training
- Training models and hyper-parameter search
- Performance metrics
- Prediction
- Ensemble of trained models
- Process optimization for the target


## Agent Instructions

### 1. Environment
- Run Python only through `.\.venv\Scripts\python.exe`, from the repository root.
- Run `main.py` in headless mode to suppress plotting window during test, otherwise the program never ends 

### 2. Workflow
1. **Clarify:** if anything is ambiguous, ask before acting. Don't hesitate to grill users on decisions.
2. **Propose:** before changing existing code, state what will change and why. Wait for approval.
3. **Implement:** once approved, implement, review, and commit. If new decision arises during implementation, ask users for their decision.
4. **Review loop:** for new or significant code, launch reviewer subagents. They critique the code, probe edge cases, check error handling, and run tests. Fix what it finds. Repeat up to 3 iterations.
5. **Commit & push:**
   - Use Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`).
   - Split any change over 50 lines into readable, logically grouped commits.

### 3. Code Style
- Follow PEP 8 but with one exception: one line ≤ 120 characters.
- Follow the idea of DRY and clean code.
- Follow the idea of minimal diff: change only the lines the request needs. Never rewrite whole files or functions without permission. Use targeted search-and-replace edits, not full-file overwrites.
- Every new function gets type hints and a concise docstring.
- For a function with coding line > 30, add 1~2 lines of comment on code block to describe the purpose of the block. 
- Cyclomatic complexity ≤ 10 per function. Above that, split into single-responsibility helpers.
- No need for backward compatibility or alias to legacy keys/functions. Tests should be modified to adapt new function whiling keep the original behavior test intact.
- Use full, explicit names like `learning_rate` (not `lr`).

### 4. Modularity
- New features go in dedicated modules directory. Expose them as functions or classes that `main.py` can import.
- Keep coupling between modules low and avoid cyclic import. Prefix internal helpers with `_`.
- A public function with ≥ 6 parameters takes a config `dict` instead. `_`-prefixed helpers are exempt.

### 5. Configuration
- Single source of truth must be strictly followed.
- Default values live in config files and never be defined in codes.
- Config values are verified by Pydantic when loaded. Missing keys raise exception.
- Don't re-cast schema-typed values (`int()`, `float()`, `bool()`, `str()`) or repeat static checks (positive, odd, …) in code. Runtime checks are only for what the schema can't know.

### 6. Errors & Unexpected Cases
- Error messages must say what failed, which value caused it, and what was expected.
- Never fail or fall back in unexpected cases silently.
- Never quietly work around an unexpected case, fall back to default, or guess at a fix. Ask the user first.