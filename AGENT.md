# GGRD-PO — Agent Guide


## Project Overview
This project aiming to first predict the property of a wafer by its process history and second use the predictive power to search for the best process sequence that leads to a specified taget value of that property.

The models are evaluated reliably by repeated K-fold cross-validation. And an esemble of many trained models are utlized to improve prediction power. The raw data are process history of wafers and their measured property.


## Repository Layout
`doc/future_model_candidates.md` contains a list of candidate models to implement in the future. Ignore `doc/scratch.md`.
| Path | Contents or Purpose |
|------|----------|
| `src/main.py` | End-to-end pipeline orchestration |
| `src/main.ipynb` | for data exploration |
| `src/config/` | holds the primary pipeline configuration file `config.yaml` |
| `src/dataset/` | for bundling data into datasaet for training|
| `src/inference/` | makes prediction by trained models |
| `src/model/` | model and cost definition |
| `src/feature/` | model and cost definition |
| `src/raw/` | raw data storage |
| `src/test/` | test suite (`test_*.py`) |
| `src/trainer/` | trains models |
| `src/utility/` | general utility functions supporting basic operation |
| `src/visual/` | fucntions to plot and save figures |
| `src/optimize/` | for process target definition and process optimization functions |
| `src/analysis/` | analyzes feature importance or reliability of optimized processes |
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