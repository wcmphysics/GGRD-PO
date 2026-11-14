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
