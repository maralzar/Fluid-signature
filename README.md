# Reasoning Signature MVP

Learn transferable **Reasoning Signatures** on FLUID supernodes by distilling instrumented OWL 2 RL reasoning traces into a **Reasoning Behavior Tensor (RBT)**, then training a lightweight GNN student on LUBM1/LUBM2 with cross-graph transfer evaluation.

## Pipeline

```text
Ontology + KG -> FLUID summarization + instrumented OWL2 RL teacher trace
teacher events -> RBT per supernode
RBT + summary graph -> GNN student -> Reasoning Signatures -> transfer eval
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run end-to-end

```bash
export PYTHONPATH=.
python scripts/01_prepare_data.py
python scripts/02_run_fluid.py --graph lubm1
python scripts/02_run_fluid.py --graph lubm2
python scripts/03_build_rbt.py --graph lubm1
python scripts/03_build_rbt.py --graph lubm2
python scripts/04_train_student.py
python scripts/05_eval_transfer.py
```

Or:

```bash
bash scripts/run_pipeline.sh
```

## Outputs

- `data/processed/*.summary_graph.pt` — FLUID summary graphs
- `data/processed/*.rbt.pt` — Reasoning Behavior Tensors `[supernodes, 8, 5, 6]`
- `data/processed/*.reasoning_signatures.pt` — neural semantic/topological signatures and decoded RBT predictions
- `checkpoints/lubm1_student.pt` — trained student model
- `results/mvp_transfer.json` — transfer evaluation report with reconstruction MSE, semantic/topology alignment, random and materialization baselines

The default config trains on `lubm1` at scale 3 and evaluates transfer to `lubm2` at scale 5. `mvp_passed` is a strict diagnostic for matched semantic-signature alignment against random pairs; the report also includes direct source/target RBT reconstruction MSE for the student inference objective.

## Tests

```bash
pytest tests/ -q
```

## Phase 2 (transfer hardening + ablations)

Uses divergent graph scales (LUBM1 scale=3, LUBM2 scale=6), stronger disentanglement losses, and ablation studies:

```bash
bash scripts/run_phase2.sh
```

Outputs additionally include `results/ablations.json` with loss-component ablations.

See [docs/design_decisions.md](docs/design_decisions.md) for research design choices.
