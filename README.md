# chessGPT

Small decoder-only transformer that learns to play chess from game records. Each game is encoded as a sequence of UCI move tokens, prefixed with Elo-bucket tokens for both players.

## Model

- Pre-norm (RMSNorm) transformer blocks with RoPE, KV cache and top-p sampling, written from scratch in PyTorch.
- Vocabulary: special tokens (`<pad>`, `<bos>`, `<eos>`, per-color Elo buckets 1600–2400) plus every from→to square pair reachable by a queen or knight on an empty board, plus promotions.
- Configs in `configs/`: `debug-3m` (~3.7M params), `10m` (~9.6M), `30m` (~29.4M).

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

Prepare a dataset from a zstd-compressed PGN dump (e.g. a Lichess monthly database). Games are filtered by Elo (≥1600), length (10–150 plies), time control (≥3 min) and termination type:

```sh
uv run chessgpt-prepare lichess_db_standard_rated_2013-01.pgn.zst src/chessgpt/data/processed/2013-01
```

Train from an experiment config (run from the repo root; paths in the config are relative to it). Checkpoints, logs and TensorBoard events go to `output_dir`:

```sh
uv run chessgpt-train configs/debug-3m.yaml
```

Measure move legality of a checkpoint — share of legal generated moves, share of games without illegal moves, mean length before the first failure. Reports are saved next to the checkpoint in `reports/`:

```sh
uv run chessgpt-eval runs/debug-3m/best.pt --games 500 --temperatures 0.7 1.0
```

## Layout

```
src/chessgpt/
  data/        PGN streaming, filtering, encoding, binary dataset writer
  model/       tokenizer, attention, RoPE, blocks, transformer, sampling
  train/       training loop, optimizer, checkpoints
  evals/       move legality evaluation and reports
  experiment.py  YAML config → training run
```
