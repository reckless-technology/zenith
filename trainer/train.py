# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Zenith NNUE trainer (PyTorch, independent of any other engine's trainer).

Reads Zenith's own self-play records (`fen;stm_score_cp;wdl`), trains a 768->HIDDEN perspective network
with SCReLU, and exports a quantised `.nnue` file that the C engine loads directly. A float checkpoint
is saved alongside for inspection.

    python trainer/train.py --data 'data/pilot/shard_*.txt' --out nets/zenith-pilot.nnue \
        --epochs 30 --batch-size 16384 --lr 1e-3 --wdl-lambda 0.5 --device cuda

Loss: MSE( sigmoid(prediction), wdl_lambda*wdl + (1-wdl_lambda)*sigmoid(score/EVALUATION_SCALE) ), so the
network output is a logit whose centipawn value is prediction * EVALUATION_SCALE.
"""

import argparse
import glob
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn

from features import (EVALUATION_SCALE, HIDDEN_SIZE, INPUT_FEATURES, MAX_ACTIVE_FEATURES, NNUE_MAGIC,
                      NUM_OUTPUT_BUCKETS, PADDING_INDEX, QUANT_ACCUMULATOR, QUANT_OUTPUT, position_features)


# --------------------------------------------------------------------------------------------------
# Data loading / featurisation
# --------------------------------------------------------------------------------------------------
def _featurise_lines(lines):
    """Parse a chunk of `fen;score;wdl` lines into padded feature arrays. Runs in a worker process."""
    count = len(lines)
    own_indices = np.full((count, MAX_ACTIVE_FEATURES), PADDING_INDEX, dtype=np.int16)
    opponent_indices = np.full((count, MAX_ACTIVE_FEATURES), PADDING_INDEX, dtype=np.int16)
    scores = np.zeros(count, dtype=np.float32)
    results = np.zeros(count, dtype=np.float32)
    kept = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(";", 2)
        if len(parts) != 3:
            continue  # tolerate truncated last lines (e.g. datagen killed mid-write)
        fen, score_text, wdl_text = parts
        try:
            _, own, opponent = position_features(fen)
            score = float(score_text)
            result = float(wdl_text)
        except (ValueError, KeyError, IndexError):
            continue
        if len(own) > MAX_ACTIVE_FEATURES:
            continue
        own_indices[kept, : len(own)] = own
        opponent_indices[kept, : len(opponent)] = opponent
        scores[kept] = score
        results[kept] = result
        kept += 1
    return own_indices[:kept], opponent_indices[:kept], scores[:kept], results[:kept]


def load_dataset(data_globs, cache_path=None):
    """Featurise all shards into padded index arrays. One pass, one process (robust and debuggable); a
    per-shard progress line is printed. `data_globs` is one glob or a list of globs. If `cache_path` is
    given it is loaded when present and written otherwise, so repeated runs skip featurisation entirely."""
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        print(f"loaded featurised cache {cache_path}: {cached['scores'].shape[0]:,} positions")
        return cached["own"], cached["opponent"], cached["scores"], cached["results"]

    if isinstance(data_globs, str):
        data_globs = [data_globs]
    paths = sorted(path for pattern in data_globs for path in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no data files matched: {data_globs}")

    own_parts, opponent_parts, score_parts, result_parts = [], [], [], []
    total = 0
    start_time = time.time()
    for path in paths:
        with open(path) as handle:
            own, opponent, scores, results = _featurise_lines(handle.readlines())
        own_parts.append(own)
        opponent_parts.append(opponent)
        score_parts.append(scores)
        result_parts.append(results)
        total += scores.shape[0]
        print(f"  featurised {os.path.basename(path)}  ({total:,} positions, {time.time() - start_time:.0f}s)")

    own_indices = np.concatenate(own_parts)
    opponent_indices = np.concatenate(opponent_parts)
    scores = np.concatenate(score_parts)
    results = np.concatenate(result_parts)
    print(f"featurised {len(scores):,} positions in {time.time() - start_time:.0f}s")

    if cache_path:
        np.savez(cache_path, own=own_indices, opponent=opponent_indices, scores=scores, results=results)
        print(f"wrote featurised cache {cache_path}")
    return own_indices, opponent_indices, scores, results


def featurise_shard(text_path, npz_path, chunk_bytes=256 * 1024 * 1024):
    """Featurise one `fen;score;wdl` text shard into a single .npz. To keep peak RAM to ~one shard's arrays
    (~13GB) — so several shards can be featurised in parallel without OOM — we count lines first, PRE-ALLOCATE
    the output arrays, and fill them chunk-by-chunk in place (no np.concatenate, which would transiently
    double memory). Resumable: skips if the target exists; writes atomically via a .tmp."""
    if os.path.exists(npz_path):
        print(f"  skip {os.path.basename(npz_path)} (already featurised)", flush=True)
        return
    start_time = time.time()
    with open(text_path) as handle:
        line_count = sum(1 for _ in handle)  # upper bound on kept rows

    own = np.full((line_count, MAX_ACTIVE_FEATURES), PADDING_INDEX, dtype=np.int16)
    opponent = np.full((line_count, MAX_ACTIVE_FEATURES), PADDING_INDEX, dtype=np.int16)
    scores = np.zeros(line_count, dtype=np.float32)
    results = np.zeros(line_count, dtype=np.float32)

    kept = 0
    with open(text_path) as handle:
        while True:
            lines = handle.readlines(chunk_bytes)
            if not lines:
                break
            chunk_own, chunk_opponent, chunk_scores, chunk_results = _featurise_lines(lines)
            count = chunk_scores.shape[0]
            own[kept : kept + count] = chunk_own
            opponent[kept : kept + count] = chunk_opponent
            scores[kept : kept + count] = chunk_scores
            results[kept : kept + count] = chunk_results
            kept += count

    temporary = npz_path + ".tmp.npz"
    np.savez(temporary, own=own[:kept], opponent=opponent[:kept], scores=scores[:kept], results=results[:kept])
    os.replace(temporary, npz_path)
    print(f"  featurised {os.path.basename(text_path)} -> {os.path.basename(npz_path)} "
          f"({kept:,} positions, {time.time() - start_time:.0f}s)", flush=True)


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
def screlu(x):
    """Squared clipped ReLU: clamp(x, 0, 1)^2."""
    clamped = torch.clamp(x, 0.0, 1.0)
    return clamped * clamped


class PerspectiveNetwork(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        # Feature transformer as an EmbeddingBag: row i (i < 768) is the weight column for feature i, which
        # is exactly the [768, HIDDEN] on-disk layout. mode="sum" adds the (<=32) active feature columns per
        # position WITHOUT materialising a [batch, 32, HIDDEN] intermediate — essential at HIDDEN=1024 on an
        # 8GB GPU. Row 768 is a forced-zero padding slot.
        self.feature_transformer = nn.EmbeddingBag(INPUT_FEATURES + 1, hidden_size, mode="sum",
                                                   padding_idx=PADDING_INDEX)
        self.feature_transformer_bias = nn.Parameter(torch.zeros(hidden_size))
        # One output head per material bucket; each position trains only its own head (gather below), so
        # sparse phases (few pieces) get heads specialised to them instead of a compromise single head.
        self.output = nn.Linear(2 * hidden_size, NUM_OUTPUT_BUCKETS)
        nn.init.normal_(self.feature_transformer.weight, std=0.01)
        with torch.no_grad():
            self.feature_transformer.weight[PADDING_INDEX].zero_()

    def forward(self, own_indices, opponent_indices):
        own_accumulator = self.feature_transformer(own_indices) + self.feature_transformer_bias
        opponent_accumulator = self.feature_transformer(opponent_indices) + self.feature_transformer_bias
        hidden = torch.cat([screlu(own_accumulator), screlu(opponent_accumulator)], dim=1)
        heads = self.output(hidden)  # [batch, NUM_OUTPUT_BUCKETS]
        # Material bucket per sample: every piece emits exactly one own-perspective feature, so the count of
        # non-padding indices IS the piece count; bucket = (pieces - 2) // 4 (features.output_bucket).
        piece_count = (own_indices != PADDING_INDEX).sum(dim=1)
        bucket = torch.clamp((piece_count - 2) // 4, 0, NUM_OUTPUT_BUCKETS - 1)
        return heads.gather(1, bucket.unsqueeze(1)).squeeze(1)


# --------------------------------------------------------------------------------------------------
# Quantised export
# --------------------------------------------------------------------------------------------------
def export_quantised_net(model, path):
    with torch.no_grad():
        transformer = model.feature_transformer.weight[:INPUT_FEATURES].cpu().numpy()  # [6144, HIDDEN]
        transformer_bias = model.feature_transformer_bias.cpu().numpy()                # [HIDDEN]
        output_weight = model.output.weight.cpu().numpy()                              # [BUCKETS, 2*HIDDEN]
        output_bias = model.output.bias.cpu().numpy()                                  # [BUCKETS]

    transformer_q = np.rint(transformer * QUANT_ACCUMULATOR).astype(np.int16)
    transformer_bias_q = np.rint(transformer_bias * QUANT_ACCUMULATOR).astype(np.int16)
    output_weight_q = np.rint(output_weight * QUANT_OUTPUT).astype(np.int16)
    output_bias_q = np.rint(output_bias * QUANT_ACCUMULATOR * QUANT_OUTPUT).astype(np.int32)

    with open(path, "wb") as handle:
        handle.write(NNUE_MAGIC)
        handle.write(transformer_q.tobytes())        # [6144][HIDDEN] int16, feature-major
        handle.write(transformer_bias_q.tobytes())   # [HIDDEN] int16
        handle.write(output_weight_q.tobytes())      # [BUCKETS][2*HIDDEN] int16, bucket-major
        handle.write(output_bias_q.tobytes())        # [BUCKETS] int32
    saturated = int(np.sum(np.abs(transformer * QUANT_ACCUMULATOR) > 32767))
    print(f"exported {path}  (transformer weights saturating int16: {saturated})")


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------
def train(args):
    own_indices, opponent_indices, scores, results = load_dataset(args.data, args.cache)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {device}")

    # Keep the feature indices as int16 in RAM (4x smaller than int64) and cast each batch to long on the
    # GPU — lets us hold ~200M+ positions in memory instead of ~70M.
    own_indices = torch.from_numpy(own_indices)
    opponent_indices = torch.from_numpy(opponent_indices)
    scores = torch.from_numpy(scores)
    results = torch.from_numpy(results)

    target = args.wdl_lambda * results + (1.0 - args.wdl_lambda) * torch.sigmoid(scores / EVALUATION_SCALE)

    model = PerspectiveNetwork(args.hidden_size).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    # Deterministic train/validation split so val loss is comparable across runs (overfit detector).
    position_count = own_indices.shape[0]
    shuffle = torch.randperm(position_count, generator=torch.Generator().manual_seed(0))
    val_count = int(position_count * args.val_fraction)
    val_index = shuffle[:val_count]
    train_index = shuffle[val_count:]
    print(f"training on {train_index.numel():,} positions ({val_index.numel():,} held out for validation), "
          f"{args.epochs} epochs, batch {args.batch_size}, hidden {args.hidden_size}")

    def mean_loss_over(index):
        model.eval()
        total, seen = 0.0, 0
        with torch.no_grad():
            for start in range(0, index.numel(), args.batch_size):
                batch = index[start : start + args.batch_size]
                prediction = model(own_indices[batch].to(device).long(), opponent_indices[batch].to(device).long())
                loss = torch.sum((torch.sigmoid(prediction) - target[batch].to(device)) ** 2)
                total += loss.item()
                seen += batch.numel()
        return total / max(seen, 1)

    for epoch in range(args.epochs):
        model.train()
        permutation = train_index[torch.randperm(train_index.numel())]
        running_loss = 0.0
        batches = 0
        epoch_start = time.time()
        for start in range(0, permutation.numel(), args.batch_size):
            batch = permutation[start : start + args.batch_size]
            batch_own = own_indices[batch].to(device, non_blocking=True).long()
            batch_opponent = opponent_indices[batch].to(device, non_blocking=True).long()
            batch_target = target[batch].to(device, non_blocking=True)

            prediction = model(batch_own, batch_opponent)
            loss = torch.mean((torch.sigmoid(prediction) - batch_target) ** 2)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            batches += 1
        scheduler.step()
        val_loss = mean_loss_over(val_index) if val_count else 0.0
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train {running_loss / batches:.6f}  val {val_loss:.6f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - epoch_start:.1f}s")

    torch.save(model.state_dict(), args.out.replace(".nnue", ".pt"))
    export_quantised_net(model, args.out)


def _load_shard_npz(path, wdl_lambda):
    """Load one per-shard .npz into CPU tensors: (own, opponent, target). Indices stay int16 (cast per
    batch on the GPU); target folds score+wdl into the training label so we never keep both around."""
    cached = np.load(path)
    own = torch.from_numpy(cached["own"])
    opponent = torch.from_numpy(cached["opponent"])
    scores = torch.from_numpy(cached["scores"])
    results = torch.from_numpy(cached["results"])
    target = wdl_lambda * results + (1.0 - wdl_lambda) * torch.sigmoid(scores / EVALUATION_SCALE)
    return own, opponent, target


def stream_train(args):
    """Train over per-shard .npz caches ONE SHARD AT A TIME so RAM never holds the whole dataset — this is
    what lets us scale past ~230M positions. Shuffling is shard-order (reseeded per epoch) plus a full
    in-shard permutation; with ~95M-position shards that is effectively i.i.d. batches. The last shard is
    held out for a fixed validation-loss probe."""
    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, "*.npz")))
    if not shard_paths:
        raise SystemExit(f"no shard .npz files in {args.shard_dir}")
    validation_path = shard_paths[-1]
    train_paths = shard_paths[:-1] if len(shard_paths) > 1 else shard_paths

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {device}")
    print(f"streaming {len(train_paths)} training shards from {args.shard_dir} "
          f"(validation held out: {os.path.basename(validation_path)}), {args.epochs} epochs, "
          f"batch {args.batch_size}, hidden {args.hidden_size}", flush=True)

    model = PerspectiveNetwork(args.hidden_size).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    # Fixed validation subset (deterministic) so val loss is comparable across runs.
    val_own, val_opponent, val_target = _load_shard_npz(validation_path, args.wdl_lambda)
    val_limit = min(val_own.shape[0], 1_000_000)
    val_index = torch.randperm(val_own.shape[0], generator=torch.Generator().manual_seed(0))[:val_limit]
    val_own, val_opponent, val_target = val_own[val_index], val_opponent[val_index], val_target[val_index]

    def validation_loss():
        model.eval()
        total, seen = 0.0, 0
        with torch.no_grad():
            for start in range(0, val_limit, args.batch_size):
                stop = start + args.batch_size
                prediction = model(val_own[start:stop].to(device).long(),
                                   val_opponent[start:stop].to(device).long())
                total += torch.sum((torch.sigmoid(prediction) - val_target[start:stop].to(device)) ** 2).item()
                seen += val_own[start:stop].shape[0]
        return total / max(seen, 1)

    for epoch in range(args.epochs):
        model.train()
        shard_order = torch.randperm(len(train_paths), generator=torch.Generator().manual_seed(epoch)).tolist()
        running_loss, batches, positions = 0.0, 0, 0
        epoch_start = time.time()
        for shard_number in shard_order:
            own, opponent, target = _load_shard_npz(train_paths[shard_number], args.wdl_lambda)
            count = own.shape[0]
            permutation = torch.randperm(count)
            for start in range(0, count, args.batch_size):
                batch = permutation[start : start + args.batch_size]
                batch_own = own[batch].to(device, non_blocking=True).long()
                batch_opponent = opponent[batch].to(device, non_blocking=True).long()
                batch_target = target[batch].to(device, non_blocking=True)
                prediction = model(batch_own, batch_opponent)
                loss = torch.mean((torch.sigmoid(prediction) - batch_target) ** 2)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
                batches += 1
            positions += count
            del own, opponent, target
        scheduler.step()
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train {running_loss / max(batches, 1):.6f}  "
              f"val {validation_loss():.6f}  lr {scheduler.get_last_lr()[0]:.2e}  {positions:,} pos  "
              f"{time.time() - epoch_start:.0f}s", flush=True)

    torch.save(model.state_dict(), args.out.replace(".nnue", ".pt"))
    export_quantised_net(model, args.out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", help="one or more globs of fen;score;wdl shards (monolithic mode)")
    parser.add_argument("--out", help="output .nnue path")
    parser.add_argument("--featurise-shard", nargs=2, metavar=("TEXT", "NPZ"),
                        help="featurise a single text shard into NPZ and exit (for parallel featurisation)")
    parser.add_argument("--shard-dir", help="directory of per-shard .npz caches for streaming training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wdl-lambda", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--cache", default=None, help="optional .npz featurisation cache (load if present, else write)")
    parser.add_argument("--val-fraction", type=float, default=0.01, help="fraction held out for validation loss")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.featurise_shard:
        featurise_shard(args.featurise_shard[0], args.featurise_shard[1])
        return
    if not args.out:
        parser.error("--out is required for training")
    if args.shard_dir:
        stream_train(args)
    else:
        if not args.data:
            parser.error("--data is required for monolithic training (or pass --shard-dir)")
        train(args)


if __name__ == "__main__":
    main()
