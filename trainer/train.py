"""Zenith NNUE trainer (PyTorch, independent of any other engine's trainer).

Reads Zenith's own self-play records (`fen;stm_score_cp;wdl`), trains a 768->HIDDEN perspective network
with SCReLU, and exports a quantised `.nnue` file that the C++ engine loads directly. A float checkpoint
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
                      PADDING_INDEX, QUANT_ACCUMULATOR, QUANT_OUTPUT, position_features)


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
        fen, score_text, wdl_text = line.rsplit(";", 2)
        _, own, opponent = position_features(fen)
        if len(own) > MAX_ACTIVE_FEATURES:
            continue
        own_indices[kept, : len(own)] = own
        opponent_indices[kept, : len(opponent)] = opponent
        scores[kept] = float(score_text)
        results[kept] = float(wdl_text)
        kept += 1
    return own_indices[:kept], opponent_indices[:kept], scores[:kept], results[:kept]


def load_dataset(data_glob, cache_path=None):
    """Featurise all shards into padded index arrays. One pass, one process (robust and debuggable); a
    per-shard progress line is printed. If `cache_path` is given it is loaded when present and written
    otherwise, so repeated training runs on the same data skip featurisation entirely."""
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        print(f"loaded featurised cache {cache_path}: {cached['scores'].shape[0]:,} positions")
        return cached["own"], cached["opponent"], cached["scores"], cached["results"]

    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise SystemExit(f"no data files matched: {data_glob}")

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
        # Feature transformer as an embedding table: row i (i < 768) is the weight column for feature i,
        # which is exactly the [768, HIDDEN] on-disk layout. Row 768 is a forced-zero padding slot.
        self.feature_transformer = nn.Embedding(INPUT_FEATURES + 1, hidden_size, padding_idx=PADDING_INDEX)
        self.feature_transformer_bias = nn.Parameter(torch.zeros(hidden_size))
        self.output = nn.Linear(2 * hidden_size, 1)
        nn.init.normal_(self.feature_transformer.weight, std=0.01)
        with torch.no_grad():
            self.feature_transformer.weight[PADDING_INDEX].zero_()

    def forward(self, own_indices, opponent_indices):
        own_accumulator = self.feature_transformer(own_indices).sum(dim=1) + self.feature_transformer_bias
        opponent_accumulator = self.feature_transformer(opponent_indices).sum(dim=1) + self.feature_transformer_bias
        hidden = torch.cat([screlu(own_accumulator), screlu(opponent_accumulator)], dim=1)
        return self.output(hidden).squeeze(1)


# --------------------------------------------------------------------------------------------------
# Quantised export
# --------------------------------------------------------------------------------------------------
def export_quantised_net(model, path):
    with torch.no_grad():
        transformer = model.feature_transformer.weight[:INPUT_FEATURES].cpu().numpy()  # [768, HIDDEN]
        transformer_bias = model.feature_transformer_bias.cpu().numpy()                # [HIDDEN]
        output_weight = model.output.weight[0].cpu().numpy()                           # [2*HIDDEN]
        output_bias = float(model.output.bias[0].cpu())

    transformer_q = np.rint(transformer * QUANT_ACCUMULATOR).astype(np.int16)
    transformer_bias_q = np.rint(transformer_bias * QUANT_ACCUMULATOR).astype(np.int16)
    output_weight_q = np.rint(output_weight * QUANT_OUTPUT).astype(np.int16)
    output_bias_q = np.int32(round(output_bias * QUANT_ACCUMULATOR * QUANT_OUTPUT))

    with open(path, "wb") as handle:
        handle.write(NNUE_MAGIC)
        handle.write(transformer_q.tobytes())        # [768][HIDDEN] int16, feature-major
        handle.write(transformer_bias_q.tobytes())   # [HIDDEN] int16
        handle.write(output_weight_q.tobytes())      # [2*HIDDEN] int16
        handle.write(struct.pack("<i", int(output_bias_q)))  # int32
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

    own_indices = torch.from_numpy(own_indices.astype(np.int64))
    opponent_indices = torch.from_numpy(opponent_indices.astype(np.int64))
    scores = torch.from_numpy(scores)
    results = torch.from_numpy(results)

    target = args.wdl_lambda * results + (1.0 - args.wdl_lambda) * torch.sigmoid(scores / EVALUATION_SCALE)

    model = PerspectiveNetwork(args.hidden_size).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    position_count = own_indices.shape[0]
    print(f"training on {position_count:,} positions for {args.epochs} epochs, batch {args.batch_size}")

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(position_count)
        running_loss = 0.0
        batches = 0
        epoch_start = time.time()
        for start in range(0, position_count, args.batch_size):
            batch = permutation[start : start + args.batch_size]
            batch_own = own_indices[batch].to(device, non_blocking=True)
            batch_opponent = opponent_indices[batch].to(device, non_blocking=True)
            batch_target = target[batch].to(device, non_blocking=True)

            prediction = model(batch_own, batch_opponent)
            loss = torch.mean((torch.sigmoid(prediction) - batch_target) ** 2)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            batches += 1
        scheduler.step()
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {running_loss / batches:.6f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - epoch_start:.1f}s")

    torch.save(model.state_dict(), args.out.replace(".nnue", ".pt"))
    export_quantised_net(model, args.out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="glob of fen;score;wdl shard files")
    parser.add_argument("--out", required=True, help="output .nnue path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wdl-lambda", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--cache", default=None, help="optional .npz featurisation cache (load if present, else write)")
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
