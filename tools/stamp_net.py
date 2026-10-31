#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Stamp a weight payload into the shared Reckless `RNNUE1` net container.

The header is written by `tools/rnnue1.py` from its own constants — the architecture fields, the
king-bucket map's digest and the payload length all come from there, never from an argument. A payload of
any other shape therefore gets a header that describes *this* architecture and is refused by every
engine, rather than a header that describes the payload and loads the wrong weights.

Two inputs are accepted, and the only difference is where the payload starts:

    python3 tools/stamp_net.py nets/reckless-lc0-runf-sb870.nnue  # re-stamp in place
    python3 tools/stamp_net.py raw.payload nets/zenith-new.nnue  # a bare payload from the trainer

A file already in the RNNUE1 container is read back through its own checks first, so re-stamping a
corrupt net fails rather than quietly re-blessing it. A file carrying one of the retired 8-byte `ZNNUE`
magics has its magic dropped and the rest taken as the payload: those formats stored exactly these four
arrays in exactly this order, with the output bias already int32 and no trailing padding, so the weights
cross over byte for byte and the engine's evaluation does not move. `nnuecheck` and the bench signature
are what prove that.
"""
import sys

import rnnue1  # the shared net container, beside this script in tools/

RETIRED_MAGICS = (b"ZNNUE5\0\0", b"ZNNUE4\0\0", b"ZNNUE3\0\0")


def payload_of(path):
    """The weight payload inside `path`, whichever container it is in. Raises SystemExit if it is neither."""
    with open(path, "rb") as source:
        head = source.read(8)
    if head == rnnue1.MAGIC:
        return rnnue1.read_payload(path)
    with open(path, "rb") as source:
        image = source.read()
    if head in RETIRED_MAGICS:
        print(f"{path}: retired {head[:6].decode()} container, taking the payload after its 8-byte magic")
        return image[8:]
    if len(image) == rnnue1.PAYLOAD_BYTES:
        return image  # a bare payload straight from the trainer
    raise SystemExit(f"{path}: not an RNNUE1 net, not a retired ZNNUE net, and not a bare "
                     f"{rnnue1.PAYLOAD_BYTES}-byte payload ({len(image)} bytes)")


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: stamp_net.py <in.nnue|in.payload> [out.nnue]   (one argument stamps in place)")
    source_path = sys.argv[1]
    target_path = sys.argv[2] if len(sys.argv) == 3 else source_path
    payload = payload_of(source_path)
    if len(payload) != rnnue1.PAYLOAD_BYTES:
        raise SystemExit(f"{source_path}: its payload is {len(payload)} bytes; this architecture's payload is "
                         f"{rnnue1.PAYLOAD_BYTES} — stamp it with a tool built for its own architecture")
    written = rnnue1.write_net(target_path, payload)
    print(f"stamped {target_path}  ({written} bytes: {rnnue1.HEADER_BYTES}-byte RNNUE1 header, "
          f"format_version {rnnue1.FORMAT_VERSION}, payload_crc {rnnue1.crc64_xz(payload):#018x}, "
          f"king_bucket_map_crc {rnnue1.crc64_xz(rnnue1.KING_BUCKET_MAP):#018x})")


if __name__ == "__main__":
    main()
