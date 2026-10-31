# Third-party notices

Zenith itself is free software under the **GNU General Public License v3.0 or later** — see
[LICENSE](LICENSE). This file is the single home for the notices Zenith owes for third-party material it
ships or embeds, and for the attributions it keeps on the record. It is shipped in every release archive
beside the engine, and `zenith license` prints the training-data notice from the binary itself - as a CLI
subcommand and as a command of the UCI loop, `licence` accepted for either, as it is in every Reckless
engine. A release binary embeds the network, so it has to embed the notice with it.

---

## 1. NNUE training data — Leela Chess Zero self-play training data (ODbL 1.0 / DbCL 1.0)

**This is an attribution obligation, and it applies to every Zenith binary from 3.0.0 onward**, because the
network is embedded in the engine at build time (`Makefile NET`).

The network Zenith ships from 3.0.0, `nets/reckless-lc0-runf-sb870.nnue`, was trained **entirely** on
**Leela Chess Zero self-play training data**, published at
<https://storage.lczero.org/files/training_data/>. That database is licensed under the **Open Database
License 1.0 (ODbL-1.0)**, <http://opendatacommons.org/licenses/odbl/1.0/>, with its contents under the
**Database Contents License 1.0 (DbCL-1.0)**, <http://opendatacommons.org/licenses/dbcl/1.0/>.

A trained network is a **Produced Work** under ODbL 1.0, and distributing a Produced Work obliges a notice
naming the source database and its licence. That notice is this:

> Position data from the Leela Chess Zero self-play training data
> (https://storage.lczero.org/files/training_data/), run test91, used under the Open Database
> License 1.0 (http://opendatacommons.org/licenses/odbl/1.0/).

> Position data from the Leela Chess Zero self-play training data
> (https://storage.lczero.org/files/training_data/), run test60, used under the Open Database
> License 1.0 (http://opendatacommons.org/licenses/odbl/1.0/).

One line per run, because a notice naming only one of the two runs the pool was drawn from is an
incomplete notice. The pool behind the shipped network was **62.83 % `test91`** and **37.17 % `test60`**.
Each line is the one the extractor in the shared Reckless NNUE pipeline writes into every shard's `.meta`
provenance sidecar, which records the source name, the source home, the run, and both licences with their
URLs for each shard it writes.

**What the obligation is, and what it is not.**

- The obligation is **the notice**. ODbL 1.0 does **not** require the network itself to be licensed under
  ODbL, and `nets/reckless-lc0-runf-sb870.nnue` is **not** ODbL-licensed: it is a Produced Work, and the
  licence's share-alike reaches derivative *databases*, not Produced Works made from one.
- The network is **not** GPL merely because GPLv3 code was used to train it or reads it. Training weights is
  not a derivative work of the training program, and the weights are a data file the engine loads; Zenith's
  *source code* is GPLv3-or-later, and that is a separate matter from the network's status.
- The network was **not** trained in this repository. Zenith has no trainer of its own: the trainers, the
  architecture and the `RNNUE1` container live in
  [`nnue-training`](https://github.com/reckless-technology/nnue-training), the single source for the
  Reckless NNUE, and this engine is a consumer that loads a net and verifies it. Zenith's `datagen` can
  generate training data, but **it did not produce the shipped weights**, which is why the file does not
  carry a `zenith-` name. The *data* is Leela's, under ODbL; the *network* came from the shared Reckless
  pipeline.

See [NNUE_TRAINING.md §9](NNUE_TRAINING.md#9-current-release-net) for the net's architecture, lineage and
measurements.

---

## 2. Perft reference counts — the Ethereal project (GPL-3.0)

`src/uci.c` carries the 128-position `standard.epd` perft suite from Ethereal
(<https://github.com/AndyGrant/Ethereal>) — FENs with their known-correct node counts, used as movegen
reference data by `./build/zenith perft`. Ethereal is licensed GPL-3.0, which is compatible with Zenith's
GPL-3.0-or-later. **No Ethereal code is included or derived**: only the test positions and their expected
counts.

---

## 3. Polyglot opening-book format

`src/book.c` implements the standard **Polyglot** `.bin` format. Its Zobrist key constants come from the
public [book-format specification](http://hgm.nubati.net/book_format.html) (the 781 random values the format
defines), and `./build/zenith bookcheck` validates Zenith's key computation against the 9 official spec test
vectors. The format and its constants are a published specification; the implementation is Zenith's own.

The opening book Zenith ships and embeds, `books/zenith-book-r5.bin`, is built **entirely from Zenith's own
self-play** by `tools/build_book.py` and `tools/add_book_moves.py`. No third-party book is bundled. The
optional `make get-book` convenience target *downloads* `performance.bin` from the GPL-3.0
[python-chess](https://github.com/niklasf/python-chess) repository into the gitignored `books/`; that file is
not part of this repository and is not distributed with any release.

---

## 4. On the record: the earlier networks' training data (no notice owed)

Zenith's networks up to and including `zenith-cap2` were trained on positions from a public self-play dataset
generated by **PlentyChess**, by **Yoshie2000**. That data is no longer used, those networks are no longer in
the tree, and **no release ships a network derived from it**, so nothing is owed here — the credit is kept
because those networks were measurably stronger for having been trained on that data and the record should
show where they came from. The full lineage is in
[NNUE_TRAINING.md](NNUE_TRAINING.md#provenance-of-earlier-networks).

The techniques Zenith implements — bitboards, magic bitboards, PVS, transposition tables,
null-move/LMR/futility pruning, SEE, NNUE, Lazy SMP — are published and textbook (see the
[Chess Programming Wiki](https://www.chessprogramming.org)), and are implemented here independently. Zenith
shares no source code with any other engine.

---

## 5. What is and is not third-party code here

Stated from the tree rather than from habit. Every file in `src/` and `datagen/` carries
`SPDX-License-Identifier: GPL-3.0-or-later`, there is no other SPDX identifier anywhere in the repository,
and nothing is vendored: `src/generated/*.inc` are produced from this repository's own
`tools/generate_tables.py`. The engine links the C runtime, `libm` and the platform's threads and nothing
else, so **no third-party code is redistributed in a release**.

What *is* third-party in a release is **data**, which is why sections 1 to 3 exist: the NNUE network's
training positions (section 1), Ethereal's perft reference counts (section 2) and the Polyglot
specification's 781-value random array (section 3), the last two compiled into the binary.

The net and book tools in `tools/` need Python — and nothing else: every one of them is **stdlib-only**,
including `tools/rnnue1.py`, which is a verbatim copy of `nnue-training/spec/rnnue1.py` (GPL-3.0-or-later,
Copyright (C) 2026 Jonny Reckless, the same licence and author as this repository, so it is a sibling copy
and not third-party material). No release ships those tools and no binary links them, so Python is a
build-time dependency of a developer's machine, not material Zenith distributes. The trainers that need
PyTorch are in `nnue-training` and are not in this tree at all.

**On bullet, and why no MIT notice is owed here.** `./build/zenith-bullet2text` (`datagen/datagen.c`)
reads the 32-byte `bulletformat::ChessBoard` record that the [bullet](https://github.com/jw1912/bullet)
trainer writes. That is **format compatibility, independently implemented from the record layout** — no
bullet code is included, derived from or linked here, so bullet's MIT licence attaches nothing to this
repository. (`nnue-training`, which does start two files from bullet's own examples, carries the MIT
notice bullet is owed; it is owed there and not here.)
