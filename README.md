# gobblet-cuda

Strongly solves [Gobblet Gobblers](https://boardgamegeek.com/boardgame/32004/gobblet-gobblers)
(3×3 board, two pieces each of three sizes) by retrograde analysis of every reachable position.
The whole game takes **0.37 s on a B200, 0.43 s on an H100 and 0.95 s on an RTX 3090**.

There is a CPU solver (Rust + rayon) and a GPU solver (CUDA kernels compiled at run time with
NVRTC). The rules are implemented twice, independently, so the two tables check each other.

## Results

| Rules | Reachable positions | Result |
|---|---|---|
| Standard (pieces may move) | 341,024,631 | **first player wins in 13 plies** |
| Placement only (`--no-move`) | 148,599,441 | draw |
| Two sizes (`--sizes 2`) | 252,238 | draw |

Positions are counted up to the 8 board symmetries and swapping sides. In the standard game
104,150 positions are draws, and with best play no decided position lasts more than 24 plies.

First moves in the standard game (result for the first player, plies; squares `0 1 2 / 3 4 5 / 6 7 8`):

```
small : W15 W13 W15 W13 W15 W13 W15 W13 W15
medium: L14 L12 L14 L12 L12 L12 L14 L12 L14
large : W13 W15 W13 W15 W13 W15 W13 W15 W13
```

A medium piece loses wherever it goes; a small or a large piece wins wherever it goes.

## Speed (standard game)

| Solver | Hardware | Enumerate | Retrograde | Total |
|---|---|---|---|---|
| **GPU** | NVIDIA B200 | 0.162 s | 0.204 s | **0.366 s** |
| GPU | NVIDIA H100 SXM 80GB HBM3 | 0.176 s | 0.255 s | 0.431 s |
| GPU | NVIDIA RTX 3090 (250 W cap) | 0.496 s | 0.452 s | 0.948 s |
| CPU | Apple M3 (8 cores) | 11.5 s | 19.0 s | 30.6 s |
| [possiblywrong](https://github.com/possibly-wrong/gobblet) (C++, single thread) | Ryzen 9 5950X | | | 17 min 03 s |

GPU totals run from "lookup tables on the device" to "every value decided", median of 3–5 runs
(spread under 0.01 s). Process start, CUDA initialization, NVRTC compilation and copying the
2.9 GB table back for the summary add about 7 s of wall-clock time. Logs are in [`bench/`](bench/);
the H100 and B200 runs (on interruptible [vast.ai](https://vast.ai) instances, using
[`bench/vast-run.sh`](bench/vast-run.sh)) come from a build whose output was still in Japanese,
with the same solver code.

On the B200 the enumeration is barely faster than on the H100 (0.176 → 0.162 s) despite ~2.4× the
memory bandwidth: past the H100 it is bound by the atomics on the bitmap and by computing the next
positions, not by bandwidth. The retrograde passes do scale with bandwidth.

Other published solves of the same game take about 10 min (Miyagi Prefectural Sendai Daisan
High School math club, C), 25 min (possiblywrong, C++, laptop), 30 min
([brianhliou](https://github.com/brianhliou/gobblet-gobblers), Rust, 14 cores) and a day
([Qiita](https://qiita.com/yasagureprog/items/94c0cb01005b2b94b837), Python on Colab).

## Verification

- Every position's result and depth matches the table produced by possiblywrong's independent
  solver ([`reference/possiblywrong/run.sh`](reference/possiblywrong/run.sh)), for both the
  standard and the placement-only rules: 0 mismatches (`gobblet-cuda pw`).
- The CPU and GPU tables agree on all 2,881,473,967 indices for the standard and placement-only
  rules, and on all indices of both two-size variants (`gobblet-cuda compare`).
- The reachable counts match possiblywrong's published 341,024,631 / 148,599,441 / 252,238.

## How it works

- **Encoding.** A position is 54 bits from the side to move's view: 18 bits per piece size,
  2 bits per square (ours / theirs). Larger pieces sit on top, so any bit pattern is a valid board.
- **Dense indexing.** One size can be placed in Σ_{a,b≤2} C(9,a)·C(9−a,b) = 1423 ways, so a
  position maps to an index in 0..1423³ (≈ 2.9 billion) with three lookups in a 2^18-entry table.
  No hash table anywhere.
- **Symmetry.** Only the smallest of the 8 symmetric images (as an integer) carries a value.
  Alternating two mirrors (top-bottom and anti-diagonal) visits all 8 images in 7 cheap steps.
- **Enumeration.** Breadth first from the empty board; finished games are not expanded. On the GPU,
  a plain read of the bitmap filters out positions already found before any atomic is issued.
- **Depths 0 and 1.** Depth 0 is the terminal check. Depth 1 (complete a line / every move uncovers
  an opponent's line) needs only the terminal check of the next boards, not the value table.
- **Retrograde passes.** For k = 2, 3, …: an unknown position is a win in k if a move reaches a
  loss of depth < k, and a loss in k if every move reaches a win of depth < k. Values written in
  pass k have depth k and are ignored by the same pass, so the table is updated in place and the
  positions inside a pass are independent. Positions still unknown at the end are draws.
- **Values.** One byte per index (win / loss and depth): 2.9 GB for the standard game.

The rules follow possiblywrong: the result is judged on the board after the whole move (lifting a
piece that uncovers an opponent's line is fine if the moved piece covers that line again), if a move
leaves lines for both players the opponent wins, a piece never returns to its own square, and play
that never ends is a draw.

## Usage

```sh
cargo build --release                  # CPU solver only; works without CUDA (e.g. on macOS)
./target/release/gobblet-cuda solve --passes
cargo test --release                   # small variants; add -- --ignored for the full game

cargo build --release --features cuda  # needs the CUDA toolkit (nvcc and NVRTC)
./target/release/gobblet-cuda solve --backend gpu --passes
./target/release/gobblet-cuda compare  # CPU vs GPU, all indices
./target/release/gobblet-cuda pw path/to/gobblet_3_2_1.dat --backend gpu
```

Options: `--sizes 1..3`, `--no-move` (placement only). The CPU solver of the standard game needs
about 3.3 GB of RAM; the GPU solver needs about 7.6 GB of device memory. If `nvcc` is not on the
`PATH` at build time, set `CUDARC_CUDA_VERSION` (for example `12080`).

## License

MIT
