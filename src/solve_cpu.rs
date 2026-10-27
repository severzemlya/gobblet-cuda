//! Reference solver on the CPU (rayon).
//!
//! 1. Enumerate every canonical position reachable from the empty board, breadth first
//!    (finished games are not expanded).
//! 2. Depth 0 is the terminal check. Depth 1 depends only on the terminal check of the next
//!    positions, so it is decided without reading the value table.
//! 3. For k = 2, 3, …, an unknown position becomes
//!    - a win in k if some move reaches a loss of depth < k, or
//!    - a loss in k if every move reaches a win of depth < k.
//!
//!    Values written during pass k have depth k and are ignored by the same pass, so the passes
//!    can update the table in place. When a pass decides nothing, the remaining positions are draws.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
use std::time::Instant;

use rayon::prelude::*;

use crate::report::{PassStat, Report};
use crate::rules::{self, value, Outcome, Tables, Variant};

pub struct Solution {
    /// Value per position index (unreachable indices stay 0).
    pub values: Vec<u8>,
    /// Indices of the reachable positions, in breadth-first order.
    pub reachable: Vec<u32>,
    pub report: Report,
}

struct Bitmap(Vec<AtomicU64>);

impl Bitmap {
    fn new(bits: u64) -> Self {
        Bitmap((0..bits.div_ceil(64)).map(|_| AtomicU64::new(0)).collect())
    }
    /// Set bit `i`; true if it was not set before.
    #[inline(always)]
    fn insert(&self, i: u32) -> bool {
        let bit = 1u64 << (i % 64);
        self.0[(i / 64) as usize].fetch_or(bit, Relaxed) & bit == 0
    }
}

pub fn solve(variant: Variant) -> Solution {
    let t = Tables::new();
    let n = variant.num_indices();
    let mut report = Report::new(variant, "cpu");
    let total = Instant::now();

    // ---- 1. enumerate reachable positions ----
    let phase = Instant::now();
    let seen = Bitmap::new(n);
    let start = t.index(rules::canonical(0));
    seen.insert(start);
    let mut reachable: Vec<u32> = vec![start];
    let mut lo = 0;
    while lo < reachable.len() {
        let level = Instant::now();
        let next: Vec<u32> = reachable[lo..]
            .par_chunks(1 << 12)
            .flat_map_iter(|chunk| {
                let mut out = Vec::new();
                for &i in chunk {
                    let s = t.state(i);
                    if rules::terminal(s).is_some() {
                        continue;
                    }
                    rules::for_each_move(s, variant, |m| {
                        let j = t.index(rules::next_state(m));
                        if seen.insert(j) {
                            out.push(j);
                        }
                        true
                    });
                }
                out
            })
            .collect();
        report.bfs.push(((reachable.len() - lo) as u64, next.len() as u64, level.elapsed().as_secs_f64()));
        lo = reachable.len();
        reachable.extend(next);
    }
    drop(seen);
    report.reachable = reachable.len() as u64;
    report.time_enumerate = phase.elapsed().as_secs_f64();

    // ---- 2. depths 0 and 1, from the terminal check alone ----
    let phase = Instant::now();
    let values: Vec<AtomicU8> = (0..n).into_par_iter().map(|_| AtomicU8::new(0)).collect();
    let pass = Instant::now();
    let [no_moves, win1, loss1, terminal] = [(); 4].map(|_| AtomicU64::new(0));
    let mut unknown: Vec<u32> = reachable
        .par_iter()
        .copied()
        .filter(|&i| {
            let s = t.state(i);
            let v = &values[i as usize];
            match rules::terminal(s) {
                Some(Outcome::Win) => v.store(value::win(0), Relaxed),
                Some(Outcome::Loss) => v.store(value::loss(0), Relaxed),
                None => {
                    let (mut any, mut all_lose) = (false, true);
                    let finished = rules::for_each_move(s, variant, |m| {
                        any = true;
                        match rules::terminal(rules::swap(m)) {
                            Some(Outcome::Loss) => false, // we completed a line: win in 1
                            r => {
                                all_lose &= r == Some(Outcome::Win); // we uncovered theirs
                                true
                            }
                        }
                    });
                    if !finished {
                        v.store(value::win(1), Relaxed);
                        win1.fetch_add(1, Relaxed);
                    } else if !any {
                        // No legal move (placement-only rules, all pieces played): stays a draw.
                        no_moves.fetch_add(1, Relaxed);
                    } else if all_lose {
                        v.store(value::loss(1), Relaxed);
                        loss1.fetch_add(1, Relaxed);
                    } else {
                        return true;
                    }
                    return false;
                }
            }
            terminal.fetch_add(1, Relaxed);
            false
        })
        .collect();
    report.no_move_states = no_moves.into_inner();
    report.terminal = terminal.into_inner();
    report.passes.push(PassStat { depth: 0, wins: 0, losses: 0, seconds: 0.0 });
    report.passes.push(PassStat {
        depth: 1,
        wins: win1.into_inner(),
        losses: loss1.into_inner(),
        seconds: pass.elapsed().as_secs_f64(),
    });

    // ---- 3. depths 2, 3, … ----
    for k in 2..=value::MAX_DEPTH {
        let pass = Instant::now();
        let wins = AtomicU64::new(0);
        let losses = AtomicU64::new(0);
        unknown.par_iter().for_each(|&i| {
            let s = t.state(i);
            let mut all_win = true;
            let finished = rules::for_each_move(s, variant, |m| {
                let v = values[t.index(rules::next_state(m)) as usize].load(Relaxed);
                if value::is_loss_before(v, k) {
                    return false; // a move that beats the opponent
                }
                all_win &= value::is_win_before(v, k);
                true
            });
            if !finished {
                values[i as usize].store(value::win(k), Relaxed);
                wins.fetch_add(1, Relaxed);
            } else if all_win {
                values[i as usize].store(value::loss(k), Relaxed);
                losses.fetch_add(1, Relaxed);
            }
        });
        let (w, l) = (wins.into_inner(), losses.into_inner());
        report.passes.push(PassStat { depth: k, wins: w, losses: l, seconds: pass.elapsed().as_secs_f64() });
        if w + l == 0 {
            break;
        }
        unknown = unknown.into_par_iter().filter(|&i| values[i as usize].load(Relaxed) == value::UNKNOWN).collect();
    }
    report.time_retrograde = phase.elapsed().as_secs_f64();
    report.time_total = total.elapsed().as_secs_f64();

    let values = into_bytes(values);
    report.summarize(&t, variant, &values, &reachable);
    Solution { values, reachable, report }
}

/// Reinterpret the atomic table as plain bytes without copying 2.9 GB.
fn into_bytes(v: Vec<AtomicU8>) -> Vec<u8> {
    let mut v = ManuallyDrop::new(v);
    // SAFETY: AtomicU8 has the same size, alignment and bit validity as u8.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut u8, v.len(), v.capacity()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_sizes_matches_possiblywrong() {
        // possiblywrong reports 252,238 states for (2 sizes, 2 per size, moves allowed).
        let sol = solve(Variant { sizes: 2, allow_move: true });
        assert_eq!(sol.report.reachable, 252_238);
        assert_eq!(sol.report.start, (0, 0)); // draw
    }

    #[test]
    fn two_sizes_placement_only_is_a_draw() {
        let sol = solve(Variant { sizes: 2, allow_move: false });
        assert_eq!(sol.report.reachable, 145_750);
        assert_eq!(sol.report.start, (0, 0));
    }

    /// The full game: about 3 GB of memory and half a minute on 8 cores.
    #[test]
    #[ignore]
    fn standard_is_a_first_player_win_in_13() {
        let sol = solve(Variant::STANDARD);
        assert_eq!(sol.report.reachable, 341_024_631);
        assert_eq!(sol.report.draws, 104_150);
        assert_eq!(sol.report.start, (1, 13));
    }
}
