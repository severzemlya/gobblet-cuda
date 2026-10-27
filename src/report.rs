//! Summary statistics and their text output.

use std::fmt::Write as _;

use rayon::prelude::*;

use crate::rules::{self, value, Tables, Variant};

#[derive(Clone, Debug)]
pub struct PassStat {
    pub depth: u8,
    pub wins: u64,
    pub losses: u64,
    pub seconds: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub variant: Option<Variant>,
    pub backend: String,
    pub device: String,
    pub reachable: u64,
    pub terminal: u64,
    pub no_move_states: u64,
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub max_depth: u8,
    /// Value of the empty board for the first player: (1 / -1 / 0, plies).
    pub start: (i8, u8),
    /// First moves: (size 1..3, square, result for the first player, plies).
    pub first_moves: Vec<(u32, u32, i8, u8)>,
    /// Order-independent checksum of (index, value) over the reachable positions.
    pub checksum: u64,
    /// Breadth-first levels: (expanded, newly found, seconds).
    pub bfs: Vec<(u64, u64, f64)>,
    pub passes: Vec<PassStat>,
    pub time_enumerate: f64,
    pub time_retrograde: f64,
    pub time_total: f64,
}

#[inline]
fn mix(mut x: u64) -> u64 {
    // SplitMix64
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

pub fn checksum(values: &[u8], reachable: &[u32]) -> u64 {
    reachable.par_iter().map(|&i| mix((i as u64) << 8 | values[i as usize] as u64)).reduce(|| 0, u64::wrapping_add)
}

impl Report {
    pub fn new(variant: Variant, backend: &str) -> Self {
        Report { variant: Some(variant), backend: backend.into(), ..Default::default() }
    }

    pub fn summarize(&mut self, t: &Tables, variant: Variant, values: &[u8], reachable: &[u32]) {
        let (wins, losses, draws, max_depth) = reachable
            .par_iter()
            .map(|&i| match value::decode(values[i as usize]) {
                (1, d) => (1, 0, 0, d),
                (-1, d) => (0, 1, 0, d),
                _ => (0, 0, 1, 0),
            })
            .reduce(|| (0u64, 0u64, 0u64, 0u8), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3.max(b.3)));
        self.wins = wins;
        self.losses = losses;
        self.draws = draws;
        self.max_depth = max_depth;
        self.start = value::decode(values[t.index(rules::canonical(0)) as usize]);
        self.checksum = checksum(values, reachable);

        // The board is empty, so the first moves are "place size k on square c".
        self.first_moves.clear();
        for k in 0..variant.sizes {
            for c in 0..9u32 {
                let after = 1u64 << (18 * k + 2 * c);
                let (r, d) = value::decode(values[t.index(rules::next_state(after)) as usize]);
                // The stored value is the second player's: flip it and add the first ply.
                self.first_moves.push((k + 1, c, -r, if r == 0 { 0 } else { d + 1 }));
            }
        }
    }

    pub fn to_text(&self) -> String {
        let v = self.variant.expect("variant");
        let mut s = String::new();
        let _ = writeln!(
            s,
            "== Gobblet Gobblers retrograde analysis ({}) backend={} {}",
            v.name(),
            self.backend,
            self.device
        );
        let _ = writeln!(s, "reachable positions (up to symmetry and side swap): {}", self.reachable);
        let _ = writeln!(s, "  breadth-first levels: {}", self.bfs.len());
        let _ = writeln!(s, "  terminal: {}", self.terminal);
        if self.no_move_states > 0 {
            let _ = writeln!(s, "  non-terminal without legal moves (draws): {}", self.no_move_states);
        }
        let _ = writeln!(
            s,
            "side to move wins {} / loses {} / draws {} (longest: {} plies)",
            self.wins, self.losses, self.draws, self.max_depth
        );
        let _ = writeln!(s, "empty board: {}", describe(self.start));
        let _ = writeln!(s, "first moves (result for the first player; squares 0 1 2 / 3 4 5 / 6 7 8):");
        for k in 1..=v.sizes {
            let row: Vec<String> = self.first_moves.iter().filter(|m| m.0 == k).map(|m| short(m.2, m.3)).collect();
            let name = ["small ", "medium", "large "][k as usize - 1];
            let _ = writeln!(s, "  {name}: {}", row.join(" "));
        }
        let _ = writeln!(s, "checksum: {:016x}", self.checksum);
        let _ = writeln!(
            s,
            "time: enumerate {:.3}s / retrograde {:.3}s / total {:.3}s ({} passes)",
            self.time_enumerate,
            self.time_retrograde,
            self.time_total,
            self.passes.len().saturating_sub(1)
        );
        s
    }

    pub fn passes_text(&self) -> String {
        let mut s = String::from("level\texpanded\tfound\tseconds\n");
        for (level, (expanded, found, secs)) in self.bfs.iter().enumerate() {
            let _ = writeln!(s, "{level}\t{expanded}\t{found}\t{secs:.4}");
        }
        s.push_str("depth\twins\tlosses\tseconds\n");
        for p in &self.passes {
            let _ = writeln!(s, "{}\t{}\t{}\t{:.4}", p.depth, p.wins, p.losses, p.seconds);
        }
        s
    }
}

fn describe((r, d): (i8, u8)) -> String {
    match r {
        1 => format!("first player wins in {d} plies"),
        -1 => format!("first player loses in {d} plies"),
        _ => "draw".into(),
    }
}

fn short(r: i8, d: u8) -> String {
    match r {
        1 => format!("W{d:<2}"),
        -1 => format!("L{d:<2}"),
        _ => "D  ".into(),
    }
}
