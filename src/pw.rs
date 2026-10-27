//! Cross-check against the tables written by possiblywrong's solver
//! (`gobblet_<sizes>_<per_size>_<move>.dat`, see `reference/possiblywrong/run.sh`).
//!
//! That file is a hash table of 2^29 `u64`s: the low 54 bits of an entry are a position and the
//! high 10 bits its value. Positions are square-major (6 bits per square, 2 bits per size,
//! 01 = side to move, 10 = opponent), so they are re-packed into our size-major blocks and
//! canonicalized before looking up our value.

use std::io::Read;

use rayon::prelude::*;

use crate::rules::{self, value, State, Tables};

const EMPTY: u64 = 0x3;
const STATE_MASK: u64 = (1 << 54) - 1;

fn from_pw(square_major: u64) -> State {
    let mut s = 0;
    for c in 0..9 {
        for k in 0..3 {
            let f = (square_major >> (6 * c + 2 * k)) & 3;
            s |= f << (18 * k + 2 * c);
        }
    }
    s
}

/// Their value as (1 win / 0 draw / -1 loss, plies). For draws the field counts moves, not
/// plies, so it is dropped.
fn decode_pw(entry: u64) -> (i8, u8) {
    let v = 2 - (entry >> 62) as i8;
    let moves = ((entry << 2) as i64) >> 56;
    let moves = if moves < 0 { -moves - 1 } else { moves } as u8;
    (v, if v == 0 { 0 } else { moves })
}

pub struct Comparison {
    pub entries: u64,
    pub mismatches: u64,
    pub examples: Vec<String>,
}

pub fn compare(path: &str, t: &Tables, values: &[u8]) -> std::io::Result<Comparison> {
    let mut file = std::fs::File::open(path)?;
    let len = file.metadata()?.len() as usize;
    assert_eq!(len % 8, 0, "not a table of u64");
    let mut table = vec![0u64; len / 8];
    // SAFETY: any byte pattern is a valid u64. The file was written by a little-endian machine.
    file.read_exact(unsafe { std::slice::from_raw_parts_mut(table.as_mut_ptr() as *mut u8, len) })?;

    let check = |e: u64| {
        let s = rules::canonical(from_pw(e & STATE_MASK));
        let mine = value::decode(values[t.index(s) as usize]);
        let theirs = decode_pw(e);
        (mine != theirs).then(|| format!("ours={mine:?} theirs={theirs:?}\n{}", rules::show(s)))
    };
    let (entries, mismatches) = table
        .par_iter()
        .filter(|&&e| e != EMPTY)
        .map(|&e| (1u64, check(e).is_some() as u64))
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let examples = if mismatches == 0 {
        vec![]
    } else {
        table.iter().filter(|&&e| e != EMPTY).filter_map(|&e| check(e)).take(5).collect()
    };
    Ok(Comparison { entries, mismatches, examples })
}
