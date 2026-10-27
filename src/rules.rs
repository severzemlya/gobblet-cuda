//! Position encoding and game rules.
//!
//! A position is a 54-bit integer (`u64`) seen from the side to move. Each piece size owns an
//! 18-bit block: block k (0 = small, 1 = medium, 2 = large) starts at bit 18k. Inside a block,
//! square c (0..9, row-major) uses two bits: the low bit is a piece of the side to move, the high
//! bit a piece of the opponent.
//!
//! Pieces of different sizes can share a square and the larger one is always on top, so the
//! stacking order is implied by the sizes and every bit pattern is a physically possible board.
//!
//! The rules follow possiblywrong (<https://github.com/possibly-wrong/gobblet>):
//! - on your turn, play a piece from your reserve or move one of your visible pieces
//! - a piece may go to an empty square or over smaller pieces only (never back to its own square)
//! - the result is judged on the board after the whole move; lifting a piece that uncovers an
//!   opponent's line is fine as long as the moved piece covers that line again
//! - if a move leaves lines for both players, the opponent wins
//! - play that never ends is a draw

pub type State = u64;

pub const BLOCK_BITS: u32 = 18;
pub const BLOCK_MASK: u64 = (1 << BLOCK_BITS) - 1;
/// The low bit of every square in a block (9 bits).
pub const CELL_LOW: u64 = 0x15555;
/// The side-to-move bits of all three blocks.
pub const ME: u64 = CELL_LOW | CELL_LOW << 18 | CELL_LOW << 36;
/// Pieces of each size per player.
pub const PER_SIZE: u32 = 2;
/// Placements of one size (one block): Σ_{a,b≤2} C(9,a)·C(9−a,b).
pub const BLOCK_STATES: u64 = 1423;

/// The 8 lines of three, as masks of the squares' low bits.
pub const LINES: [u64; 8] = [
    0x15, 0x540, 0x15000, // rows
    0x1041, 0x4104, 0x10410, // columns
    0x10101, 0x1110, // diagonals
];

/// The 8 board symmetries: `PERMS[t][c]` is where square c goes. Used to test `canonical`.
#[cfg_attr(not(test), allow(dead_code))]
pub const PERMS: [[u8; 9]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8], // identity
    [2, 5, 8, 1, 4, 7, 0, 3, 6], // rotate 90°
    [8, 7, 6, 5, 4, 3, 2, 1, 0], // rotate 180°
    [6, 3, 0, 7, 4, 1, 8, 5, 2], // rotate 270°
    [2, 1, 0, 5, 4, 3, 8, 7, 6], // mirror left-right
    [6, 7, 8, 3, 4, 5, 0, 1, 2], // mirror top-bottom
    [0, 3, 6, 1, 4, 7, 2, 5, 8], // transpose
    [8, 5, 2, 7, 4, 1, 6, 3, 0], // anti-transpose
];

/// Both bits of square 0, in all three blocks.
const FIELD3: u64 = 3 | 3 << 18 | 3 << 36;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    /// Number of piece sizes (1..=3).
    pub sizes: u32,
    /// Whether pieces on the board may be moved (false: placement only).
    pub allow_move: bool,
}

impl Variant {
    pub const STANDARD: Variant = Variant { sizes: 3, allow_move: true };

    /// Size of the index space (1423^sizes).
    pub fn num_indices(&self) -> u64 {
        BLOCK_STATES.pow(self.sizes)
    }

    pub fn name(&self) -> String {
        format!("sizes={} per_size={} move={}", self.sizes, PER_SIZE, self.allow_move as u8)
    }
}

/// Dense position ⇄ index tables: each block maps to 0..1423, so a position maps to 0..1423³.
pub struct Tables {
    /// 18-bit block → 0..1423 (`u16::MAX` for impossible patterns).
    pub rank: Vec<u16>,
    /// 0..1423 → 18-bit block.
    pub unrank: Vec<u32>,
}

impl Tables {
    pub fn new() -> Self {
        let mut rank = vec![u16::MAX; 1 << BLOCK_BITS];
        let mut unrank = Vec::with_capacity(BLOCK_STATES as usize);
        for p in 0..(1u32 << BLOCK_BITS) {
            let me = p & CELL_LOW as u32;
            let op = (p >> 1) & CELL_LOW as u32;
            if me & op != 0 || me.count_ones() > PER_SIZE || op.count_ones() > PER_SIZE {
                continue;
            }
            rank[p as usize] = unrank.len() as u16;
            unrank.push(p);
        }
        assert_eq!(unrank.len() as u64, BLOCK_STATES);
        Tables { rank, unrank }
    }

    #[inline(always)]
    pub fn index(&self, s: State) -> u32 {
        let n = BLOCK_STATES as u32;
        let r = |k: u32| self.rank[((s >> (BLOCK_BITS * k)) & BLOCK_MASK) as usize] as u32;
        r(0) + n * (r(1) + n * r(2))
    }

    #[inline(always)]
    pub fn state(&self, i: u32) -> State {
        let n = BLOCK_STATES as u32;
        let b0 = self.unrank[(i % n) as usize] as u64;
        let j = i / n;
        let b1 = self.unrank[(j % n) as usize] as u64;
        let b2 = self.unrank[(j / n) as usize] as u64;
        b0 | b1 << 18 | b2 << 36
    }
}

impl Default for Tables {
    fn default() -> Self {
        Self::new()
    }
}

/// Swap sides (the same board seen from the opponent).
#[inline(always)]
pub fn swap(s: State) -> State {
    ((s & ME) << 1) | ((s >> 1) & ME)
}

/// Apply a square permutation to all three blocks (slow; used by tests).
#[cfg_attr(not(test), allow(dead_code))]
pub fn permute(s: State, perm: &[u8; 9]) -> State {
    let mut out = 0;
    for (c, &d) in perm.iter().enumerate() {
        let x = s & (FIELD3 << (2 * c));
        let shift = 2 * (d as i32 - c as i32);
        out |= if shift >= 0 { x << shift } else { x >> -shift };
    }
    out
}

/// Mask of the given squares, replicated over the three blocks.
const fn cells3(cells: &[usize]) -> u64 {
    let mut m = 0;
    let mut i = 0;
    while i < cells.len() {
        m |= FIELD3 << (2 * cells[i]);
        i += 1;
    }
    m
}
const ROW0: u64 = cells3(&[0, 1, 2]);
const ROW1: u64 = cells3(&[3, 4, 5]);
const ROW2: u64 = cells3(&[6, 7, 8]);
const AT_P16: u64 = cells3(&[0]);
const AT_P8: u64 = cells3(&[1, 3]);
const AT_0: u64 = cells3(&[2, 4, 6]);
const AT_M8: u64 = cells3(&[5, 7]);
const AT_M16: u64 = cells3(&[8]);

/// Mirror top-bottom (swap rows 0 and 2).
#[inline(always)]
pub fn flip_rows(s: State) -> State {
    ((s & ROW0) << 12) | (s & ROW1) | ((s & ROW2) >> 12)
}

/// Mirror about the anti-diagonal: square (r, c) → (2−c, 2−r).
#[inline(always)]
pub fn anti_transpose(s: State) -> State {
    ((s & AT_P16) << 16) | ((s & AT_P8) << 8) | (s & AT_0) | ((s & AT_M8) >> 8) | ((s & AT_M16) >> 16)
}

/// The smallest of the 8 symmetric images. Alternating the two mirrors walks through all
/// 8 elements of the symmetry group of the square.
#[inline(always)]
pub fn canonical(s: State) -> State {
    let mut best = s;
    let mut x = s;
    for step in 0..7 {
        x = if step % 2 == 0 { flip_rows(x) } else { anti_transpose(x) };
        best = best.min(x);
    }
    best
}

/// Occupied squares of a block, as low bits.
#[inline(always)]
fn occupied(block: u64) -> u64 {
    (block | block >> 1) & CELL_LOW
}

#[inline(always)]
fn blocks(s: State) -> [u64; 3] {
    [s & BLOCK_MASK, (s >> 18) & BLOCK_MASK, s >> 36]
}

/// The pieces visible from above: (side to move, opponent), as low bits.
#[inline(always)]
pub fn visible(s: State) -> (u64, u64) {
    let [b0, b1, b2] = blocks(s);
    let o2 = occupied(b2);
    let o1 = occupied(b1);
    // For a mask of low bits x, x * 3 = x | x << 1 covers both bits of each square.
    let v = b2 | (b1 & !(o2 * 3)) | (b0 & !((o2 | o1) * 3));
    (v & CELL_LOW, (v >> 1) & CELL_LOW)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The side to move has won (the opponent uncovered its line).
    Win,
    /// The side to move has lost (the opponent completed a line).
    Loss,
}

/// Whether the squares in `m` cover a whole line.
#[inline(always)]
fn has_line(m: u64) -> bool {
    for l in LINES {
        if m & l == l {
            return true;
        }
    }
    false
}

/// The result for the side to move if the game is over.
#[inline(always)]
pub fn terminal(s: State) -> Option<Outcome> {
    let (me, op) = visible(s);
    if has_line(me) {
        Some(Outcome::Win)
    } else if has_line(op) {
        Some(Outcome::Loss)
    } else {
        None
    }
}

/// Call `f` with the board after each legal move of the side to move (sides not yet swapped).
/// Stops early and returns false as soon as `f` returns false; returns true otherwise.
#[inline(always)]
pub fn for_each_move(s: State, variant: Variant, mut f: impl FnMut(State) -> bool) -> bool {
    let b = blocks(s);
    let o = [occupied(b[0]), occupied(b[1]), occupied(b[2])];
    for k in 0..variant.sizes as usize {
        let larger = o[k + 1..].iter().fold(0, |acc, &x| acc | x);
        // Squares a size-k piece may land on: no piece of size ≥ k there.
        let allowed = !(o[k] | larger) & CELL_LOW;
        let shift = BLOCK_BITS as usize * k;

        if (b[k] & CELL_LOW).count_ones() < PER_SIZE {
            let mut a = allowed;
            while a != 0 {
                let dst = a & a.wrapping_neg();
                a &= a - 1;
                if !f(s | dst << shift) {
                    return false;
                }
            }
        }

        if variant.allow_move {
            // Our visible size-k pieces (nothing larger on top).
            let mut src = b[k] & CELL_LOW & !larger;
            while src != 0 {
                let from = src & src.wrapping_neg();
                src &= src - 1;
                let lifted = s ^ (from << shift);
                let mut a = allowed;
                while a != 0 {
                    let dst = a & a.wrapping_neg();
                    a &= a - 1;
                    if !f(lifted | dst << shift) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// The board after a move, as the canonical position of the next side to move.
#[inline(always)]
pub fn next_state(after_move: State) -> State {
    canonical(swap(after_move))
}

/// Render the visible board (side to move = X, opponent = O, digits are sizes 1..3).
pub fn show(s: State) -> String {
    let b = blocks(s);
    let mut out = String::new();
    for r in 0..3 {
        for c in 0..3 {
            let cell = 3 * r + c;
            let mut txt = " . ".to_string();
            for k in (0..3).rev() {
                let f = (b[k] >> (2 * cell)) & 3;
                if f != 0 {
                    txt = format!("{}{} ", if f == 1 { 'X' } else { 'O' }, k + 1);
                    break;
                }
            }
            out.push_str(&txt);
        }
        out.push('\n');
    }
    out
}

/// One-byte values: 0 = unknown (a draw if it stays unknown), 1 + d = the side to move wins in
/// d plies, 128 + d = the side to move loses in d plies (d ≤ 126).
pub mod value {
    pub const UNKNOWN: u8 = 0;
    pub const MAX_DEPTH: u8 = 126;

    #[inline(always)]
    pub fn win(d: u8) -> u8 {
        debug_assert!(d <= MAX_DEPTH);
        1 + d
    }
    #[inline(always)]
    pub fn loss(d: u8) -> u8 {
        debug_assert!(d <= MAX_DEPTH);
        128 + d
    }
    /// A known win whose depth is below `limit`.
    #[inline(always)]
    pub fn is_win_before(v: u8, limit: u8) -> bool {
        v != UNKNOWN && v < 128 && v - 1 < limit
    }
    /// A known loss whose depth is below `limit`.
    #[inline(always)]
    pub fn is_loss_before(v: u8, limit: u8) -> bool {
        v >= 128 && v - 128 < limit
    }
    /// (1 = win / -1 = loss / 0 = unknown, depth)
    pub fn decode(v: u8) -> (i8, u8) {
        match v {
            UNKNOWN => (0, 0),
            1..=127 => (1, v - 1),
            _ => (-1, v - 128),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_roundtrip() {
        let t = Tables::new();
        for i in (0..BLOCK_STATES.pow(3) as u32).step_by(9_973) {
            assert_eq!(t.index(t.state(i)), i);
        }
    }

    #[test]
    fn perms_are_symmetries() {
        // Every symmetry maps the 8 lines onto the 8 lines.
        for perm in &PERMS {
            for &line in &LINES {
                let moved = permute(line, perm) & CELL_LOW;
                assert!(LINES.contains(&moved), "{perm:?} {line:#x}");
            }
        }
    }

    #[test]
    fn canonical_matches_all_perms() {
        let t = Tables::new();
        for i in (0..BLOCK_STATES.pow(3) as u32).step_by(6_007) {
            let s = t.state(i);
            let slow = PERMS.iter().map(|p| permute(s, p)).min().unwrap();
            assert_eq!(canonical(s), slow);
        }
    }

    #[test]
    fn swap_is_involution() {
        let t = Tables::new();
        for i in (0..BLOCK_STATES.pow(3) as u32).step_by(7_919) {
            let s = t.state(i);
            assert_eq!(swap(swap(s)), s);
            assert!(t.index(swap(s)) < BLOCK_STATES.pow(3) as u32);
        }
    }

    #[test]
    fn empty_board_moves() {
        // First move: 3 sizes × 9 squares = 27 moves; up to symmetry, corner / edge / center per size.
        let mut all = vec![];
        for_each_move(0, Variant::STANDARD, |n| {
            all.push(next_state(n));
            true
        });
        assert_eq!(all.len(), 27);
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 9);
    }

    #[test]
    fn covered_line_does_not_count() {
        // A top row of our small pieces, with the opponent's large piece on square 1.
        let me_small = 0x15; // squares 0, 1, 2, side to move, block 0
        let op_large = (2u64 << 2) << 36; // square 1, opponent, large
        assert_eq!(terminal(me_small), Some(Outcome::Win));
        assert_eq!(terminal(me_small | op_large), None);
    }
}
