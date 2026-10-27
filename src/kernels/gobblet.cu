// Retrograde analysis kernels for Gobblet Gobblers (compiled at run time with NVRTC).
//
// The position encoding and the rules are the same as in src/rules.rs, but written again from
// scratch, so comparing the CPU and GPU tables also checks the rule implementations.
//
// The host prepends `#define SIZES` and `#define ALLOW_MOVE`.

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define BLOCK_MASK 0x3FFFFull
#define CELL_LOW 0x15555ull
#define ME_MASK (CELL_LOW | (CELL_LOW << 18) | (CELL_LOW << 36))
#define FIELD3 (3ull | (3ull << 18) | (3ull << 36))
#define NB 1423u
#define PER_SIZE 2

#define V_UNKNOWN 0
#define V_WIN(d) ((u8)(1 + (d)))
#define V_LOSS(d) ((u8)(128 + (d)))

__constant__ u64 LINES[8] = {0x15, 0x540, 0x15000, 0x1041, 0x4104, 0x10410, 0x10101, 0x1110};

struct Ctx {
    const u16* rank;   // 2^18 entries (global memory)
    const u32* unrank; // 1423 entries (shared memory)
};

__device__ __forceinline__ u64 state_of(const Ctx& c, u32 i) {
    u64 b0 = c.unrank[i % NB];
    u32 j = i / NB;
    u64 b1 = c.unrank[j % NB];
    u64 b2 = c.unrank[j / NB];
    return b0 | (b1 << 18) | (b2 << 36);
}

__device__ __forceinline__ u32 index_of(const Ctx& c, u64 s) {
    u32 r0 = __ldg(&c.rank[s & BLOCK_MASK]);
    u32 r1 = __ldg(&c.rank[(s >> 18) & BLOCK_MASK]);
    u32 r2 = __ldg(&c.rank[s >> 36]);
    return r0 + NB * (r1 + NB * r2);
}

__device__ __forceinline__ u64 swap_side(u64 s) { return ((s & ME_MASK) << 1) | ((s >> 1) & ME_MASK); }

// Both bits of square c, in all three blocks.
#define C3(c) (FIELD3 << (2 * (c)))
#define ROW0 (C3(0) | C3(1) | C3(2))
#define ROW1 (C3(3) | C3(4) | C3(5))
#define ROW2 (C3(6) | C3(7) | C3(8))

// Mirror top-bottom.
__device__ __forceinline__ u64 flip_rows(u64 s) { return ((s & ROW0) << 12) | (s & ROW1) | ((s & ROW2) >> 12); }

// Mirror about the anti-diagonal: square (r, c) -> (2-c, 2-r).
__device__ __forceinline__ u64 anti_transpose(u64 s) {
    return ((s & C3(0)) << 16) | ((s & (C3(1) | C3(3))) << 8) | (s & (C3(2) | C3(4) | C3(6))) |
           ((s & (C3(5) | C3(7))) >> 8) | ((s & C3(8)) >> 16);
}

__device__ __forceinline__ u64 umin(u64 a, u64 b) { return a < b ? a : b; }

// The smallest of the 8 symmetric images; alternating the two mirrors visits all of them.
__device__ __forceinline__ u64 canonical(u64 s) {
    u64 m = s, x = s;
#pragma unroll
    for (int step = 0; step < 7; ++step) {
        x = (step % 2 == 0) ? flip_rows(x) : anti_transpose(x);
        m = umin(m, x);
    }
    return m;
}

__device__ __forceinline__ u64 occupied(u64 b) { return (b | (b >> 1)) & CELL_LOW; }

// 1 if the side to move has won, -1 if it has lost, 0 if the game goes on.
__device__ __forceinline__ int terminal(u64 s) {
    u64 b0 = s & BLOCK_MASK, b1 = (s >> 18) & BLOCK_MASK, b2 = s >> 36;
    u64 o2 = occupied(b2), o1 = occupied(b1);
    u64 v = b2 | (b1 & ~(o2 * 3)) | (b0 & ~((o2 | o1) * 3));
    u64 me = v & CELL_LOW, op = (v >> 1) & CELL_LOW;
    bool mine = false, theirs = false;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        mine |= (me & LINES[l]) == LINES[l];
        theirs |= (op & LINES[l]) == LINES[l];
    }
    return mine ? 1 : (theirs ? -1 : 0);
}

// Call visit(after) with the board after each legal move of the side to move (sides not yet
// swapped). Stops and returns false as soon as visit returns false.
template <typename F>
__device__ __forceinline__ bool for_each_after(u64 s, F visit) {
    u64 b[3] = {s & BLOCK_MASK, (s >> 18) & BLOCK_MASK, s >> 36};
    u64 o[3] = {occupied(b[0]), occupied(b[1]), occupied(b[2])};
#pragma unroll
    for (int k = 0; k < SIZES; ++k) {
        u64 larger = 0;
#pragma unroll
        for (int j = k + 1; j < 3; ++j) larger |= o[j];
        u64 allowed = ~(o[k] | larger) & CELL_LOW;
        int shift = 18 * k;
        if (__popcll(b[k] & CELL_LOW) < PER_SIZE) {
            for (u64 a = allowed; a; a &= a - 1) {
                u64 dst = a & (0 - a);
                if (!visit(s | (dst << shift))) return false;
            }
        }
        if (ALLOW_MOVE) {
            for (u64 src = b[k] & CELL_LOW & ~larger; src; src &= src - 1) {
                u64 lifted = s ^ ((src & (0 - src)) << shift);
                for (u64 a = allowed; a; a &= a - 1) {
                    u64 dst = a & (0 - a);
                    if (!visit(lifted | (dst << shift))) return false;
                }
            }
        }
    }
    return true;
}

// Call visit(j) with the index of each next position (sides swapped, canonical).
template <typename F>
__device__ __forceinline__ bool for_each_next(const Ctx& c, u64 s, F visit) {
    return for_each_after(s, [&](u64 after) { return visit(index_of(c, canonical(swap_side(after)))); });
}

__device__ __forceinline__ Ctx load_ctx(const u16* rank, const u32* unrank_g, u32* smem) {
    for (u32 i = threadIdx.x; i < NB; i += blockDim.x) smem[i] = unrank_g[i];
    __syncthreads();
    Ctx c;
    c.rank = rank;
    c.unrank = smem;
    return c;
}

// Append `value` from every lane with `keep`, contiguously, with one atomic per warp.
__device__ __forceinline__ void warp_append(bool keep, u32 value, u32* out, u64* count) {
    u32 lane = threadIdx.x & 31;
    u32 mask = __ballot_sync(0xffffffffu, keep);
    if (mask == 0) return;
    u64 base = 0;
    if (lane == 0) base = atomicAdd(count, (u64)__popc(mask));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (keep) out[base + __popc(mask & ((1u << lane) - 1))] = value;
}

// ---- 1. breadth-first expansion: expand list[lo..hi), append new positions at list[hi + *count] ----
extern "C" __global__ void bfs_expand(const u16* rank, const u32* unrank_g, u32* list, u64 lo, u64 hi, u32* seen,
                                      u64* count, u64 capacity) {
    __shared__ u32 smem[NB];
    Ctx c = load_ctx(rank, unrank_g, smem);
    u64 t = lo + (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= hi) return;
    u64 s = state_of(c, list[t]);
    if (terminal(s) != 0) return;
    for_each_next(c, s, [&](u32 j) {
        u32 bit = 1u << (j & 31);
        // Bits are only ever set, so a stale plain read can only miss a bit, never invent one.
        // Most next positions were found already and need no atomic at all.
        if (seen[j >> 5] & bit) return true;
        if ((atomicOr(&seen[j >> 5], bit) & bit) == 0) {
            u64 pos = hi + atomicAdd(count, 1ull);
            if (pos < capacity) list[pos] = j;
        }
        return true;
    });
}

// ---- 2. depths 0 and 1; collect the rest (unknown, with legal moves) into `unknown` ----
// Depth 0 is the terminal check itself, so depth 1 (win by completing a line / every move
// uncovers an opponent's line) follows from the terminal check of the next boards, with no
// reads of the value table. stats = [wins in 1, losses in 1, non-terminal with no legal move]
extern "C" __global__ void init_terminal(const u16* rank, const u32* unrank_g, const u32* list, u64 n, u8* values,
                                         u32* unknown, u64* unknown_count, u64* stats) {
    __shared__ u32 smem[NB];
    Ctx c = load_ctx(rank, unrank_g, smem);
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    bool keep = false, won = false, lost = false, stuck = false;
    u32 i = 0;
    if (t < n) {
        i = list[t];
        u64 s = state_of(c, i);
        int r = terminal(s);
        if (r > 0) {
            values[i] = V_WIN(0);
        } else if (r < 0) {
            values[i] = V_LOSS(0);
        } else {
            bool any = false, all_lose = true;
            bool finished = for_each_after(s, [&](u64 after) {
                any = true;
                int rn = terminal(swap_side(after)); // seen from the opponent
                if (rn < 0) return false;            // we completed a line: win in 1
                all_lose &= (rn > 0);                // we uncovered theirs
                return true;
            });
            if (!finished) {
                values[i] = V_WIN(1);
                won = true;
            } else if (!any) {
                stuck = true; // no legal move: stays a draw
            } else if (all_lose) {
                values[i] = V_LOSS(1);
                lost = true;
            } else {
                keep = true;
            }
        }
    }
    u32 lane = threadIdx.x & 31;
    u32 wm = __ballot_sync(0xffffffffu, won), lm = __ballot_sync(0xffffffffu, lost);
    u32 sm = __ballot_sync(0xffffffffu, stuck);
    if (lane == 0) {
        if (wm) atomicAdd(&stats[0], (u64)__popc(wm));
        if (lm) atomicAdd(&stats[1], (u64)__popc(lm));
        if (sm) atomicAdd(&stats[2], (u64)__popc(sm));
    }
    warp_append(keep, i, unknown, unknown_count);
}

// ---- 3. retrograde pass at depth k; stats = [wins, losses] ----
// Values written by this pass have depth k and are ignored by the depth checks below, so the
// table can be updated in place (a stale read only returns an older value).
extern "C" __global__ void retro_pass(const u16* rank, const u32* unrank_g, const u32* unknown, u64 n, u8* values,
                                      u32 k, u32* next_unknown, u64* next_count, u64* stats) {
    __shared__ u32 smem[NB];
    Ctx c = load_ctx(rank, unrank_g, smem);
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    bool won = false, lost = false;
    u32 i = 0;
    if (t < n) {
        i = unknown[t];
        u64 s = state_of(c, i);
        bool all_win = true;
        bool finished = for_each_next(c, s, [&](u32 j) {
            u8 v = values[j];
            if (v >= 128 && (u32)(v - 128) < k) return false; // a move that beats the opponent
            all_win &= (v != V_UNKNOWN && v < 128 && (u32)(v - 1) < k);
            return true;
        });
        if (!finished) {
            values[i] = V_WIN(k);
            won = true;
        } else if (all_win) {
            values[i] = V_LOSS(k);
            lost = true;
        }
    }
    u32 lane = threadIdx.x & 31;
    u32 wm = __ballot_sync(0xffffffffu, won), lm = __ballot_sync(0xffffffffu, lost);
    if (lane == 0) {
        if (wm) atomicAdd(&stats[0], (u64)__popc(wm));
        if (lm) atomicAdd(&stats[1], (u64)__popc(lm));
    }
    warp_append(t < n && !won && !lost, i, next_unknown, next_count);
}
