//! CUDA solver. Same steps as `solve_cpu`, one kernel per step (`kernels/gobblet.cu`).
//!
//! Device memory for the standard game (≈ 7.6 GB):
//! - `values`: one byte per index, 1423³ ≈ 2.9 GB
//! - `seen`: bitmap of found positions, ≈ 360 MB (freed after enumeration)
//! - `list`: reachable positions in breadth-first order, plus two unknown lists used alternately

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::report::{PassStat, Report};
use crate::rules::{self, value, Tables, Variant};
use crate::solve_cpu::Solution;

const THREADS: u32 = 256;
/// Room for the reachable list (the standard game has 341M positions).
const LIST_CAPACITY: u64 = 400_000_000;

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn cfg(n: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(THREADS as u64) as u32, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

struct Kernels {
    bfs_expand: CudaFunction,
    init_terminal: CudaFunction,
    retro_pass: CudaFunction,
}

fn compile(ctx: &Arc<CudaContext>, variant: Variant) -> R<Kernels> {
    let major = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
    let src = format!(
        "#define SIZES {}\n#define ALLOW_MOVE {}\n{}",
        variant.sizes,
        variant.allow_move as u32,
        include_str!("kernels/gobblet.cu")
    );
    let opts =
        cudarc::nvrtc::CompileOptions { arch: Some(arch), options: vec!["-std=c++17".into()], ..Default::default() };
    let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts)?;
    let module = ctx.load_module(ptx)?;
    Ok(Kernels {
        bfs_expand: module.load_function("bfs_expand")?,
        init_terminal: module.load_function("init_terminal")?,
        retro_pass: module.load_function("retro_pass")?,
    })
}

fn read(stream: &Arc<CudaStream>, x: &CudaSlice<u64>) -> R<Vec<u64>> {
    Ok(stream.memcpy_dtov(x)?)
}

pub fn solve(variant: Variant) -> R<Solution> {
    let t = Tables::new();
    let n = variant.num_indices();
    let mut report = Report::new(variant, "gpu");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    report.device = ctx.name().unwrap_or_default();
    let k = compile(&ctx, variant)?;

    let rank = stream.memcpy_stod(&t.rank)?;
    let unrank = stream.memcpy_stod(&t.unrank)?;
    stream.synchronize()?;
    let total = Instant::now();

    // ---- 1. enumerate reachable positions ----
    let phase = Instant::now();
    let mut seen = stream.alloc_zeros::<u32>(n.div_ceil(32) as usize)?;
    let mut list = stream.alloc_zeros::<u32>(LIST_CAPACITY as usize)?;
    let start = t.index(rules::canonical(0));
    let word = (start / 32) as usize;
    stream.memcpy_htod(&[start], &mut list.slice_mut(0..1))?;
    stream.memcpy_htod(&[1u32 << (start % 32)], &mut seen.slice_mut(word..word + 1))?;
    let mut count = stream.alloc_zeros::<u64>(1)?;
    let (mut lo, mut hi) = (0u64, 1u64);
    while lo < hi {
        let level = Instant::now();
        stream.memset_zeros(&mut count)?;
        let mut b = stream.launch_builder(&k.bfs_expand);
        b.arg(&rank).arg(&unrank).arg(&mut list).arg(&lo).arg(&hi).arg(&mut seen).arg(&mut count).arg(&LIST_CAPACITY);
        unsafe { b.launch(cfg(hi - lo)) }?;
        let found = read(&stream, &count)?[0];
        if hi + found > LIST_CAPACITY {
            return Err(format!("more reachable positions than LIST_CAPACITY ({LIST_CAPACITY})").into());
        }
        report.bfs.push((hi - lo, found, level.elapsed().as_secs_f64()));
        lo = hi;
        hi += found;
    }
    drop(seen);
    let reachable_n = hi;
    report.reachable = reachable_n;
    report.time_enumerate = phase.elapsed().as_secs_f64();

    // ---- 2. depths 0 and 1 in one kernel ----
    let phase = Instant::now();
    let mut values = stream.alloc_zeros::<u8>(n as usize)?;
    let mut unknown =
        [stream.alloc_zeros::<u32>(reachable_n as usize)?, stream.alloc_zeros::<u32>(reachable_n as usize)?];
    let mut init_stats = stream.alloc_zeros::<u64>(3)?;
    stream.memset_zeros(&mut count)?;
    let pass = Instant::now();
    {
        let mut b = stream.launch_builder(&k.init_terminal);
        b.arg(&rank).arg(&unrank).arg(&list).arg(&reachable_n).arg(&mut values);
        b.arg(&mut unknown[0]).arg(&mut count).arg(&mut init_stats);
        unsafe { b.launch(cfg(reachable_n)) }?;
    }
    let mut unknown_n = read(&stream, &count)?[0];
    let st = read(&stream, &init_stats)?;
    report.no_move_states = st[2];
    report.terminal = reachable_n - unknown_n - st[0] - st[1] - st[2];
    report.passes.push(PassStat { depth: 0, wins: 0, losses: 0, seconds: 0.0 });
    report.passes.push(PassStat { depth: 1, wins: st[0], losses: st[1], seconds: pass.elapsed().as_secs_f64() });

    // ---- 3. depths 2, 3, … ----
    let mut stats = stream.alloc_zeros::<u64>(2)?;
    let mut cur = 0;
    for depth in 2..=value::MAX_DEPTH as u32 {
        if unknown_n == 0 {
            break;
        }
        let pass = Instant::now();
        stream.memset_zeros(&mut count)?;
        stream.memset_zeros(&mut stats)?;
        let [a, b_] = &mut unknown;
        let (src, dst) = if cur == 0 { (&*a, b_) } else { (&*b_, a) };
        let mut b = stream.launch_builder(&k.retro_pass);
        b.arg(&rank).arg(&unrank).arg(src).arg(&unknown_n).arg(&mut values);
        b.arg(&depth).arg(dst).arg(&mut count).arg(&mut stats);
        unsafe { b.launch(cfg(unknown_n)) }?;
        let st = read(&stream, &stats)?;
        unknown_n = read(&stream, &count)?[0];
        report.passes.push(PassStat {
            depth: depth as u8,
            wins: st[0],
            losses: st[1],
            seconds: pass.elapsed().as_secs_f64(),
        });
        cur ^= 1;
        if st[0] + st[1] == 0 {
            break;
        }
    }
    stream.synchronize()?;
    report.time_retrograde = phase.elapsed().as_secs_f64();
    report.time_total = total.elapsed().as_secs_f64();

    // ---- 4. copy back and summarize (not timed) ----
    let values_h = stream.memcpy_dtov(&values)?;
    let reachable_h = stream.memcpy_dtov(&list.slice(0..reachable_n as usize))?;
    report.summarize(&t, variant, &values_h, &reachable_h);
    Ok(Solution { values: values_h, reachable: reachable_h, report })
}
