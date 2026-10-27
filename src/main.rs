//! Strongly solve Gobblet Gobblers by retrograde analysis of every reachable position.
//!
//! Usage:
//!   gobblet-cuda solve   [--sizes N] [--no-move] [--backend cpu|gpu] [--passes]
//!   gobblet-cuda compare [--sizes N] [--no-move]      compare the CPU and GPU tables
//!   gobblet-cuda pw <gobblet_*.dat> [--sizes N] [--no-move] [--backend cpu|gpu]
//!                                                     compare with possiblywrong's table
//!
//! Options:
//!   --sizes N     number of piece sizes, 1..=3 (default 3)
//!   --no-move     placement only: pieces on the board may not move
//!   --backend B   cpu (default) or gpu (needs a build with --features cuda)
//!   --passes      print per-level and per-depth statistics

mod pw;
mod report;
mod rules;
mod solve_cpu;
#[cfg(feature = "cuda")]
mod solve_gpu;

use rules::{Tables, Variant};

struct Args {
    command: String,
    positional: Vec<String>,
    variant: Variant,
    backend: String,
    passes: bool,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let command = it.next().unwrap_or_else(|| usage());
    let mut a = Args { command, positional: vec![], variant: Variant::STANDARD, backend: "cpu".into(), passes: false };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--sizes" => {
                a.variant.sizes =
                    it.next().and_then(|v| v.parse().ok()).filter(|v| (1..=3).contains(v)).unwrap_or_else(|| usage())
            }
            "--no-move" => a.variant.allow_move = false,
            "--backend" => a.backend = it.next().unwrap_or_else(|| usage()),
            "--passes" => a.passes = true,
            "-h" | "--help" => usage(),
            _ if arg.starts_with('-') => usage(),
            _ => a.positional.push(arg),
        }
    }
    a
}

/// Print the usage text (the `//!` header of this file) and exit.
fn usage() -> ! {
    let text: Vec<&str> = include_str!("main.rs")
        .lines()
        .take_while(|l| l.starts_with("//!"))
        .map(|l| l.trim_start_matches("//!").strip_prefix(' ').unwrap_or(""))
        .collect();
    eprintln!("{}", text.join("\n"));
    std::process::exit(2)
}

fn run(variant: Variant, backend: &str) -> solve_cpu::Solution {
    match backend {
        "cpu" => solve_cpu::solve(variant),
        #[cfg(feature = "cuda")]
        "gpu" => solve_gpu::solve(variant).unwrap_or_else(|e| {
            eprintln!("GPU solve failed: {e}");
            std::process::exit(1)
        }),
        other => {
            eprintln!("backend {other} is not available (gpu needs a build with --features cuda)");
            std::process::exit(2)
        }
    }
}

fn main() {
    let args = parse_args();
    match args.command.as_str() {
        "solve" => {
            let sol = run(args.variant, &args.backend);
            print!("{}", sol.report.to_text());
            if args.passes {
                print!("{}", sol.report.passes_text());
            }
        }
        "compare" => {
            let cpu = run(args.variant, "cpu");
            print!("{}", cpu.report.to_text());
            let gpu = run(args.variant, "gpu");
            print!("{}", gpu.report.to_text());
            let diff = cpu.values.iter().zip(&gpu.values).filter(|(a, b)| a != b).count();
            println!("indices whose CPU and GPU values differ: {diff} / {}", cpu.values.len());
            if diff != 0 || cpu.reachable.len() != gpu.reachable.len() {
                std::process::exit(1);
            }
        }
        "pw" => {
            let path = args.positional.first().unwrap_or_else(|| usage());
            let sol = run(args.variant, &args.backend);
            print!("{}", sol.report.to_text());
            let cmp = pw::compare(path, &Tables::new(), &sol.values).unwrap_or_else(|e| {
                eprintln!("cannot read {path}: {e}");
                std::process::exit(1)
            });
            println!("possiblywrong table: {} positions, {} value mismatches", cmp.entries, cmp.mismatches);
            for e in &cmp.examples {
                println!("{e}");
            }
            if cmp.mismatches != 0 || cmp.entries != sol.report.reachable {
                std::process::exit(1);
            }
        }
        _ => usage(),
    }
}
