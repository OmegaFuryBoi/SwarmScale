#!/usr/bin/env python3
"""
SwarmScale benchmark runner.

Builds every version with the right -D flags, runs each configuration
REPEATS times, takes the median, and writes results/bench.csv.

    python3 bench.py            # full run (3 repeats each)
    python3 bench.py --quick    # 1 repeat each, for a fast check
    python3 bench.py --only omp_threads schedules   # run some experiments only
"""
import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

STEPS = 100
BIN = "bench_bin"
OUT = "results/bench.csv"

HAVE_NVCC = shutil.which("nvcc") is not None
HAVE_MPI = shutil.which("mpicc") is not None and shutil.which("mpirun") is not None


# ---------------------------------------------------------------- building

def build(version, n):
    """Compile one version for one N (cached: built once per run)."""
    exe = os.path.join(BIN, f"{version}_{n}")
    if os.path.exists(exe):
        return exe
    flags = f"-DN={n} -DSTEPS={STEPS} -DSAVE_EVERY=0"
    cmds = {
        "omp":    f"gcc -O2 -fopenmp {flags} swarm.c -o {exe} -lm",
        "cuda":   f"nvcc -O2 --fmad=false -arch=sm_89 -Xcompiler -fopenmp {flags} swarm_cuda.cu -o {exe} -lgomp",
        "mpi":    f"mpicc -O2 -fopenmp {flags} swarm_mpi.c -o {exe} -lm",
        "hybrid": f"nvcc -O2 --fmad=false -arch=sm_89 -ccbin mpicxx -Xcompiler -fopenmp {flags} swarm_hybrid.cu -o {exe} -lgomp",
    }
    print(f"  building {version} N={n}", flush=True)
    r = subprocess.run(cmds[version], shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"build failed:\n{cmds[version]}\n{r.stderr}")
    return exe


# ---------------------------------------------------------------- running

PATTERNS = {
    "total_s":     r"total time\s*:\s*([\d.]+) s",
    "neigh_s":     r"neighbour search\s*:\s*([\d.]+) s",
    "density_s":   r"density field\s*:\s*([\d.]+) s",
    "decide_s":    r"decide agents\s*:\s*([\d.]+) s",
    "comm_s":      r"communication\s*:\s*([\d.]+) s",
    "kernel_s":    r"GPU kernel\s+:\s*([\d.]+) s",
    "imbalance":   r"load imbalance\s*:\s*([\d.]+)",
}


def run_once(version, exe, procs, threads, schedule):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_SCHEDULE=schedule)
    if version in ("mpi", "hybrid"):
        cmd = ["mpirun", "--oversubscribe", "-np", str(procs), exe]
    else:
        cmd = [exe]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"run failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    values = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, r.stdout)
        values[key] = float(m.group(1)) if m else None
    return values


def measure(version, n, procs, threads, schedule, repeats):
    exe = build(version, n)
    runs = [run_once(version, exe, procs, threads, schedule) for _ in range(repeats)]
    row = {}
    for key in PATTERNS:
        vals = [r[key] for r in runs if r[key] is not None]
        row[key] = round(statistics.median(vals), 6) if vals else ""
    row["ms_per_step"] = round(1000 * row["total_s"] / STEPS, 3)
    return row


# ---------------------------------------------------------------- experiments

def experiments():
    """(experiment, version, N, processes, threads, schedule)"""
    e = []
    for n in (1000, 2000, 4000):
        e.append(("serial_scaling", "omp", n, 1, 1, "dynamic"))
    for t in (1, 2, 4, 6, 8, 12):
        e.append(("omp_threads", "omp", 4000, 1, t, "dynamic"))
    for s in ("static", "guided", "static,16", "dynamic", "dynamic,16"):
        e.append(("schedules", "omp", 4000, 1, 12, s))
    if HAVE_NVCC:
        for n in (2000, 4000, 8000, 16000):
            e.append(("cpu_vs_gpu", "omp", n, 1, 12, "dynamic"))
            e.append(("cpu_vs_gpu", "cuda", n, 1, 12, "dynamic"))
    if HAVE_MPI:
        for p in (1, 2, 4):
            e.append(("mpi_procs", "mpi", 4000, p, 1, "dynamic"))
        for n in (4000, 16000):
            e.append(("hybrid", "mpi", n, 4, 3, "dynamic"))
            if HAVE_NVCC:
                for p, t in ((1, 12), (2, 6), (4, 3)):
                    e.append(("hybrid", "hybrid", n, p, t, "dynamic"))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 repeat instead of 3")
    ap.add_argument("--only", nargs="*", help="run only these experiments")
    args = ap.parse_args()
    repeats = 1 if args.quick else 3

    os.makedirs(BIN, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    todo = [x for x in experiments() if not args.only or x[0] in args.only]
    print(f"{len(todo)} configurations, {repeats} repeat(s) each, {STEPS} steps per run")
    if not HAVE_NVCC:
        print("note: nvcc not found, skipping CUDA and hybrid runs")
    if not HAVE_MPI:
        print("note: mpicc/mpirun not found, skipping MPI and hybrid runs")

    fields = ["experiment", "version", "n", "procs", "threads", "schedule",
              "ms_per_step"] + list(PATTERNS)
    start = time.time()
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k, (exp, ver, n, p, t, s) in enumerate(todo, 1):
            print(f"[{k}/{len(todo)}] {exp}: {ver} N={n} {p}x{t} {s}", flush=True)
            row = measure(ver, n, p, t, s, repeats)
            row.update(experiment=exp, version=ver, n=n, procs=p, threads=t, schedule=s)
            w.writerow(row)
            f.flush()
            print(f"        {row['ms_per_step']} ms/step", flush=True)
    print(f"done in {time.time() - start:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
