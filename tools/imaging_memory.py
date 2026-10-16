#!/usr/bin/env python3
"""Memory cost of imaging time series (RQ7, Appendix Tables app.tab.conv_mem / app.tab.conv_mem_d).

Runs each conversion in a fresh subprocess (tools/imaging_memory_cell.py) and writes
results/imaging_memory.json.

    python tools/imaging_memory.py
"""
import json, os, socket, subprocess, sys, time, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CELL = os.path.join(HERE, "imaging_memory_cell.py")
ENV = dict(os.environ, OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
GPU_SAFETY_MB = 4000  # cap expected GPU footprint on the shared server


def run_cell(args, timeout=1200):
    cmd = [PY, CELL] + [str(a) for a in args]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)
        if p.returncode != 0:
            err = (p.stderr or "").strip().splitlines()
            return dict(error="subprocess failed", detail=err[-1] if err else "", args=args)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return dict(error="timeout", args=args)


def gpu_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", ENV["CUDA_VISIBLE_DEVICES"]], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def uvh_expected_mb(B, d):
    # dominant term: (B*d, 3, 224, 224) float32 output + 1-ch resize intermediate
    return B * d * 4 * 224 * 224 * (3 + 1) / 2**20 + 200


def main():
    results = dict(
        experiment="imaging peak memory",
        hostname=socket.gethostname(),
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        where_conversion_runs=dict(
            GAF_RP_STFT_Wavelet_Filterbank=("CPU, numpy, inside the dataloader/dataset builder "
                                            "(time2img/imagedata.py: image_transform_forecasting), "
                                            "precomputed before training"),
            UVH=("torch ops inside the model forward "
                 "(VisionReg/visionReg.py: UTSRegression._time2img), i.e. on the GPU "
                 "in the actual pipeline; measured on both CPU (RSS) and GPU (cuda peak)")),
        dtype=dict(numpy_methods="float64 input (StandardScaler output), float64 conversion output",
                   UVH="float32 torch input/output",
                   batch_GAF="float64 numpy in, stored as float32 torch (repo behavior)"),
        uvh_periodicity=24,
        table_A=[], table_B=[])

    # ---- Table A: single variate, T sweep ----
    methods = ["GAF", "RP", "STFT", "Wavelet", "Filterbank"]
    for T in (336, 1152, 2304):
        for m in methods:
            print(f"[A] {m} T={T}", flush=True)
            results["table_A"].append(run_cell(
                ["--kind", "single", "--method", m, "--T", T]))
        # Wavelet variant matching profile_cost.py (63 scales)
        print(f"[A] Wavelet63 T={T}", flush=True)
        r = run_cell(["--kind", "single", "--method", "Wavelet", "--T", T,
                      "--wavelet_scales", 63])
        r["method"] = "Wavelet(63 scales, profile_cost config)"
        results["table_A"].append(r)
        # UVH: CPU and (small) GPU
        print(f"[A] UVH T={T} cpu", flush=True)
        results["table_A"].append(run_cell(
            ["--kind", "uvh", "--T", T, "--d", 1, "--B", 1, "--device", "cpu"]))
        free = gpu_free_mb()
        need = uvh_expected_mb(1, 1)
        if free > need + 1500:
            print(f"[A] UVH T={T} cuda (free={free}MB)", flush=True)
            results["table_A"].append(run_cell(
                ["--kind", "uvh", "--T", T, "--d", 1, "--B", 1, "--device", "cuda"]))
        else:
            results["table_A"].append(dict(kind="uvh", method="uvh", T=T, d=1, B=1,
                                           device="cuda", skipped=f"insufficient free GPU memory ({free} MB)"))

    # ---- Table B: batch B=32, T=336, d sweep ----
    for d in (7, 21, 321, 862):
        print(f"[B] UVH d={d} cpu", flush=True)
        results["table_B"].append(run_cell(
            ["--kind", "uvh", "--T", 336, "--d", d, "--B", 32, "--device", "cpu"]))
        need = uvh_expected_mb(32, d)
        free = gpu_free_mb()
        if need < GPU_SAFETY_MB and free > need + 1500:
            print(f"[B] UVH d={d} cuda (need~{need:.0f}MB free={free}MB)", flush=True)
            results["table_B"].append(run_cell(
                ["--kind", "uvh", "--T", 336, "--d", d, "--B", 32, "--device", "cuda"]))
        else:
            results["table_B"].append(dict(
                kind="uvh", method="uvh", T=336, d=d, B=32, device="cuda",
                skipped=f"expected ~{need:.0f} MB exceeds safety cap on shared GPU (free={free} MB)"))
        print(f"[B] GAF d={d} cpu", flush=True)
        results["table_B"].append(run_cell(
            ["--kind", "batch_gaf", "--method", "GAF", "--T", 336, "--d", d, "--B", 32],
            timeout=3600))

    out = os.path.join(os.path.dirname(HERE), "results", "imaging_memory.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("WROTE", out)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
