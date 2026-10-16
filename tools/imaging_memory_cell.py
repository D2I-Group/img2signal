#!/usr/bin/env python3
"""Measure peak memory of ONE time-series-to-image conversion cell (called by imaging_memory.py).

Run in a FRESH subprocess per cell. Uses the paper repo's own conversion code:
  * GAF/RP/STFT/Wavelet/Filterbank: time2img plotters, exactly as invoked by
    time2img/imagedata.py:image_transform_forecasting (CPU, numpy).
  * UVH: VisionReg/visionReg.py:UTSRegression._time2img (torch; GPU in the
    real pipeline, measured here on the requested device via a shim object
    carrying pad_left/periodicity/resize, calling the unbound repo method).

Prints one JSON dict to stdout.
"""
import argparse, gc, json, os, resource, sys, time, tracemalloc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np


def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def maxrss_mb():
    # linux: ru_maxrss in KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def nbytes_of(obj):
    import torch
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, (list, tuple)):
        return sum(nbytes_of(o) for o in obj)
    if np.iscomplexobj(obj):
        a = np.asarray(obj)
        return a.nbytes
    a = np.asarray(obj)
    return a.nbytes


def build_single(method, T, wavelet_scales=None):
    """One univariate series of length T -> conversion closure (repo plotters)."""
    from time2img import gaf, rp, spectrogram as sp
    x = np.random.randn(T).astype(np.float64)  # dataset yields float64 (StandardScaler)
    if method == "GAF":
        p = gaf.GAF_plotter()
        conv = lambda a: p.plot(a)
    elif method == "RP":
        p = rp.RP_plotter()
        conv = lambda a: p.plot(a)
    elif method == "STFT":
        p = sp.STFT_Plotter()
        conv = lambda a: p.plot(a)
    elif method == "Wavelet":
        p = sp.WaveletForecast_Plotter()
        if wavelet_scales:
            sc = np.arange(1, wavelet_scales + 1)
            conv = lambda a: p.plot(a, scales=sc)
        else:
            conv = lambda a: p.plot(a)  # repo default: scales=np.arange(1, 400)
    elif method == "Filterbank":
        p = sp.Filterbank_Plotter()
        conv = lambda a: p.plot(a)
    else:
        raise ValueError(method)
    warm = np.random.randn(96).astype(np.float64)
    conv(warm)  # warm-up: imports/JIT/caches before baseline
    return x, conv, "float64 (numpy input, as produced by the repo's StandardScaler data loader)"


def build_uvh(T, d, B, device):
    """Repo's UTSRegression._time2img via shim (torch), input (B, T, d) float32."""
    import torch
    from PIL import Image
    from VisionReg.visionReg import UTSRegression
    from VisionReg.utils import safe_resize

    class Shim:
        pass

    shim = Shim()
    shim.periodicity = 24  # ETTh1 default (scripts/forecasting/*: periodicity=24)
    shim.pad_left = (24 - T % 24) % 24
    shim.resize = safe_resize((224, 224), interpolation=Image.BILINEAR)  # mae_base 224, default bilinear

    x = torch.randn(B, T, d, dtype=torch.float32, device=device)
    conv = lambda a: UTSRegression._time2img(shim, a)
    warm = torch.randn(1, 48, 1, dtype=torch.float32, device=device)
    conv(warm)
    return x, conv, "float32 (torch model input)"


def build_batch_gaf(T, d, B):
    """Batch conversion exactly as the dataset builder does it:
    image_transform_forecasting per sample, results held (as ImageTimeSeriesForecasting does)."""
    from time2img.imagedata import image_transform_forecasting

    class NS:
        time2img = "GAF"

    samples = [np.random.randn(T, d).astype(np.float64) for _ in range(B)]

    def conv(_):
        return [image_transform_forecasting(s, NS) for s in samples]

    image_transform_forecasting(np.random.randn(96, 2).astype(np.float64), NS)  # warm-up
    return samples, conv, "float64 numpy in -> float32 torch tensors stored (repo behavior)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["single", "uvh", "batch_gaf"])
    ap.add_argument("--method", default="")
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wavelet_scales", type=int, default=0)
    a = ap.parse_args()

    res = dict(kind=a.kind, method=a.method or a.kind, T=a.T, d=a.d, B=a.B, device=a.device)

    if a.kind == "single":
        x, conv, dtype_note = build_single(a.method, a.T, a.wavelet_scales or None)
    elif a.kind == "uvh":
        x, conv, dtype_note = build_uvh(a.T, a.d, a.B, a.device)
    else:
        x, conv, dtype_note = build_batch_gaf(a.T, a.d, a.B)
    res["dtype"] = dtype_note

    gc.collect()
    use_cuda = a.device.startswith("cuda")
    if use_cuda:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated() / 2**20
        t0 = time.perf_counter()
        out = conv(x)
        torch.cuda.synchronize()
        res["time_s"] = round(time.perf_counter() - t0, 4)
        res["gpu_peak_alloc_MB"] = round(torch.cuda.max_memory_allocated() / 2**20 - base_alloc, 2)
        res["gpu_base_alloc_MB"] = round(base_alloc, 2)
        res["memory_kind"] = "GPU (torch.cuda.max_memory_allocated delta)"
    else:
        base_rss = rss_mb()
        base_maxrss = maxrss_mb()
        tracemalloc.start()
        t0 = time.perf_counter()
        out = conv(x)
        res["time_s"] = round(time.perf_counter() - t0, 4)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        res["tracemalloc_peak_MB"] = round(peak / 2**20, 2)
        res["rss_after_MB_delta"] = round(rss_mb() - base_rss, 2)
        res["maxrss_delta_MB"] = round(max(0.0, maxrss_mb() - base_maxrss), 2)
        res["memory_kind"] = "CPU (tracemalloc peak + RSS deltas)"
        if a.kind == "uvh":
            res["note"] = "torch CPU tensors are not visible to tracemalloc; use RSS deltas"

    res["out_MB"] = round(nbytes_of(out) / 2**20, 3)
    try:
        import torch
        if isinstance(out, torch.Tensor):
            res["out_shape"] = list(out.shape)
        elif isinstance(out, np.ndarray):
            res["out_shape"] = list(out.shape)
        elif isinstance(out, list) and len(out) and hasattr(out[0], "shape"):
            res["out_shape"] = [len(out)] + list(out[0].shape)
        else:
            res["out_shape"] = list(np.asarray(out).shape)
    except Exception:
        pass

    print(json.dumps(res))


if __name__ == "__main__":
    main()
