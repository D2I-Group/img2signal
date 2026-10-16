#!/usr/bin/env python3
import os, sys, time, json, gc
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

DEV = "cuda"
OUT = os.path.join(REPO, "results", "cost_profile.json")

def cuda_reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def mb(x): return round(x / 1024**2, 1)

def profile_imaging():
    from time2img import gaf, rp, spectrogram as sp, heatmap as hm
    rows = []
    plotters = {
        "GAF": gaf.GAF_plotter(), "RP": rp.RP_plotter(),
        "STFT": sp.STFT_Plotter(), "Wavelet": sp.Wavelet_Plotter(),
        "Filterbank": sp.Filterbank_Plotter(), "UVH": hm.AdaptiveHeatmap_Plotter(),
    }
    def call(p, name, x):
        if name == "Wavelet":
            return p.plot(torch.from_numpy(x), scales=np.arange(1, 64))
        return p.plot(x)

    warm = np.random.randn(336).astype(np.float32)
    for name, p in plotters.items():
        try: call(p, name, warm)
        except Exception: pass

    for T in (336, 720, 1152, 2304):
        x = np.random.randn(T).astype(np.float32)
        for name, p in plotters.items():
            try:
                t0 = time.perf_counter()
                for _ in range(3):
                    img = call(p, name, x)
                dt = (time.perf_counter() - t0) / 3
                arr = np.asarray(img)
                rows.append(dict(stage="imaging", method=name, T=T,
                                 ms_per_variate=round(dt * 1e3, 2),
                                 out_shape=list(arr.shape),
                                 out_MB=round(arr.nbytes / 1024**2, 3)))
            except Exception as e:
                rows.append(dict(stage="imaging", method=name, T=T, error=type(e).__name__))
    return rows

def profile_variate_scaling():
    """Under variate-independence a batch of B samples becomes B*d images."""
    rows = []
    for d, ds in ((7, "ETTh1"), (21, "Weather"), (321, "Electricity"), (862, "Traffic")):
        for B in (8, 32):
            n_img = B * d
            per_img_MB = 224 * 224 * 3 * 4 / 1024**2
            rows.append(dict(stage="variate_scaling", dataset=ds, d=d, batch=B,
                             images_per_batch=n_img,
                             image_tensor_MB=round(n_img * per_img_MB, 1)))
    return rows

def profile_backbones():
    from utscls.visioncls import ViT_MODELS, DINOv2Backbone
    from utscls import models_vit, models_mae
    rows = []
    builders = {
        "ViT-B (vit_base)":  lambda: models_vit.vit_base_patch16(),
        "MAE-B (mae_base)":  lambda: models_mae.mae_vit_base_patch16(),
        "DINOv2-B":          lambda: DINOv2Backbone(224),
    }
    for name, build in builders.items():
        try:
            m = build().to(DEV).eval()
            n_par = sum(p.numel() for p in m.parameters())
            for B in (8, 32):
                x = torch.randn(B, 3, 224, 224, device=DEV)
                cuda_reset()
                with torch.no_grad():
                    for _ in range(2): _ = m.forward_features(x)
                torch.cuda.synchronize()
                fwd_peak = torch.cuda.max_memory_allocated()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(5): _ = m.forward_features(x)
                torch.cuda.synchronize()
                lat = (time.perf_counter() - t0) / (5 * B)
                cuda_reset()
                out = m.forward_features(x)
                loss = out.float().pow(2).mean()
                loss.backward()
                torch.cuda.synchronize()
                bwd_peak = torch.cuda.max_memory_allocated()
                m.zero_grad(set_to_none=True)
                rows.append(dict(stage="backbone", model=name, batch=B,
                                 params_M=round(n_par / 1e6, 1),
                                 feat_dim=int(out.shape[1]) if out.ndim == 2 else None,
                                 fwd_peak_MB=mb(fwd_peak), train_peak_MB=mb(bwd_peak),
                                 infer_ms_per_sample=round(lat * 1e3, 3)))
            del m; cuda_reset()
        except Exception as e:
            rows.append(dict(stage="backbone", model=name, error=f"{type(e).__name__}: {e}"))
    return rows

if __name__ == "__main__":
    torch.manual_seed(0)
    all_rows = []
    print("[1/3] imaging stage ..."); all_rows += profile_imaging()
    print("[2/3] variate scaling ..."); all_rows += profile_variate_scaling()
    print("[3/3] backbones ..."); all_rows += profile_backbones()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(all_rows, open(OUT, "w"), indent=1)
    print(f"\nwrote {len(all_rows)} rows -> {OUT}\n")
    for r in all_rows:
        print("  " + json.dumps(r))
