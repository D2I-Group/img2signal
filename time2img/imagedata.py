import time2img.gaf as gaf
import time2img.rp as rp
import time2img.spectrogram as sp
import time2img.heatmap as hm
from visionts.util import safe_resize
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def image_transform(input_series, args, resizer):
    if args.time2img == 'GAF':
        image_plotter = gaf.GAF_plotter()
    elif args.time2img == 'MVH':
        image_plotter = None
    elif args.time2img == 'UVH':
        image_plotter = hm.AdaptiveHeatmap_Plotter()
    elif args.time2img == 'RP':
        image_plotter = rp.RP_plotter()
    elif args.time2img == 'STFT':
        image_plotter = sp.STFT_Plotter()
    elif args.time2img == 'Wavelet':
        image_plotter = sp.Wavelet_Plotter()
    elif args.time2img == 'Filterbank':
        image_plotter = sp.Filterbank_Plotter()

    if image_plotter:
        result = []
        _, D = input_series.shape
        D = args.max_dim if D > args.max_dim else D
        
        for i in range(D):
            try:
                img_result = image_plotter.plot(input_series[:, i])
                img_result = torch.tensor(img_result, dtype=torch.float32).unsqueeze(0)
                img_result = resizer(img_result)
                result.append(img_result)
            except:
                print(f"Error processing dimension {i} of the time series.")
                result.append(torch.rand((1, 224, 224), dtype=torch.float32))
        result = torch.cat(result, dim=0)
    else:
        # MVH: all variates in one image. Optionally permute variate order
        # to test sensitivity to the (otherwise arbitrary) row ordering.
        seed = getattr(args, 'variate_perm', 0)
        arr = input_series
        if seed:
            import numpy as _np
            rng = _np.random.RandomState(seed)
            arr = arr[:, rng.permutation(arr.shape[1])]
        img_result = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        result = resizer(img_result)
    return result


def image_transform_forecasting(input_series, args):
    if args.time2img == 'GAF':
        image_plotter = gaf.GAF_plotter()
    elif args.time2img == 'RP':
        image_plotter = rp.RP_plotter()
    elif args.time2img == 'STFT':
        image_plotter = sp.STFT_Plotter()
    elif args.time2img == 'Wavelet':
        image_plotter = sp.WaveletForecast_Plotter()
    elif args.time2img == 'Filterbank':
        image_plotter = sp.Filterbank_Plotter()

    if image_plotter:
        result = []
        _, D = input_series.shape
        for i in range(D):
            img_result = image_plotter.plot(input_series[:, i])
            img_result = torch.tensor(img_result, dtype=torch.float32).unsqueeze(0)
            result.append(img_result)
        result = torch.cat(result, dim=0)
    else:
        img_result = torch.tensor(input_series, dtype=torch.float32).unsqueeze(0)
    return result


class ImageTimeSeries(Dataset):
    def __init__(self, input_set, args):
        interpolation = {
                "bilinear": Image.BILINEAR,
                "nearest": Image.NEAREST,
                "bicubic": Image.BICUBIC,
            }[args.interpolation]
        if args.resize == 'default':
            resizer = safe_resize((224, 224), interpolation=interpolation)
        elif args.resize == 'sam':
            resizer = safe_resize((1024, 1024), interpolation=interpolation)
        elif args.resize == 'simmim':
            resizer = safe_resize((192, 192), interpolation=interpolation)
        self.data = [image_transform(i[0], args, resizer) for i in tqdm(input_set)]
        self.label = [i[1] for i in input_set]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], torch.tensor(0)
    

class ImageTimeSeriesForecasting(Dataset):
    def __init__(self, input_set, args):
        tot = len(input_set)
        self.data = [image_transform_forecasting(input_set[i][0], args) for i in tqdm(range(tot))]
        self.y = [input_set[i][1] for i in range(tot)]
        self.x_mark = [input_set[i][2] for i in range(tot)]
        self.y_mark = [input_set[i][3] for i in range(tot)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.y[idx]
        x_mark = self.x_mark[idx]
        y_mark = self.y_mark[idx]
        return x, y, x_mark, y_mark