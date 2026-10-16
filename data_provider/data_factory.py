from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Custom_MVH, ShuffledDatasetForcasting, ShuffledDatasetClassification
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from time2img.imagedata import ImageTimeSeries, ImageTimeSeriesForecasting
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'customMVH': Custom_MVH,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        if args.time2img:
            temp_set = ImageTimeSeries(data_set, args)
            data_loader = DataLoader(
                temp_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=bool(getattr(args, 'dataset_norm', 1)),
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        data_set.return_channel = not bool(getattr(args, 'dataset_norm', 1))
        perm_seed = int(getattr(args, 'variate_perm', 0))
        if getattr(args, 'model', None) == 'MVH_RQ3' and perm_seed:
            permutation = np.random.RandomState(perm_seed).permutation(data_set.data_x.shape[1])
            data_set.data_x = data_set.data_x[:, permutation]
            data_set.data_y = data_set.data_y[:, permutation]
        print(flag, len(data_set))

        if args.time2img:
            temp_set = ImageTimeSeriesForecasting(data_set, args)
            print(f'{args.time2img} transform done')
            data_loader = DataLoader(
                temp_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader
        

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader


def data_provider_shuffle(args, flag='test'):
    assert flag == 'test' # Shuffle test only

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    
    if args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag.upper(),
        )
        if args.time2img:
            shuffled_set = ShuffledDatasetClassification(data_set, args)
            temp_set = ImageTimeSeries(shuffled_set, args)
            data_loader = DataLoader(
                temp_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        
        print('Before shuffle', len(data_set))
        data_set = ShuffledDatasetForcasting(data_set, args)
        print('After shuffle', len(data_set))

        assert args.time2img == None, 'Only conduct RQ6 forecasting on Univariate Heatmap'
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
