import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import logging
from utils.logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='From Images to Signals: LVMs for time series')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--save_dir', type=str, default='.', help='save dir')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # classification task
    parser.add_argument('--time2img', type=str, default=None, help='image transformation method for classification task')
    parser.add_argument('--max_dim', type=int, default=75, help='max dimension for classification')
    parser.add_argument('--resize', type=str, default='default', help='resize method for classification')
    # interpolation set below
    # ckpt, load_ckpt, backbone set below

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # forecasting - ViT
    parser.add_argument('--load_ckpt', type=bool, default=True)
    parser.add_argument('--no_ckpt', action='store_true', default=False,
                        help='train from scratch, no checkpoint')
    parser.add_argument('--ckpt', type=str, default="./ckpt/")
    parser.add_argument('--backbone', type=str, default='vit_base')
    parser.add_argument('--vm_dim', type=int, default=768)
    parser.add_argument('--periodicity', type=int, default=0)
    parser.add_argument('--interpolation', type=str, default='bilinear')
    parser.add_argument('--num_features', type=int, default=0)
    parser.add_argument('--context_len', type=int, default=0)
    parser.add_argument('--head_dim', type=int, default=256)
    parser.add_argument('--head_dropout', type=float, default=0.1)

    ### forecasting - MAE
    parser.add_argument('--norm_const', type=float, default=0.4)
    parser.add_argument('--align_const', type=float, default=0.4)
    parser.add_argument('--vm_arch', type=str, default='vit_base')
    parser.add_argument('--ft_type', type=str, default='ln')

    ### SimMIM - NEED TO REDOWNLOAD CHECKPOINT IF CHANGED
    parser.add_argument('--mim_model_type', type=str, default='swin', help='model type')
    parser.add_argument('--mim_image_size', type=int, default=192, help='image size')
    parser.add_argument('--mim_swin_patch_size', type=int, default=4, help='patch size')
    parser.add_argument('--mim_swin_in_chans', type=int, default=3, help='input channels')
    parser.add_argument('--mim_swin_embed_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--mim_swin_depths', type=list, default=[2, 2, 18, 2], help='depths of each swin stage')
    parser.add_argument('--mim_swin_num_heads', type=list, default=[4, 8, 16, 32], help='number of heads in each stage')
    parser.add_argument('--mim_swin_window_size', type=int, default=6, help='window size')
    parser.add_argument('--mim_swin_mlp_ratio', type=float, default=4.0, help='mlp ratio')
    parser.add_argument('--mim_swin_qkv_bias', type=bool, default=True, help='qkv bias')
    parser.add_argument('--mim_swin_qk_scale', type=float, default=None, help='qk scale')
    parser.add_argument('--mim_swin_drop_rate', type=float, default=0.0, help='drop rate')
    parser.add_argument('--mim_swin_drop_path_rate', type=float, default=0.0, help='drop path rate')
    parser.add_argument('--mim_swin_ape', type=bool, default=False, help='absolute positional embedding')
    parser.add_argument('--mim_swin_patch_norm', type=bool, default=True, help='patch norm')
    parser.add_argument('--mim_swin_use_checkpoint', type=bool, default=False, help='use checkpoint')
    parser.add_argument('--mim_ckpt', type=str, default='simmim_pretrain__swin_base__img192_window6__800ep.pth',
                        help='checkpoint file name')
    
    parser.add_argument('--shuffle_flag', type=str, default=None, help='sf_all, sf_half, ex_half, mask')

    parser.add_argument('--head_tokens', type=str, default='all', help='tokens read by the classification head: all | cls | mean')
    parser.add_argument('--variate_perm', type=int, default=0, help='seed for permuting variates in MVH images, 0 = native order')

    parser.add_argument('--norm_mode', type=str, default='instance', help='normalization for MAE forecasting: instance | dataset')
    parser.add_argument('--dataset_norm', type=int, default=1,
                        help='1: standardize input, 0: raw input')

    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')

    parser.add_argument('--sep_enc_dec', type=str, default=None, help='enc, dec')


    args = parser.parse_args()
    if args.no_ckpt:
        args.load_ckpt = False
        print('LVM Train from scratch')
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            if args.save_dir != '.':
                setting = '_'
            else:
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if args.shuffle_flag:
                exp.test_shuffle(setting)
            else:
                exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        if args.save_dir != '.':
                setting = '_'
        else:
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
