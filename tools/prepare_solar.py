"""Convert Solar-AL (solar_AL.txt from LSTNet) into a CSV readable by `--data custom`.

    python tools/prepare_solar.py --src ./dataset/solar/solar_AL.txt --dst ./dataset/solar/solar.csv

Solar-AL has no timestamps; it is 10-minute data for 2006, so a synthetic `date` column is added.
The last column is renamed to `OT`, the default target of the custom loader.
"""
import argparse

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='./dataset/solar/solar_AL.txt')
    parser.add_argument('--dst', default='./dataset/solar/solar.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.src, header=None)
    df.columns = [str(c) for c in df.columns[:-1]] + ['OT']
    df.insert(0, 'date', pd.date_range('2006-01-01', periods=len(df), freq='10min'))
    df.to_csv(args.dst, index=False)
    print(f'wrote {args.dst}: {df.shape[0]} rows, {df.shape[1] - 1} variates')
