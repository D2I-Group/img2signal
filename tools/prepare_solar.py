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
