import pandas as pd
import numpy as np

SEC_TO_WORKDAY = 28800 # 8 * 60 * 60

def prepare_data():
    filename = 'seed.csv'
    print(f'Reading data from {filename}')
    df = pd.read_csv(filename, sep=',', on_bad_lines='skip')
    print(f'Imported {len(df)} rows')

    df['summary'] = df['summary'].fillna('')
    df['description'] = df['description'].fillna('')

    df['text'] = df['summary'] + ' ' + df['description']

    df['logged_days'] = df['time_spent'] / SEC_TO_WORKDAY

    # Dropping rows having neither estimates nor worklogs
    df = df.dropna(subset=['logged_days', 'estimate'])

    # Dropping rows with an implicit zero estimate
    df = df[df['estimate'] > 0]

    print(f'Rows left after filtering: {len(df)}')


if __name__ == '__main__':
    prepare_data()



