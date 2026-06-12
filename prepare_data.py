import pandas as pd
import numpy as np
from config import CONFIG

SEC_TO_WORKDAY = 28800 # 8 * 60 * 60

def prepare_data():
    filename = CONFIG['seed_path']
    print(f'Reading data from {filename}')
    df = pd.read_csv(filename, sep=',', on_bad_lines='skip')
    print(f'Imported {len(df)} rows')

    df['summary'] = df['summary'].fillna('')
    df['description'] = df['description'].fillna('')

    df['text'] = df['summary'] + ' ' + df['description']

    df['logged_days'] = df['time_spent'] / SEC_TO_WORKDAY

    # Dropping rows having neither estimates nor worklogs
    df = df.dropna(subset=['logged_days', 'estimate'])

    # Dropping rows with an implicit zero estimate and estimates higher than 10 FTE
    df = df[df['estimate'] > 0]
    df = df[df['estimate'] <= 10.0]

    print(f'Rows left after filtering: {len(df)}')

    ratio = df['logged_days'] / df['estimate']

    ratio_conditions = [
        (ratio <= 1.0),
        (ratio > 1.0) & (ratio <= 1.25),
        (ratio > 1.25) & (ratio <= 1.5),
        (ratio > 1.5)
    ]

    ratio_labels = [0, 1, 2, 3]

    df['risk_level'] = np.select(ratio_conditions, ratio_labels, default=0)

    prepared_df = df[['text', 'estimate', 'risk_level']]

    output_filename = CONFIG['dataset_path']
    print(f'Writing to {output_filename}')
    prepared_df.to_csv(output_filename, index=False)
    print(f'Saved to {output_filename}')

    print(f'Risk class distribution is:')
    print(prepared_df['risk_level'].value_counts())


if __name__ == '__main__':
    prepare_data()



