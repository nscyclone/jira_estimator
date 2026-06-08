import pandas as pd
from sklearn.model_selection import train_test_split
from config import CONFIG

def split_data():
    filename = CONFIG['dataset_path']
    print(f'Reading data from {filename}')
    df = pd.read_csv(filename)
    print(f'Imported {len(df)} rows')

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['risk_level'],
        random_state=42
    )

    test_df, val_df = train_test_split(
        val_df,
        test_size=0.5,
        stratify=val_df['risk_level'],
        random_state=42
    )

    train_df_filename, val_df_filename, test_df_filename = CONFIG['train_path'], CONFIG['val_path'], CONFIG['test_path']
    train_df.to_csv(train_df_filename, index=False)
    val_df.to_csv(val_df_filename, index=False)
    test_df.to_csv(test_df_filename, index=False)
    print(f'Saved the training ({len(train_df)} rows), validation ({len(val_df)} rows) and test ({len(test_df)} rows) datasets to {train_df_filename}, {val_df_filename} and {test_df_filename}')


if __name__ == '__main__':
    split_data()