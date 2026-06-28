import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import CONFIG
from feature_engineering import DEV_WORDS, TEST_WORDS, ANALYSIS_WORDS

SEC_TO_WORKDAY = 28800  # 8 * 60 * 60


def _parse_region(raw):
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return 'БАЗОВЫЙ'
    if not isinstance(parsed, list) or len(parsed) == 0:
        return 'БАЗОВЫЙ'
    return ', '.join(sorted(str(item.get('value', 'БАЗОВЫЙ')) for item in parsed if isinstance(item, dict)))


def _parse_subsystem(raw):
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return ''
    if not isinstance(parsed, dict) or 'value' not in parsed:
        return ''
    child = parsed.get('child', {}).get('value', '')
    return f"{parsed['value']}/{child}".rstrip('/')


def _parse_commitments(raw):
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return ''
    if not isinstance(parsed, dict):
        return ''
    return parsed.get('value', '')


def prepare_data():
    filename = CONFIG['seed_path']
    print(f'Reading data from {filename}')
    df = pd.read_csv(filename, sep=',', on_bad_lines='skip')
    print(f'Imported {len(df)} rows')

    df['summary'] = df['summary'].fillna('')
    df['description'] = df['description'].fillna('')
    df['text'] = df['summary'] + ' ' + df['description']

    df['has_description'] = (df['description'].str.strip() != '').astype(int)
    df['has_code_block'] = df['description'].str.contains('{code', case=False, regex=False).astype(int)

    text_lower = df['text'].str.lower()
    df['is_dev_task'] = text_lower.apply(lambda x: int(any(w in x for w in DEV_WORDS)))
    df['is_test_task'] = text_lower.apply(lambda x: int(any(w in x for w in TEST_WORDS)))
    df['is_analysis_task'] = text_lower.apply(lambda x: int(any(w in x for w in ANALYSIS_WORDS)))

    df['text_len'] = df['text'].str.len()
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))

    df['logged_days'] = df['time_spent'] / SEC_TO_WORKDAY

    df['region'] = df['region'].apply(_parse_region)
    df['subsystem'] = df['subsystem'].apply(_parse_subsystem)
    df['commitments'] = df['commitments'].apply(_parse_commitments)

    df = df.dropna(subset=['logged_days', 'estimate'])
    df = df[df['estimate'] > 0]
    df = df[df['estimate'] <= 10.0]

    print(f'Rows left after filtering: {len(df)}')

    ratio = df['logged_days'] / df['estimate']
    ratio_conditions = [
        (ratio <= 1.0),
        (ratio > 1.0) & (ratio <= 1.5),
        (ratio > 1.5),
    ]
    df['risk_level'] = np.select(ratio_conditions, [0, 1, 2], default=0)

    prepared_df = df[[
        'text', 'estimate', 'logged_days', 'risk_level', 'region', 'subsystem', 'commitments',
        'has_description', 'has_code_block', 'is_dev_task', 'is_test_task', 'is_analysis_task',
        'text_len', 'word_count',
    ]]

    output_filename = CONFIG['dataset_path']
    print(f'Writing to {output_filename}')
    prepared_df.to_csv(output_filename, index=False)
    print(f'Saved to {output_filename}')

    print('Risk class distribution is:')
    print(prepared_df['risk_level'].value_counts())


if __name__ == '__main__':
    prepare_data()
