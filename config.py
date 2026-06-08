CONFIG = {
    # Model & tokenizer
    'model_name': 'cointegrated/rubert-tiny2',
    'max_length': 512,
    'num_risk_classes': 4,

    # Hyperparams
    'batch_size': 64,
    'lr': 1e-5,
    'epochs': 20,

    # Datasets
    'seed_path': 'data/seed.csv',
    'dataset_path': 'data/dataset.csv',
    'train_path': 'data/train.csv',
    'val_path': 'data/val.csv',

    # Model output
    'model_save_path': 'models/multitask_rubert.pt'
}
