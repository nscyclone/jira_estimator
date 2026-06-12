CONFIG = {
    # Model & tokenizer
    'model_name': 'sergeyzh/BERTA',
    'max_length': 256,
    'num_risk_classes': 4,

    # Hyperparams
    'batch_size': 32,
    'lr': 1e-5,
    'epochs': 20,

    # Datasets
    'seed_path': 'data/seed.csv',
    'dataset_path': 'data/dataset.csv',
    'train_path': 'data/train.csv',
    'val_path': 'data/val.csv',
    'test_path': 'data/test.csv',

    # Model output
    'model_save_path': 'models/multitask_rubert.pt',
    'embeddings_save_path': 'embeddings',
    'catboost_estimate_model_save_path': 'models/catboost_estimate_model.cbm',
    'catboost_risk_model_save_path': 'models/catboost_risk_model.cbm'
}
