# path: src/validation.py
import pandas as pd
import os
import logging
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ValidationStrategy:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def generate_cv_folds(self, n_splits: int = 5) -> pd.DataFrame:
        """
        generates data-leakage free splits grouped by well_id.
        ensures an entire well is either completely in train or completely in validation.
        """
        logging.info(f"generating {n_splits}-fold groupkfold cross-validation strategy...")
        
        all_files = os.listdir(self.data_path)
        well_ids = list(set([f.split('__')[0] for f in all_files if 'horizontal_well' in f]))
        
        df_meta = pd.DataFrame({'well_id': well_ids})
        gkf = GroupKFold(n_splits=n_splits)
        df_meta['fold'] = -1
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_meta, groups=df_meta['well_id'])):
            df_meta.loc[val_idx, 'fold'] = fold
            
        logging.info("cross-validation strategy built successfully.")
        return df_meta

if __name__ == "__main__":
    train_path = "/kaggle/input/competitions/rogii-wellbore-geology-prediction/train"
    validator = ValidationStrategy(train_path)
    folds_df = validator.generate_cv_folds()
    print(folds_df['fold'].value_counts())
