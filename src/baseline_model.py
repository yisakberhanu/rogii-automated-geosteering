# path: src/baseline_model.py
import pandas as pd
import numpy as np
import os
import logging
from sklearn.metrics import root_mean_squared_error
from src.data_ingestion import WellboreDataPipeline
from src.validation import ValidationStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineConstantTrendModel:
    """
    a robust engineering baseline model.
    uses the last known trend/relationship from the context zone 
    to extrapolate tvt forward through the evaluation zone.
    """
    def __init__(self):
        pass

    def fit_predict(self, horiz_df: pd.DataFrame) -> np.ndarray:
        # copy to avoid altering original data
        df = horiz_df.copy()
        
        # identify split boundary dynamically
        eval_mask = df['TVT_input'].isna()
        
        # if no evaluation zone exists, return true tvt
        if not eval_mask.any():
            return df['TVT'].values
            
        # find the last valid index before the forecast zone starts
        last_valid_idx = df[~eval_mask].index[-1]
        
        # calculate the baseline relationship: tvt relative to md changes
        # tracking how tvt moves per unit of measured depth
        context_df = df.loc[:last_valid_idx]
        
        # simple baseline: calculate median rate of tvt change over the last 100 rows
        window = min(100, len(context_df))
        recent_context = context_df.iloc[-window:]
        
        md_diff = recent_context['MD'].diff().mean()
        tvt_diff = recent_context['TVT'].diff().mean()
        
        # fallback rate if step changes are zero or irregular
        tvt_rate_per_md = (tvt_diff / md_diff) if md_diff != 0 else 0.0
        
        # generate baseline forecasts
        predictions = df['TVT_input'].values.copy()
        last_known_tvt = df.loc[last_valid_idx, 'TVT']
        last_known_md = df.loc[last_valid_idx, 'MD']
        
        # fill the evaluation zone step by step based on spatial tracking
        for i in df[eval_mask].index:
            current_md = df.loc[i, 'MD']
            delta_md = current_md - last_known_md
            predictions[i] = last_known_tvt + (delta_md * tvt_rate_per_md)
            
        return predictions

def run_cross_validation():
    train_path = "/kaggle/input/competitions/rogii-wellbore-geology-prediction/train"
    
    pipeline = WellboreDataPipeline(train_path)
    validator = ValidationStrategy(train_path)
    
    folds_df = validator.generate_cv_folds(n_splits=5)
    
    # execute validation tracking on fold 0 as our validation anchor
    val_wells = folds_df[folds_df['fold'] == 0]['well_id'].values[:5] # evaluate on 5 sample wells first
    
    scores = []
    logging.info("starting baseline evaluation loop...")
    
    for well_id in val_wells:
        horiz_df, _ = pipeline.load_well_pair(well_id)
        
        model = BaselineConstantTrendModel()
        preds = model.fit_predict(horiz_df)
        
        # evaluate score only on the hidden evaluation zone to match kaggle exactly
        eval_mask = horiz_df['TVT_input'].isna()
        if eval_mask.any():
            y_true = horiz_df.loc[eval_mask, 'TVT'].values
            y_pred = preds[eval_mask]
            
            rmse = root_mean_squared_error(y_true, y_pred)
            scores.append(rmse)
            logging.info(f"well {well_id} - evaluation zone rmse: {rmse:.4f}")
            
    if scores:
        logging.info(f"mean baseline rmse across validation subset: {np.mean(scores):.4f}")

if __name__ == "__main__":
    run_cross_validation()
