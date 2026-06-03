# path: src/data_ingestion.py
import pandas as pd
import numpy as np
import os
import logging

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WellboreDataPipeline:
    def __init__(self, data_path: str):
        """
        initializes the data pipeline for the rogii geosteering project.
        """
        self.data_path = data_path
        if os.path.exists(data_path):
            self.horizontal_files = [f for f in os.listdir(data_path) if 'horizontal_well.csv' in f]
        else:
            self.horizontal_files = []
            logging.warning(f"path {data_path} does not exist.")
        logging.info(f"initialized pipeline. found {len(self.horizontal_files)} horizontal wells.")

    def load_well_pair(self, well_id: str):
        """
        loads both the horizontal well and its corresponding typewell reference.
        fills missing values in gamma ray (gr) using linear interpolation.
        """
        horiz_file = f"{well_id}__horizontal_well.csv"
        type_file = f"{well_id}__typewell.csv"
        
        horiz_df = pd.read_csv(os.path.join(self.data_path, horiz_file))
        type_df = pd.read_csv(os.path.join(self.data_path, type_file))
        
        # handle missing sensor data seamlessly
        if 'GR' in horiz_df.columns:
            horiz_df['GR'] = horiz_df['GR'].interpolate(method='linear').bfill().ffill()
            
        return horiz_df, type_df

    def analyze_evaluation_zones(self, sample_size: int = 5) -> pd.DataFrame:
        """
        dynamically detects the split between known context and the hidden evaluation zone.
        """
        logging.info(f"analyzing evaluation zones for {sample_size} sample wells...")
        results = []
        
        for file in self.horizontal_files[:sample_size]:
            well_id = file.split('__')[0]
            df = pd.read_csv(os.path.join(self.data_path, file))
            
            total_rows = len(df)
            nan_rows = df['TVT_input'].isna().sum()
            known_rows = total_rows - nan_rows
            
            first_nan_idx = df[df['TVT_input'].isna()].index[0] if nan_rows > 0 else "no nans"
            
            results.append({
                'well_id': well_id,
                'total_rows': total_rows,
                'context_rows': known_rows,
                'forecast_rows': nan_rows,
                'forecast_start_idx': first_nan_idx
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    train_path = "/kaggle/input/competitions/rogii-wellbore-geology-prediction/train"
    pipeline = WellboreDataPipeline(train_path)
    zone_stats = pipeline.analyze_evaluation_zones(sample_size=5)
    print(zone_stats)
