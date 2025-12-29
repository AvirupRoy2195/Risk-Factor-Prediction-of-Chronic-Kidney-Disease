import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime

class ModelMonitor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.log_file = self.config['app']['log_file']
        self.train_data_path = self.config['app']['training_data']
    
    def log_prediction(self, input_data: dict, prediction: str, probability: float):
        """Logs a single prediction event to CSV."""
        # Add timestamp and result
        log_entry = input_data.copy()
        log_entry['timestamp'] = datetime.now().isoformat()
        log_entry['prediction'] = prediction
        log_entry['probability'] = probability
        
        # Convert to DataFrame
        df_log = pd.DataFrame([log_entry])
        
        # Append to file (header only if file doesn't exist)
        file_exists = os.path.isfile(self.log_file)
        df_log.to_csv(self.log_file, mode='a', header=not file_exists, index=False)
    
    def check_drift(self):
        """
        Simple Drift Detection: Compares mean of key features 
        between Training Data and Inference Logs.
        """
        if not os.path.exists(self.log_file):
            return {"status": "No logs yet", "drift_detected": False}
            
        logs = pd.read_csv(self.log_file)
        train = pd.read_csv(self.train_data_path)
        
        # Ensure numeric
        key_features = self.config['features']['important']
        drift_report = {}
        drift_detected = False
        
        for feature in key_features:
            if feature in logs.columns and feature in train.columns:
                # Basic Statistical Test: Compare Means (simplified for MVP)
                # In production, use KS-test or PSI
                train_mean = train[feature].mean()
                log_mean = logs[feature].mean() # Last N samples could be better
                
                # Check for % deviation
                if train_mean != 0:
                    deviation = abs(log_mean - train_mean) / abs(train_mean)
                else:
                    deviation = abs(log_mean)
                
                drift_report[feature] = {
                    "train_mean": round(train_mean, 4),
                    "live_mean": round(log_mean, 4),
                    "deviation": round(deviation, 4)
                }
                
                if deviation > self.config['monitoring']['drift_threshold']:
                    drift_detected = True
        
        return {"status": "Success", "drift_detected": drift_detected, "details": drift_report}

if __name__ == "__main__":
    monitor = ModelMonitor()
    print(monitor.check_drift())
