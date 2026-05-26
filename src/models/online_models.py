"""
Online Learning Models using River
For continuous learning from streaming data
"""

from river import forest, ensemble, linear_model, tree, naive_bayes, drift
from river import metrics
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnlineCrashPredictor:
    """Online learning model for crash prediction"""
    
    def __init__(self, model_type='arf', drift_detection=True):
        """
        Initialize online model
        
        Args:
            model_type: 'arf', 'ht', 'lr', 'nb', 'ensemble'
            drift_detection: Enable drift detection
        """
        self.model_type = model_type
        self.drift_detection = drift_detection
        self.model = self._initialize_model(drift_detection)
        self.metric = metrics.ROCAUC()
        self.samples_processed = 0
        self.drift_detected_flag = False
        
        logger.info(f"Initialized {model_type} model with drift detection: {drift_detection}")
    
    def _initialize_model(self, drift_detection):
        """Initialize the selected model"""
        if drift_detection:
            self.drift_detector = drift.ADWIN(delta=0.002)
        else:
            self.drift_detector = None
        
        if self.model_type == 'arf':
            # Adaptive Random Forest
            return forest.ARFClassifier(
                n_models=10,
                max_features='sqrt',
                lambda_value=6
            )
        
        elif self.model_type == 'ht':
            # Hoeffding Tree
            return tree.HoeffdingTreeClassifier(
                grace_period=200,
                split_confidence=0.0001
            )
        
        elif self.model_type == 'lr':
            # Online Logistic Regression
            from river import preprocessing
            from river import compose
            
            return compose.Pipeline(
                ('scaler', preprocessing.StandardScaler()),
                ('lr', linear_model.LogisticRegression())
            )
        
        elif self.model_type == 'nb':
            # Naive Bayes
            return naive_bayes.GaussianNB()
        
        elif self.model_type == 'ensemble':
            # Ensemble of multiple models
            return ensemble.VotingClassifier([
                ('arf', forest.ARFClassifier(n_models=10)),
                ('ht', tree.HoeffdingTreeClassifier()),
                ('lr', linear_model.LogisticRegression())
            ])
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_initial(self, X, y, max_samples=10000):
        """
        Train initial model on historical data
        
        Args:
            X: Features DataFrame
            y: Labels Series
            max_samples: Max samples for initial training
        """
        logger.info(f"Initial training with {min(len(X), max_samples)} samples...")
        
        for idx in range(min(len(X), max_samples)):
            # Convert to dict
            x_dict = X.iloc[idx].to_dict()
            y_val = int(y.iloc[idx])
            
            # Learn from this sample
            self.learn_one(x_dict, y_val)
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1} samples")
        
        logger.info(f"Initial training complete. Samples processed: {self.samples_processed}")
    
    def predict_proba_one(self, x_dict):
        """
        Predict probability for one sample
        
        Args:
            x_dict: Feature dictionary
        
        Returns:
            Probability of crash (0-1)
        """
        x_clean = {k: float(v) for k, v in x_dict.items() if v is not None}
        pred_proba = self.model.predict_proba_one(x_clean)
        # Get probability of class 1 (crash)
        crash_prob = pred_proba.get(1, 0.0)
        return crash_prob
    
    def learn_one(self, x_dict, y_true):
        """
        Learn from one sample
        
        Args:
            x_dict: Feature dictionary
            y_true: True label (0 or 1)
        """
        x_clean = {k: float(v) for k, v in x_dict.items() if v is not None}
        y_int = int(y_true)
        
        # Estimate prediction error before update to feed to ADWIN
        pred_prob = self.predict_proba_one(x_clean)
        pred_label = 1 if pred_prob >= 0.5 else 0
        error = abs(y_int - pred_label)
        
        self.model.learn_one(x_clean, y_int)
        self.samples_processed += 1
        
        if self.drift_detector:
            self.drift_detector.update(error)
            if self.drift_detector.drift_detected:
                logger.warning(f"Concept Drift detected at sample {self.samples_processed}!")
                self.drift_detected_flag = True
            else:
                self.drift_detected_flag = False
    
    def save_state(self):
        """Serialize current model and returns binary bytes"""
        return pickle.dumps(self)

    @staticmethod
    def load_state(binary_data):
        """Deserialize from binary bytes"""
        return pickle.loads(binary_data)
