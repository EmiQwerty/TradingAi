"""
Model Training Pipeline - Addestra modelli per predicare win/loss
Usa RandomForest, XGBoost, LightGBM con cross-validation
Genera feature importance e interpretability
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path
import pickle
import json

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Training pipeline per modelli predittivi di win/loss.
    """
    
    def __init__(self, model_dir: str = 'ml/models'):
        """
        Args:
            model_dir: Directory per salvare modelli
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}  # {name: model}
        self.scalers = {}  # {name: scaler}
        self.feature_names = None
        self.metrics = {}  # {name: metrics}
        
        logger.info(f"ModelTrainer initialized: model_dir={model_dir}")
    
    def train_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5
    ) -> Dict[str, Dict]:
        """
        Addestra tutti i modelli disponibili.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (0/1)
            test_size: Proporzione test set
            random_state: Random seed
            cv_folds: Numero di fold per cross-validation
        
        Returns:
            Dict con metriche per ogni modello
        """
        self.feature_names = X.columns.tolist()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Dataset split: train={len(X_train)}, test={len(X_test)}")
        
        results = {}
        
        # Random Forest
        logger.info("Training RandomForest...")
        results['random_forest'] = self._train_random_forest(
            X_train, X_test, y_train, y_test, cv_folds
        )
        
        # Gradient Boosting
        logger.info("Training GradientBoosting...")
        results['gradient_boosting'] = self._train_gradient_boosting(
            X_train, X_test, y_train, y_test, cv_folds
        )
        
        # XGBoost
        if HAS_XGBOOST:
            logger.info("Training XGBoost...")
            results['xgboost'] = self._train_xgboost(
                X_train, X_test, y_train, y_test, cv_folds
            )
        
        # LightGBM
        if HAS_LIGHTGBM:
            logger.info("Training LightGBM...")
            results['lightgbm'] = self._train_lightgbm(
                X_train, X_test, y_train, y_test, cv_folds
            )
        
        self.metrics = results
        
        # Salva modelli
        self.save_all_models()
        
        return results
    
    def _train_random_forest(
        self,
        X_train, X_test, y_train, y_test,
        cv_folds
    ) -> Dict:
        """Addestra RandomForest"""
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=StratifiedKFold(cv_folds),
            scoring='f1', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predict
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_score'] = grid_search.best_score_
        
        self.models['random_forest'] = best_model
        self.scalers['random_forest'] = None  # RF non richiede scaling
        
        logger.info(f"RandomForest: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
        
        return metrics
    
    def _train_gradient_boosting(
        self,
        X_train, X_test, y_train, y_test,
        cv_folds
    ) -> Dict:
        """Addestra GradientBoosting"""
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gb, param_grid, cv=StratifiedKFold(cv_folds),
            scoring='f1', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_score'] = grid_search.best_score_
        
        self.models['gradient_boosting'] = best_model
        self.scalers['gradient_boosting'] = None
        
        logger.info(f"GradientBoosting: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
        
        return metrics
    
    def _train_xgboost(
        self,
        X_train, X_test, y_train, y_test,
        cv_folds
    ) -> Dict:
        """Addestra XGBoost"""
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=StratifiedKFold(cv_folds),
            scoring='f1', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_score'] = grid_search.best_score_
        
        self.models['xgboost'] = best_model
        self.scalers['xgboost'] = None
        
        logger.info(f"XGBoost: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
        
        return metrics
    
    def _train_lightgbm(
        self,
        X_train, X_test, y_train, y_test,
        cv_folds
    ) -> Dict:
        """Addestra LightGBM"""
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [10, 31, 50],
        }
        
        lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=StratifiedKFold(cv_folds),
            scoring='f1', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_score'] = grid_search.best_score_
        
        self.models['lightgbm'] = best_model
        self.scalers['lightgbm'] = None
        
        logger.info(f"LightGBM: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """Calcola metriche di valutazione"""
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """
        Predice label per nuovi dati.
        
        Args:
            X: Features
            model_name: Nome del modello
        
        Returns:
            Array di predizioni (0/1)
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found, using first available")
            model_name = list(self.models.keys())[0]
        
        model = self.models[model_name]
        
        # Assicura ordine feature
        X_ordered = X[self.feature_names].fillna(0)
        
        return model.predict(X_ordered)
    
    def predict_proba(self, X: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """
        Predice probabilità (confidence).
        
        Args:
            X: Features
            model_name: Nome del modello
        
        Returns:
            Array di probabilità (shape: n_samples, 2)
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found, using first available")
            model_name = list(self.models.keys())[0]
        
        model = self.models[model_name]
        X_ordered = X[self.feature_names].fillna(0)
        
        return model.predict_proba(X_ordered)
    
    def get_feature_importance(self, model_name: str = 'xgboost', top_n: int = 20) -> Dict[str, float]:
        """
        Estrae feature importance dal modello.
        
        Args:
            model_name: Nome modello
            top_n: Top N features da restituire
        
        Returns:
            Dict {feature_name: importance_score}
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not trained")
            return {}
        
        model = self.models[model_name]
        
        # Usa feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning(f"Model {model_name} doesn't have feature_importances_")
            return {}
        
        # Crea dict e ordina
        importance_dict = dict(zip(self.feature_names, importances))
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n])
        
        return importance_dict
    
    def save_all_models(self):
        """Salva tutti i modelli e scaler"""
        for name, model in self.models.items():
            model_path = self.model_dir / f"{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model: {model_path}")
        
        # Salva feature names
        feature_path = self.model_dir / "feature_names.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        # Salva metrics
        metrics_path = self.model_dir / "training_metrics.json"
        # Converti metrics per JSON serialization
        metrics_json = {}
        for model_name, metrics in self.metrics.items():
            metrics_json[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
                if not isinstance(v, dict)
            }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info("Saved training metrics")
    
    def load_model(self, model_name: str):
        """Carica modello salvato"""
        model_path = self.model_dir / f"{model_name}_model.pkl"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            self.models[model_name] = pickle.load(f)
        
        # Carica feature names
        feature_path = self.model_dir / "feature_names.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
        
        logger.info(f"Loaded model: {model_path}")
        return True
    
    def load_all_models(self):
        """Carica tutti i modelli salvati"""
        for model_path in self.model_dir.glob("*_model.pkl"):
            model_name = model_path.stem.replace('_model', '')
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            logger.info(f"Loaded model: {model_name}")
        
        # Carica feature names
        feature_path = self.model_dir / "feature_names.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
    
    def print_summary(self):
        """Stampa summary dei modelli addestrati"""
        print("\n" + "="*70)
        print("MODEL TRAINING SUMMARY")
        print("="*70)
        
        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  CV Score:  {metrics['cv_score']:.4f}")
            
            # Feature importance
            importance = self.get_feature_importance(model_name, top_n=5)
            if importance:
                print(f"  Top Features:")
                for feat, score in importance.items():
                    print(f"    - {feat}: {score:.4f}")
        
        print("\n" + "="*70 + "\n")
