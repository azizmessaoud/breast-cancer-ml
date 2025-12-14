import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_models = {}
        np.random.seed(random_state)
    
    def train_random_forest(self, X_train, y_train, tune=True):
        logger.info("\n" + "="*80)
        logger.info("TRAINING: RANDOM FOREST")
        logger.info("="*80)
        
        if tune:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            logger.info(f"✅ Best CV Score: {grid_search.best_score_:.4f}")
            logger.info(f"✅ Best Parameters: {grid_search.best_params_}")
            self.best_models['random_forest'] = grid_search.best_estimator_
            return grid_search.best_estimator_
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X_train, y_train)
            return rf
    
    def train_svm(self, X_train, y_train, tune=True):
        logger.info("\n" + "="*80)
        logger.info("TRAINING: SUPPORT VECTOR MACHINE (SVM)")
        logger.info("="*80)
        
        if tune:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
            svm = SVC(probability=True, random_state=self.random_state)
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            logger.info(f"✅ Best CV Score: {grid_search.best_score_:.4f}")
            logger.info(f"✅ Best Parameters: {grid_search.best_params_}")
            self.best_models['svm'] = grid_search.best_estimator_
            return grid_search.best_estimator_
        else:
            svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=self.random_state)
            svm.fit(X_train, y_train)
            return svm
    
    def train_gradient_boosting(self, X_train, y_train, tune=True):
        logger.info("\n" + "="*80)
        logger.info("TRAINING: GRADIENT BOOSTING")
        logger.info("="*80)
        
        if tune:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            gb = GradientBoostingClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            logger.info(f"✅ Best CV Score: {grid_search.best_score_:.4f}")
            logger.info(f"✅ Best Parameters: {grid_search.best_params_}")
            self.best_models['gradient_boosting'] = grid_search.best_estimator_
            return grid_search.best_estimator_
        else:
            gb = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
            gb.fit(X_train, y_train)
            return gb
    
    def train_xgboost(self, X_train, y_train, tune=True):
        logger.info("\n" + "="*80)
        logger.info("TRAINING: XGBOOST")
        logger.info("="*80)
        
        if tune:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5]
            }
            xgb_model = xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
            grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            logger.info(f"✅ Best CV Score: {grid_search.best_score_:.4f}")
            logger.info(f"✅ Best Parameters: {grid_search.best_params_}")
            self.best_models['xgboost'] = grid_search.best_estimator_
            return grid_search.best_estimator_
        else:
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=self.random_state, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            return xgb_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION RESULTS: {model_name.upper()}")
        logger.info(f"{'='*80}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        self.results[model_name] = metrics
        return metrics
    
    def compare_models(self):
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        
        comparison_df = pd.DataFrame({
            model_name: {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc']
            }
            for model_name, metrics in self.results.items()
        }).T
        
        logger.info("\n" + comparison_df.to_string())
        best_model_name = comparison_df['F1-Score'].idxmax()
        logger.info(f"\n✅ Best Model: {best_model_name} (F1-Score: {comparison_df.loc[best_model_name, 'F1-Score']:.4f})")
        
        return comparison_df
    
    def plot_confusion_matrices(self, models_dict, X_test, y_test):
        n_models = len(models_dict)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, model) in enumerate(models_dict.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name.upper()} - Confusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        logger.info("✅ Confusion matrices saved as 'confusion_matrices.png'")
        plt.close()
    
    def plot_roc_curves(self, models_dict, X_test, y_test):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, model in models_dict.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name.upper()} (AUC = {roc_auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        logger.info("✅ ROC curves saved as 'roc_curves.png'")
        plt.close()
    
    def plot_model_comparison(self):
        comparison_df = pd.DataFrame({
            model_name: {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc']
            }
            for model_name, metrics in self.results.items()
        }).T
        
        fig, ax = plt.subplots(figsize=(12, 6))
        comparison_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0.9, 1.0])
        ax.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("✅ Model comparison plot saved as 'model_comparison.png'")
        plt.close()
