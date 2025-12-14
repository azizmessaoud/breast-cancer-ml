from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, X_train, y_train):
        logger.info("Training Random Forest...")
        self.model.fit(X_train, y_train)
        logger.info("✅ Model trained")
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        logger.info(f"✅ Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"✅ Precision: {metrics['precision']:.4f}")
        logger.info(f"✅ Recall: {metrics['recall']:.4f}")
        logger.info(f"✅ F1-Score: {metrics['f1']:.4f}")
        return metrics
