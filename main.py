import argparse
import logging
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from sklearn.decomposition import PCA
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Breast Cancer ML Pipeline')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    args = parser.parse_args()

    try:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("="*80)
        loader = DataLoader('data.csv')
        loader.load_data()
        loader.clean_unnamed_columns()
        df = loader.get_data()

        logger.info("\n" + "="*80)
        logger.info("STEP 2: PREPROCESSING")
        logger.info("="*80)
        prep = DataPreprocessor(df)
        prep.remove_id_column()
        prep.encode_diagnosis()
        X, y = prep.split_features_target()
        prep.train_test_split_data(X, y)
        X_train, X_test = prep.standardize_features()

        logger.info("\n" + "="*80)
        logger.info("STEP 3: PCA")
        logger.info("="*80)
        pca = PCA(n_components=0.95)
        pca.fit(X_train)
        logger.info(f"PCA Components: {pca.n_components_}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE SUCCESS")
        logger.info("="*80)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
