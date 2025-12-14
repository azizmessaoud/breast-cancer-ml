import logging
import argparse
import sys
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from enhanced_model_trainer import AdvancedModelTrainer
from sklearn.decomposition import PCA
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step_1_data_loading():
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA LOADING & EXPLORATION")
    logger.info("="*80)
    try:
        loader = DataLoader('data.csv')
        loader.load_data()
        loader.clean_unnamed_columns()
        loader.explore_data()
        df = loader.get_data()
        logger.info(f"\n✅ STEP 1 COMPLETE - Data ready: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ STEP 1 FAILED: {e}")
        raise

def run_step_2_preprocessing(df):
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("="*80)
    try:
        prep = DataPreprocessor(df)
        prep.remove_id_column()
        prep.encode_diagnosis()
        X, y = prep.split_features_target()
        prep.train_test_split_data(X, y)
        X_train_scaled, X_test_scaled = prep.standardize_features()
        logger.info(f"\n✅ STEP 2 COMPLETE - Preprocessing done")
        return {
            'preprocessor': prep,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
        }
    except Exception as e:
        logger.error(f"❌ STEP 2 FAILED: {e}")
        raise

def run_step_3_pca(X_train_scaled, X_test_scaled):
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PCA DIMENSIONALITY REDUCTION")
    logger.info("="*80)
    try:
        pca = PCA()
        pca.fit(X_train_scaled)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= 0.95) + 1
        
        logger.info(f"Original features: {X_train_scaled.shape[1]}")
        logger.info(f"Components for 95% variance: {n_components}")
        
        pca_final = PCA(n_components=n_components)
        X_train_pca = pca_final.fit_transform(X_train_scaled)
        X_test_pca = pca_final.transform(X_test_scaled)
        
        variance_explained = pca_final.explained_variance_ratio_.sum()
        logger.info(f"PCA Results:")
        logger.info(f"  Reduced dimensions: {X_train_pca.shape[1]}")
        logger.info(f"  Variance retained: {variance_explained:.2%}")
        logger.info(f"\n✅ STEP 3 COMPLETE - PCA applied")
        return X_train_pca, X_test_pca
    except Exception as e:
        logger.error(f"❌ STEP 3 FAILED: {e}")
        raise

def run_step_4_model_training(X_train_pca, y_train, tune=True):
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MODEL TRAINING & HYPERPARAMETER TUNING")
    logger.info("="*80)
    try:
        trainer = AdvancedModelTrainer(random_state=42)
        logger.info("\n⏳ Training models (this may take a few minutes)...")
        
        rf_model = trainer.train_random_forest(X_train_pca, y_train, tune=tune)
        svm_model = trainer.train_svm(X_train_pca, y_train, tune=tune)
        gb_model = trainer.train_gradient_boosting(X_train_pca, y_train, tune=tune)
        xgb_model = trainer.train_xgboost(X_train_pca, y_train, tune=tune)
        
        logger.info(f"\n✅ STEP 4 COMPLETE - All models trained")
        
        return {
            'trainer': trainer,
            'models': {
                'random_forest': rf_model,
                'svm': svm_model,
                'gradient_boosting': gb_model,
                'xgboost': xgb_model
            },
            'y_train': y_train
        }
    except Exception as e:
        logger.error(f"❌ STEP 4 FAILED: {e}")
        raise

def run_step_5_evaluation(step4_result, X_test_pca, y_test):
    logger.info("\n" + "="*80)
    logger.info("STEP 5: MODEL EVALUATION & VISUALIZATION")
    logger.info("="*80)
    try:
        trainer = step4_result['trainer']
        models = step4_result['models']
        
        for model_name, model in models.items():
            trainer.evaluate_model(model, X_test_pca, y_test, model_name)
        
        comparison_df = trainer.compare_models()
        logger.info("\n⏳ Creating visualizations...")
        trainer.plot_confusion_matrices(models, X_test_pca, y_test)
        trainer.plot_roc_curves(models, X_test_pca, y_test)
        trainer.plot_model_comparison()
        
        logger.info(f"\n✅ STEP 5 COMPLETE - Evaluation and visualizations done")
        comparison_df.to_csv('model_comparison_results.csv')
        logger.info("✅ Results saved to 'model_comparison_results.csv'")
        
        return comparison_df
    except Exception as e:
        logger.error(f"❌ STEP 5 FAILED: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Enhanced Breast Cancer ML Pipeline')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4, 5], help='Run specific stage (1-5)')
    parser.add_argument('--all', action='store_true', help='Run all stages')
    parser.add_argument('--tune', action='store_true', default=True, help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    try:
        if args.all or args.stage == 1:
            df = run_step_1_data_loading()
        
        if args.all or args.stage == 2:
            if not (args.all or args.stage == 1):
                df = run_step_1_data_loading()
            result_step2 = run_step_2_preprocessing(df)
            X_train_pca = result_step2['X_train_scaled']
            X_test_pca = result_step2['X_test_scaled']
            y_train = result_step2['preprocessor'].y_train
            y_test = result_step2['preprocessor'].y_test
        
        if args.all or args.stage == 3:
            if not (args.all or args.stage in [1, 2]):
                df = run_step_1_data_loading()
                result_step2 = run_step_2_preprocessing(df)
                X_train_pca, X_test_pca = run_step_3_pca(result_step2['X_train_scaled'], result_step2['X_test_scaled'])
                y_train = result_step2['preprocessor'].y_train
                y_test = result_step2['preprocessor'].y_test
            else:
                X_train_pca, X_test_pca = run_step_3_pca(X_train_pca, X_test_pca)
        
        if args.all or args.stage == 4:
            if not (args.all or args.stage in [1, 2, 3]):
                df = run_step_1_data_loading()
                result_step2 = run_step_2_preprocessing(df)
                X_train_pca, X_test_pca = run_step_3_pca(result_step2['X_train_scaled'], result_step2['X_test_scaled'])
                y_train = result_step2['preprocessor'].y_train
                y_test = result_step2['preprocessor'].y_test
            step4_result = run_step_4_model_training(X_train_pca, y_train, tune=args.tune)
        
        if args.all or args.stage == 5:
            if not (args.all or args.stage in [1, 2, 3, 4]):
                df = run_step_1_data_loading()
                result_step2 = run_step_2_preprocessing(df)
                X_train_pca, X_test_pca = run_step_3_pca(result_step2['X_train_scaled'], result_step2['X_test_scaled'])
                y_train = result_step2['preprocessor'].y_train
                y_test = result_step2['preprocessor'].y_test
                step4_result = run_step_4_model_training(X_train_pca, y_train, tune=args.tune)
            
            comparison_df = run_step_5_evaluation(step4_result, X_test_pca, y_test)
        
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE EXECUTION COMPLETE!")
        logger.info("="*80)
        logger.info("\nGenerated Outputs:")
        logger.info("  ✅ confusion_matrices.png")
        logger.info("  ✅ roc_curves.png")
        logger.info("  ✅ model_comparison.png")
        logger.info("  ✅ model_comparison_results.csv")
        logger.info("\n🎉 All operations successful!")
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
