import logging
import sys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    try:
        import pandas, numpy, sklearn
        logger.info("✅ Imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_dataloader():
    try:
        from data_loader import DataLoader
        logger.info("✅ DataLoader imported")
        return True
    except Exception as e:
        logger.error(f"❌ DataLoader failed: {e}")
        return False

def test_preprocessor():
    try:
        from data_preprocessor import DataPreprocessor
        import pandas as pd
        dummy = pd.DataFrame({'id':[1], 'diagnosis':['M'], 'f1':[1.0]})
        p = DataPreprocessor(dummy)
        logger.info("✅ DataPreprocessor imported")
        return True
    except Exception as e:
        logger.error(f"❌ Preprocessor failed: {e}")
        return False

if __name__ == "__main__":
    results = [test_imports(), test_dataloader(), test_preprocessor()]
    if all(results):
        logger.info("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        logger.error("\n❌ SOME TESTS FAILED")
        sys.exit(1)
