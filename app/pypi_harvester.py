import requests
from pathlib import Path
import json
import logging
import time

logger = logging.getLogger(__name__)

class PyPIHarvester:
    """
    Harvests package metadata and documentation from PyPI.
    Fetches package metadata, README, and description.
    """
    
    def __init__(self, packages):
        self.base_url = "https://pypi.org/pypi"
        self.output_dir = Path("data/raw/pypi_packages")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.packages = packages
        self.harvested_count = 0
        self.error_count = 0

    def harvest_package_docs(self):
        """
        Harvest documentation for all packages.
        
        Returns:
            dict: Statistics about the harvest
        """
        logger.info(f"Starting PyPI harvest for {len(self.packages)} packages")
        
        for idx, package_name in enumerate(self.packages, 1):
            try:
                logger.info(f"[{idx}/{len(self.packages)}] Fetching: {package_name}")
                
                url = f"{self.base_url}/{package_name}/json"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Save metadata
                meta_file = self.output_dir / f"{package_name}_meta.json"
                meta_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                logger.debug(f"Saved metadata: {meta_file}")
                
                # Try to fetch README
                readme = data['info'].get('description', '')
                if readme:
                    readme_file = self.output_dir / f"{package_name}_readme.md"
                    readme_file.write_text(readme, encoding="utf-8")
                    logger.debug(f"Saved README: {readme_file}")
                else:
                    logger.warning(f"No README found for {package_name}")
                
                self.harvested_count += 1
                logger.info(f"✅ Successfully harvested {package_name}")
                
                # Be nice to PyPI
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch {package_name}: {e}")
                self.error_count += 1
            except Exception as e:
                logger.error(f"Error processing {package_name}: {e}")
                self.error_count += 1
        
        logger.info("PyPI harvest complete")
        logger.info(f"  Harvested: {self.harvested_count} packages")
        logger.info(f"  Errors: {self.error_count}")
        
        return {
            "harvested": self.harvested_count,
            "errors": self.error_count,
            "output_dir": str(self.output_dir)
        }

def get_ml_dl_viz_packages():
    """
    Get comprehensive list of popular ML, DL, and Visualization packages.
    
    Returns:
        list: 54 popular packages across ML/DL/Visualization domains
    """
    return [
        # Original packages
        "beautifulsoup4",
        "langchain",
        "pandas",
        "numpy",
        
        # Deep Learning Frameworks
        "tensorflow",
        "torch",  # PyTorch
        "keras",
        "jax",
        "flax",
        "pytorch-lightning",
        "transformers",  # HuggingFace
        "fastai",
        "mxnet",
        
        # Machine Learning Libraries
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",  # Hyperparameter optimization
        "hyperopt",
        "sklearn-pandas",
        "imbalanced-learn",
        "mlxtend",
        
        # Data Processing & Feature Engineering
        "scipy",
        "statsmodels",
        "featuretools",
        "category-encoders",
        "feature-engine",
        
        # Computer Vision
        "opencv-python",
        "pillow",
        "torchvision",
        "albumentations",
        "imageio",
        "scikit-image",
        
        # Natural Language Processing
        "nltk",
        "spacy",
        "gensim",
        "textblob",
        "sentencepiece",
        "tokenizers",
        
        # Visualization Libraries
        "matplotlib",
        "seaborn",
        "plotly",
        "bokeh",
        "altair",
        "holoviews",
        "dash",
        
        # Model Interpretability
        "shap",
        "lime",
        "eli5",
        
        # MLOps & Experiment Tracking
        "mlflow",
        "wandb",  # Weights & Biases
        "tensorboard",
    ]

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Get comprehensive ML/DL/Viz package list (54 packages)
    packages = get_ml_dl_viz_packages()
    
    print("=" * 80)
    print("PyPI Package Harvester - ML/DL/Visualization Edition")
    print("=" * 80)
    print(f"📦 Total packages to harvest: {len(packages)}")
    print("\nCategories:")
    print("  - Deep Learning: tensorflow, torch, keras, jax, transformers, etc.")
    print("  - Machine Learning: scikit-learn, xgboost, lightgbm, catboost, etc.")
    print("  - Computer Vision: opencv-python, pillow, torchvision, etc.")
    print("  - NLP: nltk, spacy, gensim, transformers, etc.")
    print("  - Visualization: matplotlib, seaborn, plotly, bokeh, etc.")
    print("  - MLOps: mlflow, wandb, tensorboard")
    print("  - Interpretability: shap, lime, eli5")
    print("\n" + "=" * 80)
    
    harvester = PyPIHarvester(packages)
    result = harvester.harvest_package_docs()
    
    print("\n" + "=" * 80)
    print("📊 Harvest Summary")
    print("=" * 80)
    print(f"✅ Packages harvested: {result['harvested']}/{len(packages)}")
    print(f"❌ Errors: {result['errors']}")
    print(f"📁 Output directory: {result['output_dir']}")
    print(f"🎯 Success rate: {(result['harvested']/len(packages)*100):.1f}%")
    print("=" * 80)
