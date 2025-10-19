"""
Test script to harvest ML/DL/Visualization packages from PyPI.
This will harvest 54 popular packages across multiple domains.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.pypi_harvester import PyPIHarvester, get_ml_dl_viz_packages
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def main():
    print("=" * 80)
    print("🚀 ML/DL/Visualization Package Harvester")
    print("=" * 80)
    
    # Get all packages
    packages = get_ml_dl_viz_packages()
    
    print(f"\n📦 Total packages to harvest: {len(packages)}")
    print("\n📚 Package Categories:")
    print("  • Deep Learning: 9 packages (TensorFlow, PyTorch, Keras, JAX, etc.)")
    print("  • Machine Learning: 10 packages (scikit-learn, XGBoost, LightGBM, etc.)")
    print("  • Data Processing: 5 packages (SciPy, statsmodels, etc.)")
    print("  • Computer Vision: 6 packages (OpenCV, Pillow, torchvision, etc.)")
    print("  • NLP: 6 packages (NLTK, spaCy, Gensim, Transformers, etc.)")
    print("  • Visualization: 7 packages (Matplotlib, Seaborn, Plotly, etc.)")
    print("  • Interpretability: 3 packages (SHAP, LIME, ELI5)")
    print("  • MLOps: 3 packages (MLflow, W&B, TensorBoard)")
    print("  • Utilities: 5 packages (BeautifulSoup4, LangChain, Pandas, NumPy, etc.)")
    
    print("\n" + "=" * 80)
    print("⏳ Starting harvest... This may take 5-10 minutes")
    print("=" * 80)
    
    # Create harvester and run
    harvester = PyPIHarvester(packages)
    result = harvester.harvest_package_docs()
    
    # Print results
    print("\n" + "=" * 80)
    print("✅ HARVEST COMPLETE")
    print("=" * 80)
    print(f"📊 Statistics:")
    print(f"  • Packages harvested: {result['harvested']}/{len(packages)}")
    print(f"  • Errors: {result['errors']}")
    print(f"  • Success rate: {(result['harvested']/len(packages)*100):.1f}%")
    print(f"  • Output directory: {result['output_dir']}")
    
    if result['errors'] > 0:
        print(f"\n⚠️  Some packages failed to harvest. Check logs for details.")
    
    print("\n💡 Next steps:")
    print("  1. Run the ingestion pipeline to load data into vector store:")
    print("     docker exec rag-app python app/ingestion_pipeline.py")
    print("  2. Test queries via Streamlit UI at http://localhost:8501")
    print("=" * 80)

if __name__ == "__main__":
    main()
