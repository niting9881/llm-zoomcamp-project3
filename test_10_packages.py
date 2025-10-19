from app.pypi_harvester import PyPIHarvester
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Test with 10 popular ML/DL packages
test_packages = [
    'scikit-learn',
    'tensorflow', 
    'torch',
    'matplotlib',
    'seaborn',
    'plotly',
    'xgboost',
    'shap',
    'mlflow',
    'opencv-python'
]

print(f"\n{'='*80}")
print(f"Testing PyPI Harvester with {len(test_packages)} ML/DL packages")
print(f"{'='*80}\n")

harvester = PyPIHarvester(test_packages)
result = harvester.harvest_package_docs()

print(f"\n{'='*80}")
print(f"Test Results:")
print(f"  Harvested: {result['harvested']}/{len(test_packages)}")
print(f"  Errors: {result['errors']}")
print(f"  Success Rate: {(result['harvested']/len(test_packages)*100):.1f}%")
print(f"{'='*80}\n")
