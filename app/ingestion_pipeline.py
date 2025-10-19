"""
Ingestion Pipeline for Python Documentation RAG System

This pipeline orchestrates the harvesting of both Python official documentation
and PyPI package documentation, then prepares the data for the RAG system.
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from python_doc_harvester import PythonDocHarvester
from pypi_harvester import PyPIHarvester, get_ml_dl_viz_packages

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Complete ingestion pipeline for Python documentation.
    Harvests both official Python docs and PyPI package documentation.
    Now includes 54 ML/DL/Visualization packages.
    """
    
    def __init__(self, use_full_package_list=True):
        """
        Initialize ingestion pipeline.
        
        Args:
            use_full_package_list: If True, harvest all 54 ML/DL/Viz packages.
                                   If False, use only the original 4 packages.
        """
        self.python_versions = ["3.14", "3.12", "3.11"]
        
        if use_full_package_list:
            # Use comprehensive ML/DL/Viz package list (54 packages)
            self.pypi_packages = get_ml_dl_viz_packages()
        else:
            # Use original 4 packages only
            self.pypi_packages = ["beautifulsoup4", "langchain", "pandas", "numpy"]
        
        self.results = {}
    
    def run(self):
        """
        Execute the complete ingestion pipeline.
        
        Returns:
            dict: Statistics about the entire harvest
        """
        logger.info("=" * 80)
        logger.info("Starting Complete Ingestion Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Harvest Python documentation
        logger.info("\n📚 STEP 1: Harvesting Python Documentation")
        logger.info("-" * 80)
        python_results = self._harvest_python_docs()
        self.results['python_docs'] = python_results
        
        # Step 2: Harvest PyPI packages
        logger.info("\n📦 STEP 2: Harvesting PyPI Packages")
        logger.info("-" * 80)
        pypi_results = self._harvest_pypi_packages()
        self.results['pypi_packages'] = pypi_results
        
        # Step 3: Summary
        self._print_summary()
        
        return self.results
    
    def _harvest_python_docs(self):
        """Harvest Python official documentation for all versions."""
        all_results = []
        
        for version in self.python_versions:
            logger.info(f"\n📖 Harvesting Python {version} documentation...")
            
            try:
                harvester = PythonDocHarvester(version)
                result = harvester.harvest_documentation(max_modules=50)
                all_results.append(result)
                
                logger.info(f"✅ Python {version}: {result['harvested']} modules")
                if result.get('errors', 0) > 0:
                    logger.warning(f"⚠️  {result['errors']} errors")
                    
            except Exception as e:
                logger.error(f"❌ Failed to harvest Python {version}: {e}")
                all_results.append({
                    "version": version,
                    "harvested": 0,
                    "errors": 1,
                    "error": str(e)
                })
        
        total_harvested = sum(r['harvested'] for r in all_results)
        total_errors = sum(r.get('errors', 0) for r in all_results)
        
        return {
            "versions": self.python_versions,
            "results": all_results,
            "total_harvested": total_harvested,
            "total_errors": total_errors
        }
    
    def _harvest_pypi_packages(self):
        """Harvest PyPI package documentation."""
        logger.info(f"\n📦 Harvesting {len(self.pypi_packages)} PyPI packages...")
        logger.info(f"Packages: {', '.join(self.pypi_packages)}")
        
        try:
            harvester = PyPIHarvester(self.pypi_packages)
            result = harvester.harvest_package_docs()
            
            logger.info(f"✅ PyPI: {result['harvested']} packages harvested")
            if result.get('errors', 0) > 0:
                logger.warning(f"⚠️  {result['errors']} errors")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to harvest PyPI packages: {e}")
            return {
                "harvested": 0,
                "errors": len(self.pypi_packages),
                "error": str(e)
            }
    
    def _print_summary(self):
        """Print a summary of the ingestion pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 INGESTION PIPELINE SUMMARY")
        logger.info("=" * 80)
        
        # Python docs summary
        python_data = self.results.get('python_docs', {})
        logger.info("\n📚 Python Documentation:")
        logger.info(f"  Versions: {', '.join(python_data.get('versions', []))}")
        logger.info(f"  Total modules harvested: {python_data.get('total_harvested', 0)}")
        logger.info(f"  Total errors: {python_data.get('total_errors', 0)}")
        
        # PyPI packages summary
        pypi_data = self.results.get('pypi_packages', {})
        logger.info("\n📦 PyPI Packages:")
        logger.info(f"  Packages: {', '.join(self.pypi_packages)}")
        logger.info(f"  Packages harvested: {pypi_data.get('harvested', 0)}/{len(self.pypi_packages)}")
        logger.info(f"  Errors: {pypi_data.get('errors', 0)}")
        
        # Overall summary
        total_items = python_data.get('total_harvested', 0) + pypi_data.get('harvested', 0)
        total_errors = python_data.get('total_errors', 0) + pypi_data.get('errors', 0)
        
        logger.info("\n📈 Overall:")
        logger.info(f"  Total items harvested: {total_items}")
        logger.info(f"  Total errors: {total_errors}")
        logger.info(f"  Success rate: {(total_items / (total_items + total_errors) * 100) if (total_items + total_errors) > 0 else 0:.1f}%")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ INGESTION PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        # Data location
        logger.info("\n📁 Data Location:")
        logger.info(f"  Python docs: data/raw/python_docs/{{version}}/")
        logger.info(f"  PyPI packages: data/raw/pypi_packages/")
        logger.info("\n💡 Next step: Run the RAG engine to ingest this data into the vector store")


def main():
    """Main entry point for the ingestion pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("ingestion_pipeline.log")
        ]
    )
    
    # Run pipeline
    pipeline = IngestionPipeline()
    results = pipeline.run()
    
    # Exit with appropriate code
    total_errors = (
        results.get('python_docs', {}).get('total_errors', 0) +
        results.get('pypi_packages', {}).get('errors', 0)
    )
    
    if total_errors > 0:
        logger.warning(f"Pipeline completed with {total_errors} errors")
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully with no errors")
        sys.exit(0)


if __name__ == "__main__":
    main()
