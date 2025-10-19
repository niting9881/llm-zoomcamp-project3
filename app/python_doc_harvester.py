import requests
from bs4 import BeautifulSoup
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class PythonDocHarvester:
    """
    Harvests Python official documentation from docs.python.org.
    Supports multiple Python versions (3.11, 3.12, 3.14).
    """
    
    def __init__(self, version: str):
        self.version = version
        self.base_url = f"https://docs.python.org/{version}/"
        self.output_dir = Path(f"data/raw/python_docs/{version}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.harvested_count = 0
        self.error_count = 0

    def harvest_documentation(self, max_modules: int = 50):
        """
        Harvest Python documentation modules.
        
        Args:
            max_modules: Maximum number of modules to harvest (default: 50)
        
        Returns:
            dict: Statistics about the harvest
        """
        logger.info(f"Starting harvest for Python {self.version}")
        
        try:
            toc_url = f"{self.base_url}library/index.html"
            logger.info(f"Fetching table of contents from: {toc_url}")
            
            response = requests.get(toc_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all module links
            modules = soup.find_all('a', class_='reference internal')
            logger.info(f"Found {len(modules)} module links")
            
            # Filter for actual library modules (ending in .html)
            module_links = [m for m in modules if m.get('href', '').endswith('.html')]
            logger.info(f"Found {len(module_links)} actual module pages")
            
            # Limit number of modules
            module_links = module_links[:max_modules]
            
            for idx, module in enumerate(module_links, 1):
                try:
                    href = module.get('href', '')
                    if not href:
                        continue
                    
                    # Build full URL
                    if href.startswith('http'):
                        module_url = href
                    elif href.startswith('/'):
                        module_url = f"https://docs.python.org{href}"
                    else:
                        module_url = f"{self.base_url}{href}"
                    
                    # Extract module name
                    module_name = href.split('/')[-1].replace('.html', '')
                    module_name = ''.join(c for c in module_name if c.isalnum() or c in ('_', '-'))
                    
                    if not module_name:
                        continue
                    
                    logger.info(f"[{idx}/{len(module_links)}] Harvesting: {module_name}")
                    
                    self._save_module_doc(module_url, module_name)
                    self.harvested_count += 1
                    
                    # Be nice to the server
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error harvesting module {href}: {e}")
                    self.error_count += 1
                    continue
            
            logger.info(f"Harvest complete for Python {self.version}")
            logger.info(f"  Harvested: {self.harvested_count} modules")
            logger.info(f"  Errors: {self.error_count}")
            
            return {
                "version": self.version,
                "harvested": self.harvested_count,
                "errors": self.error_count,
                "output_dir": str(self.output_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to harvest Python {self.version} documentation: {e}")
            return {
                "version": self.version,
                "harvested": 0,
                "errors": 1,
                "error": str(e)
            }

    def _save_module_doc(self, url: str, module_name: str):
        """Save a single module documentation page."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            doc_file = self.output_dir / f"{module_name}.html"
            doc_file.write_text(response.text, encoding="utf-8")
            
            logger.debug(f"Saved: {doc_file}")
            
        except Exception as e:
            logger.error(f"Failed to save {module_name}: {e}")
            raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    versions = ["3.14", "3.12", "3.11"]
    all_results = []
    
    print("=" * 80)
    print("Python Documentation Harvester")
    print("=" * 80)
    
    for version in versions:
        print(f"\n📚 Harvesting Python {version} documentation...")
        harvester = PythonDocHarvester(version)
        result = harvester.harvest_documentation(max_modules=50)
        all_results.append(result)
        
        print(f"✅ Python {version}: {result['harvested']} modules harvested")
        if result.get('errors', 0) > 0:
            print(f"⚠️  {result['errors']} errors encountered")
    
    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)
    total_harvested = sum(r['harvested'] for r in all_results)
    total_errors = sum(r.get('errors', 0) for r in all_results)
    print(f"Total modules harvested: {total_harvested}")
    print(f"Total errors: {total_errors}")
    print("=" * 80)
