import os

def test_pypi_harvester_files_exist():
    files = ["beautifulsoup4_readme.md", "langchain_readme.md", "numpy_readme.md", "pandas_readme.md"]
    for fname in files:
        path = os.path.join("data", "raw", "pypi_packages", fname)
        assert os.path.exists(path)
