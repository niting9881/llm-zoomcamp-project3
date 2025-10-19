import requests

def test_api_docs_search():
    response = requests.get("http://127.0.0.1:8000/api/docs/search", params={"query": "What is BeautifulSoup?"})
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "answer" in data["result"]
    assert "sources" in data["result"]
