import requests
from lxml import etree
from src.core.config import settings

def parse_data(xml_text: str) -> list[dict]:
    article_list = []
    root = etree.fromstring(xml_text.encode("utf-8"))

    articles = root.findall(".//PubmedArticle")

    # For each article, extract fields using findtext
    for article in articles:
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle")
        abstract = article.findtext(".//AbstractText")
        article_data = {"pmid": pmid, "title": title, "abstract": abstract}
        if article_data["abstract"] is not None:
            article_list.append(article_data)

    return article_list


def search_pubmed(query: str, max_results: int = 15) -> list[dict]:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    article_list = []

    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "api_key": settings.NCBI_API_KEY
    }

    try:
        response = requests.get(f"{base_url}esearch.fcgi", params=esearch_params)
        response.raise_for_status()
        pmids = response.json()["esearchresult"]["idlist"]

        pmids = ",".join(pmids)

        efetch_params = {
            "db": "pubmed",
            "id": pmids,
            "retmode": "XML",
            "api_key": settings.NCBI_API_KEY
        }

        raw_data = requests.get(f"{base_url}efetch.fcgi", params=efetch_params)
        raw_data.raise_for_status()

        article_list = parse_data(raw_data.text)

        return article_list
    except Exception as e:
        print(f"Error: {e}")


