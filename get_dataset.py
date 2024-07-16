import time
import pandas as pd
from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract, extract_metadata

# https://pub.towardsai.net/create-a-chatbot-in-python-with-langchain-and-rag-85bfba8c62d2

def get_urls_from_sitemap(resource_url: str) -> list:
    """
    Recovers the sitemap through Trafilatura
    """
    urls = sitemap_search(resource_url)
    return urls


def create_dataset(list_of_websites: list) -> pd.DataFrame:
    """
    Function that creates a Pandas DataFrame of URLs and articles.
    """
    data = []
    for website in tqdm(list_of_websites, desc="Websites"):
        urls = get_urls_from_sitemap(website)
        for url in tqdm(urls, desc="URLs"):
            html = fetch_url(url)
            body = extract(html)
            try:
                metadata = extract_metadata(html)
                title = metadata.title
                description = metadata.description
            except:
                metadata = ""
                title = ""
                description = ""
            d = {
                'url': url,
                "body": body,
                "title": title,
                "description": description
            }
            data.append(d)
            time.sleep(0.5)
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    df = df.dropna()

    return df


if __name__ == "__main__":
    list_of_websites = [
        "https://python.langchain.com/"
    ]
    df = create_dataset(list_of_websites)
    df.to_csv("./data/dataset.csv", index=False)
