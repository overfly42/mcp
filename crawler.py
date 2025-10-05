import os
import sys
import json
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import html2text


def save_progress(progress_file, data):
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_links(base_url, html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for tag in soup.find_all("a", href=True):
        url = urljoin(base_url, tag['href'])
        # Nur gleiche Domain
        if urlparse(url).netloc == urlparse(base_url).netloc:
            links.add(url.split('#')[0])  # Fragmente ignorieren
    return links


def download_and_convert(url, output_dir):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        # HTML zu Markdown
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        md_text = converter.handle(r.text)

        # Dateiname aus Pfad
        parsed = urlparse(url)
        safe_path = parsed.path.strip("/").replace("/", "_") or "index"
        filename = f"{parsed.netloc}_{safe_path}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_text)
        
        return r.text
    except Exception as e:
        print(f"Fehler beim Laden {url}: {e}")
        return None


def crawl(base_url, output_dir, progress_file):
    os.makedirs(output_dir, exist_ok=True)

    # Fortschritt laden oder initialisieren
    progress = load_progress(progress_file)
    if not progress:
        progress = {
            "base_url": base_url,
            "visited": [],
            "to_visit": [base_url]
        }

    while progress["to_visit"]:
        print(f"Noch zu besuchen: {len(progress['to_visit'])}")
        try:
            n = int(input("Wie viele Seiten als nächstes laden? (0 = abbrechen) "))
        except ValueError:
            print("Ungültige Eingabe.")
            continue

        if n <= 0:
            break

        for _ in range(min(n, len(progress["to_visit"]))):
            url = progress["to_visit"].pop(0)
            if url in progress["visited"]:
                continue
            if url.endswith('/print'):
                continue

            print(f"Lade: {url}")
            html = download_and_convert(url, output_dir)
            if html:
                links = get_links(url, html)
                for link in links:
                    if link not in progress["visited"] and link not in progress["to_visit"]:
                        progress["to_visit"].append(link)

            progress["visited"].append(url)
            save_progress(progress_file, progress)

    print("Beendet. Fortschritt gespeichert.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Verwendung: python crawler.py <base_url> <output_dir> [progress.json]")
        sys.exit(1)

    base_url = sys.argv[1]
    output_dir = sys.argv[2]
    progress_file = sys.argv[3] if len(sys.argv) > 3 else "progress.json"

    crawl(base_url, output_dir, progress_file)
