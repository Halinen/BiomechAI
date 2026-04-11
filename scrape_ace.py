"""
Scrape ACE Fitness expert articles filtered by Posture, Corrective Exercise, Muscle Imbalance tags.
Saves up to 50 articles as .txt files into knowledge_base/NASM_CES/
"""

import requests
import time
import re
from pathlib import Path

OUTPUT_DIR = Path("e:/Fitting/knowledge_base/NASM_CES")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.acefitness.org"
TAGS = ["posture", "corrective-exercise", "muscle-imbalance"]
TARGET = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)


def strip_tags(html: str) -> str:
    html = re.sub(r'<[^>]+>', '', html)
    html = html.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>') \
               .replace('&nbsp;', ' ').replace('&mdash;', '—').replace('&ndash;', '–') \
               .replace('&#39;', "'").replace('&rsquo;', "'").replace('&rdquo;', '"') \
               .replace('&hellip;', '...').replace('\xa0', ' ')
    return re.sub(r'[ \t]{2,}', ' ', html).strip()


def get_article_links() -> list[str]:
    """Collect unique article URLs from tag-filtered listing pages."""
    seen: set[str] = set()
    links: list[str] = []

    for tag in TAGS:
        page = 1
        while len(links) < TARGET:
            url = f"{BASE_URL}/resources/pros/expert-articles/?tag={tag}&page={page}"
            print(f"  Listing [{tag} p{page}]: {url}")
            try:
                r = session.get(url, timeout=15)
                if r.status_code != 200:
                    break
            except Exception as e:
                print(f"  Error: {e}")
                break

            # Article URLs: /resources/pros/expert-articles/{id}/{slug}/
            found = re.findall(
                r'href="(/resources/pros/expert-articles/\d+/[a-z0-9\-]+/)"',
                r.text
            )
            new = []
            for f in found:
                if f not in seen:
                    seen.add(f)
                    new.append(BASE_URL + f)

            if not new:
                break

            links.extend(new)
            page += 1
            time.sleep(0.4)

        if len(links) >= TARGET:
            break

    return links[:TARGET]


def fetch_article(url: str) -> tuple[str, str] | None:
    """Fetch one article, return (title, clean_text) or None."""
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

    html = r.text

    # Title from <h1>
    h1 = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL | re.IGNORECASE)
    title = strip_tags(h1.group(1)) if h1 else url.split('/')[-2].replace('-', ' ').title()

    # Extract headings and paragraphs preserving structure
    lines = []

    # Remove nav/header/footer/script/style noise first
    html_clean = re.sub(
        r'<(script|style|nav|header|footer|aside|noscript)[^>]*>.*?</\1>',
        '', html, flags=re.DOTALL | re.IGNORECASE
    )

    # Extract h2, h3, p, li blocks
    blocks = re.findall(
        r'<(h2|h3|p|li)([^>]*)>(.*?)</\1>',
        html_clean, re.DOTALL | re.IGNORECASE
    )

    for tag, _, content in blocks:
        text = strip_tags(content)
        if len(text) < 40:
            continue
        # Skip obvious nav/UI strings
        if any(kw in text.lower() for kw in ['subscribe', 'newsletter', 'sign in', 'log in',
                                               'cookie', 'privacy policy', 'terms of use',
                                               'filter by', 'view all', 'load more']):
            continue
        if tag.lower() in ('h2', 'h3'):
            lines.append(f'\n## {text}')
        elif tag.lower() == 'li':
            lines.append(f'- {text}')
        else:
            lines.append(text)

    body = '\n'.join(lines).strip()

    # Need at least 500 chars of real content
    if len(body) < 500:
        return None

    return title, f"# {title}\nSource: {url}\n\n{body}"


def safe_filename(title: str) -> str:
    s = re.sub(r'[^\w\s-]', '', title).strip()
    s = re.sub(r'[\s-]+', '_', s)
    return s[:80] + '.txt'


def main():
    print("=== ACE Fitness Article Scraper ===")
    print(f"Target: {TARGET} articles | Tags: {TAGS}\n")

    print("Step 1: Collecting article links...")
    links = get_article_links()
    print(f"Found {len(links)} unique article links\n")

    print("Step 2: Fetching articles...")
    saved = 0
    for i, url in enumerate(links, 1):
        slug = url.split('/')[-2]
        print(f"  [{i}/{len(links)}] {slug}")
        result = fetch_article(url)
        if result is None:
            print("    → Skipped")
            continue
        title, content = result
        fname = safe_filename(title)
        out_path = OUTPUT_DIR / fname
        out_path.write_text(content, encoding='utf-8')
        saved += 1
        print(f"    → Saved ({len(content)} chars): {fname}")
        time.sleep(0.7)

    print(f"\nDone! Saved {saved} articles to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
