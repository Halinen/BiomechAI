"""
Multi-source scraper for FMS/SFMA, NASM CES, PRI knowledge base.
Sources:
  1. NASM Blog (blog.nasm.org) - corrective exercise, movement assessment
  2. PubMed Central (PMC) - open-access research papers via E-utils XML API
  3. FMS (functionalmovement.com) - movement screening articles
"""

import requests
import time
import re
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

session = requests.Session()
session.headers.update(HEADERS)

KB = Path("e:/Fitting/knowledge_base")
for d in ["FMS_SFMA", "NASM_CES", "PRI"]:
    (KB / d).mkdir(parents=True, exist_ok=True)


def strip_tags(html: str) -> str:
    html = re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<[^>]+>', ' ', html)
    for ent, ch in [('&amp;','&'),('&lt;','<'),('&gt;','>'),('&nbsp;',' '),
                    ('&mdash;','—'),('&ndash;','–'),('&#39;',"'"),('&rsquo;',"'"),
                    ('&rdquo;','"'),('&hellip;','...'),('\xa0',' ')]:
        html = html.replace(ent, ch)
    return re.sub(r'[ \t]{2,}', ' ', html).strip()


NOISE = {'subscribe', 'newsletter', 'sign in', 'log in', 'cookie', 'privacy policy',
         'terms of use', 'filter by', 'view all', 'load more', 'related articles',
         'share this', 'follow us', 'copyright', 'all rights reserved'}


def extract_blocks(html: str, min_len: int = 60) -> list[str]:
    html = re.sub(r'<(script|style|nav|footer|header|aside|noscript)[^>]*>.*?</\1>',
                  '', html, flags=re.DOTALL | re.IGNORECASE)
    blocks = re.findall(r'<(h2|h3|h4|p|li)([^>]*)>(.*?)</\1>', html, re.DOTALL | re.IGNORECASE)
    lines = []
    for tag, _, content in blocks:
        text = strip_tags(content)
        if len(text) < min_len:
            continue
        if any(n in text.lower() for n in NOISE):
            continue
        tag = tag.lower()
        prefix = '\n## ' if tag in ('h2','h3','h4') else ('- ' if tag == 'li' else '')
        lines.append(prefix + text)
    return lines


def safe_filename(title: str, max_len: int = 80) -> str:
    s = re.sub(r'[^\w\s-]', '', title).strip()
    return re.sub(r'[\s-]+', '_', s)[:max_len] + '.txt'


def save_doc(out_dir: Path, title: str, url: str, lines: list[str]) -> bool:
    body = '\n'.join(lines).strip()
    if len(body) < 400:
        return False
    (out_dir / safe_filename(title)).write_text(
        f"# {title}\nSource: {url}\n\n{body}", encoding='utf-8')
    return True


# ─────────────────────────────────────────────
# SOURCE 1: NASM Blog
# ─────────────────────────────────────────────

NASM_TAGS = [
    "corrective-exercise",
    "ces",
    "muscle-imbalances",
    "posture",
    "movement-assessment",
    "flexibility",
    "injury-prevention",
]

def collect_nasm_links(target: int) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    for tag in NASM_TAGS:
        page = 1
        while len(links) < target:
            url = f"https://blog.nasm.org/tag/{tag}" + (f"/page/{page}" if page > 1 else "")
            try:
                r = session.get(url, timeout=15)
                if r.status_code != 200:
                    break
            except:
                break
            # NASM article URLs: blog.nasm.org/{slug} — just one path segment
            found = re.findall(
                r'href="(https://blog\.nasm\.org/[a-z0-9][a-z0-9\-]{4,60})"',
                r.text
            )
            found = [f for f in found if not any(x in f for x in
                     ['/tag/', '/page/', '/author/', 'hubfs', '.css', '.js', '.ico', 'twitter', 'google', 'facebook'])]
            new = [f for f in found if f not in seen]
            if not new:
                break
            for f in new:
                seen.add(f)
                links.append(f)
            page += 1
            time.sleep(0.4)
        if len(links) >= target:
            break
    return list(dict.fromkeys(links))[:target]


def scrape_nasm(target: int = 40) -> int:
    out_dir = KB / "NASM_CES"
    print(f"\n[NASM] Collecting links (target={target})...")
    links = collect_nasm_links(target)
    print(f"[NASM] Found {len(links)} unique article links")

    saved = 0
    for i, url in enumerate(links, 1):
        slug = url.rstrip('/').split('/')[-1]
        print(f"  [NASM {i}/{len(links)}] {slug}")
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                print(f"    → HTTP {r.status_code}")
                continue
        except Exception as e:
            print(f"    → Error: {e}")
            continue
        h1 = re.search(r'<h1[^>]*>(.*?)</h1>', r.text, re.DOTALL | re.IGNORECASE)
        title = strip_tags(h1.group(1)) if h1 else slug.replace('-', ' ').title()
        lines = extract_blocks(r.text)
        if save_doc(out_dir, title, url, lines):
            saved += 1
            print(f"    → Saved: {title[:65]}")
        else:
            print("    → Skipped (too short)")
        time.sleep(0.6)

    print(f"[NASM] Saved {saved} articles")
    return saved


# ─────────────────────────────────────────────
# SOURCE 2: PubMed Central (XML API)
# ─────────────────────────────────────────────

PMC_QUERIES = {
    "FMS_SFMA": [
        "Functional Movement Screen reliability validity screening",
        "SFMA Selective Functional Movement Assessment clinical",
        "movement screen score injury musculoskeletal",
        "FMS corrective exercise movement quality",
    ],
    "NASM_CES": [
        "corrective exercise muscle imbalance rehabilitation",
        "postural dysfunction corrective exercise program",
        "lower crossed syndrome hip flexor muscle imbalance",
        "upper crossed syndrome forward head posture corrective",
    ],
    "PRI": [
        "postural restoration breathing mechanics asymmetry pelvis",
        "diaphragm breathing postural control lumbar",
        "pelvic asymmetry respiratory exercise",
        "ribcage breathing pattern neuromuscular rehabilitation",
    ],
}

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def parse_pmc_xml(xml: str) -> list[str]:
    """Extract text from PMC XML (NLM format)."""
    lines = []
    # Title
    title_m = re.search(r'<article-title>(.*?)</article-title>', xml, re.DOTALL)
    if title_m:
        lines.append(f"# {strip_tags(title_m.group(1))}")

    # Abstract
    abstract = re.search(r'<abstract>(.*?)</abstract>', xml, re.DOTALL | re.IGNORECASE)
    if abstract:
        lines.append('\n## Abstract')
        for p in re.findall(r'<p>(.*?)</p>', abstract.group(1), re.DOTALL):
            text = strip_tags(p)
            if len(text) > 40:
                lines.append(text)

    # Body sections
    body = re.search(r'<body>(.*?)</body>', xml, re.DOTALL | re.IGNORECASE)
    if body:
        # Section titles
        sections = re.split(r'<sec[^>]*>', body.group(1))
        for sec in sections[1:]:
            title_m2 = re.search(r'<title>(.*?)</title>', sec, re.DOTALL)
            if title_m2:
                lines.append(f'\n## {strip_tags(title_m2.group(1))}')
            for p in re.findall(r'<p>(.*?)</p>', sec, re.DOTALL):
                text = strip_tags(p)
                if len(text) > 40:
                    lines.append(text)

    return lines


def scrape_pmc(per_category: int = 10) -> int:
    total = 0

    for category, queries in PMC_QUERIES.items():
        out_dir = KB / category
        saved_ids: set[str] = set()
        saved = 0
        print(f"\n[PMC/{category}]")

        for query in queries:
            if saved >= per_category:
                break
            # Search PMC
            search_url = (
                f"{EUTILS}/esearch.fcgi?db=pmc&term={requests.utils.quote(query)}"
                f"+AND+open+access[filter]&retmax=8&retmode=json&sort=relevance"
            )
            try:
                data = session.get(search_url, timeout=15).json()
                ids = data.get("esearchresult", {}).get("idlist", [])
            except:
                continue

            for pmcid_num in ids:
                if saved >= per_category or pmcid_num in saved_ids:
                    continue
                saved_ids.add(pmcid_num)

                # Fetch XML full text
                fetch_url = f"{EUTILS}/efetch.fcgi?db=pmc&id={pmcid_num}&rettype=full&retmode=xml"
                try:
                    r = session.get(fetch_url, timeout=20)
                    xml = r.text
                except:
                    continue

                lines = parse_pmc_xml(xml)
                if not lines:
                    continue

                title_m = re.search(r'<article-title>(.*?)</article-title>', xml, re.DOTALL)
                title = strip_tags(title_m.group(1)) if title_m else f"PMC{pmcid_num}"

                art_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid_num}/"
                body_text = '\n'.join(lines[1:]).strip()  # exclude title line for save
                if len(body_text) >= 400:
                    (out_dir / safe_filename(title)).write_text(
                        f"# {title}\nSource: {art_url}\n\n{body_text}", encoding='utf-8')
                    saved += 1
                    total += 1
                    print(f"  [PMC{pmcid_num}] Saved: {title[:60]}")
                else:
                    print(f"  [PMC{pmcid_num}] Skipped (short body)")
                time.sleep(0.8)

        print(f"[PMC/{category}] Saved {saved} papers")

    return total


# ─────────────────────────────────────────────
# SOURCE 3: FMS (functionalmovement.com)
# ─────────────────────────────────────────────

def scrape_fms(target: int = 15) -> int:
    out_dir = KB / "FMS_SFMA"
    base = "https://www.functionalmovement.com"
    seen: set[str] = set()
    links: list[str] = []

    print("\n[FMS] Collecting article links...")
    for path in ["/articles", "/learn", "/resources/articles", ""]:
        try:
            r = session.get(base + path, timeout=15)
            if r.status_code != 200:
                continue
            found = re.findall(
                r'href="(/(?:articles|blog|learn|resources|posts)/[a-z0-9\-_/]{5,})"',
                r.text
            )
            for f in found:
                full = base + f.rstrip('/')
                if full not in seen:
                    seen.add(full)
                    links.append(full)
        except:
            continue
        time.sleep(0.4)

    # Also try Gray Cook's site
    gray_cook_urls = [
        "https://graycook.com/articles/",
        "https://graycook.com/resources/",
    ]
    for url in gray_cook_urls:
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                continue
            found = re.findall(r'href="(https://graycook\.com/[a-z0-9\-/]+)"', r.text)
            for f in found:
                if f not in seen and f.count('/') >= 4:
                    seen.add(f)
                    links.append(f)
        except:
            continue

    links = list(dict.fromkeys(links))[:target]
    print(f"[FMS] Found {len(links)} links")

    if not links:
        print("[FMS] No links found - site may require JS. Skipping.")
        return 0

    saved = 0
    for i, url in enumerate(links, 1):
        slug = url.rstrip('/').split('/')[-1]
        print(f"  [FMS {i}/{len(links)}] {slug}")
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                continue
        except Exception as e:
            print(f"    → Error: {e}")
            continue
        h1 = re.search(r'<h1[^>]*>(.*?)</h1>', r.text, re.DOTALL | re.IGNORECASE)
        title = strip_tags(h1.group(1)) if h1 else slug.replace('-', ' ').title()
        lines = extract_blocks(r.text)
        if save_doc(out_dir, title, url, lines):
            saved += 1
            print(f"    → Saved: {title[:65]}")
        else:
            print("    → Skipped")
        time.sleep(0.7)

    print(f"[FMS] Saved {saved} articles")
    return saved


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Knowledge Base Builder ===")
    print("Sources: NASM Blog | PubMed Central | FMS\n")

    n1 = scrape_nasm(target=40)
    n2 = scrape_pmc(per_category=10)
    n3 = scrape_fms(target=15)

    print(f"\n{'='*45}")
    print(f"Total: NASM={n1} | PMC={n2} | FMS={n3} | Sum={n1+n2+n3}")
    for folder in ["FMS_SFMA", "NASM_CES", "PRI"]:
        files = list((KB / folder).glob("*.txt"))
        total_chars = sum(f.stat().st_size for f in files)
        print(f"  {folder}: {len(files)} files, {total_chars//1024}KB")
