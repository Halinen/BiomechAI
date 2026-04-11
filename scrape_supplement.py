"""
Supplement scraper:
  1. Red flags / referral criteria — PMC + ACE targeted slugs
  2. PRI — official website content + more specific PMC papers
"""

import requests, time, re
from pathlib import Path

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
KB = Path("e:/Fitting/knowledge_base")
KB_RED = KB / "Red_Flags"
KB_PRI = KB / "PRI"
KB_RED.mkdir(exist_ok=True)


def strip(html):
    html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    for e, c in [("&amp;","&"),("&lt;","<"),("&gt;",">"),("&nbsp;"," "),("&mdash;","—"),
                 ("&#8220;",'"'),("&#8221;",'"'),("&#8216;","'"),("&#8217;","'"),
                 ("&#39;","'"),("&rsquo;","'"),("&rdquo;",'"'),("&hellip;","..."),("\xa0"," "),("\xae","(R)")]:
        html = html.replace(e, c)
    return re.sub(r"[ \t]{2,}", " ", html).strip()


NOISE = {"subscribe","newsletter","sign in","log in","cookie","privacy policy","terms of use",
         "view all","load more","share this","follow us","copyright","all rights reserved"}


def extract_blocks(html, min_len=60):
    html = re.sub(r"<(script|style|nav|footer|header|aside|noscript)[^>]*>.*?</\1>",
                  "", html, flags=re.DOTALL | re.IGNORECASE)
    lines = []
    for tag, _, content in re.findall(r"<(h2|h3|h4|p|li)([^>]*)>(.*?)</\1>", html, re.DOTALL | re.IGNORECASE):
        t = strip(content)
        if len(t) < min_len or any(n in t.lower() for n in NOISE):
            continue
        prefix = "\n## " if tag.lower() in ("h2","h3","h4") else ("- " if tag.lower()=="li" else "")
        lines.append(prefix + t)
    return lines


def fname(title, maxlen=80):
    s = re.sub(r"[^\w\s-]", "", title).strip()
    return re.sub(r"[\s-]+", "_", s)[:maxlen] + ".txt"


def save(out_dir, title, url, lines):
    body = "\n".join(lines).strip()
    if len(body) < 300:
        return False
    (out_dir / fname(title)).write_text(f"# {title}\nSource: {url}\n\n{body}", encoding="utf-8")
    return True


# ── PMC helper ──────────────────────────────────────────────────────────────

def pmc_parse_xml(xml):
    lines = []
    ab = re.search(r"<abstract>(.*?)</abstract>", xml, re.DOTALL | re.IGNORECASE)
    if ab:
        lines.append("## Abstract")
        for p in re.findall(r"<p>(.*?)</p>", ab.group(1), re.DOTALL):
            t = strip(p)
            if len(t) > 40: lines.append(t)
    body = re.search(r"<body>(.*?)</body>", xml, re.DOTALL | re.IGNORECASE)
    if body:
        for sec in re.split(r"<sec[^>]*>", body.group(1))[1:]:
            stitle = re.search(r"<title>(.*?)</title>", sec, re.DOTALL)
            if stitle: lines.append(f"\n## {strip(stitle.group(1))}")
            for p in re.findall(r"<p>(.*?)</p>", sec, re.DOTALL):
                t = strip(p)
                if len(t) > 40: lines.append(t)
    return lines


def pmc_fetch_batch(queries, out_dir, target, saved_ids=None):
    if saved_ids is None:
        saved_ids = set()
    saved = 0
    for q in queries:
        if saved >= target: break
        url = (f"{EUTILS}/esearch.fcgi?db=pmc"
               f"&term={requests.utils.quote(q)}+AND+open+access[filter]"
               f"&retmax=10&retmode=json&sort=relevance")
        try:
            ids = S.get(url, timeout=15).json().get("esearchresult", {}).get("idlist", [])
        except: continue
        for pid in ids:
            if saved >= target or pid in saved_ids: continue
            saved_ids.add(pid)
            try:
                xml = S.get(f"{EUTILS}/efetch.fcgi?db=pmc&id={pid}&rettype=full&retmode=xml",
                            timeout=20).text
            except: continue
            tm = re.search(r"<article-title>(.*?)</article-title>", xml, re.DOTALL)
            title = strip(tm.group(1)) if tm else f"PMC{pid}"
            lines = pmc_parse_xml(xml)
            body = "\n".join(lines).strip()
            if len(body) < 400:
                print(f"  [PMC{pid}] skip (short)")
                continue
            (out_dir / fname(title)).write_text(
                f"# {title}\nSource: https://pmc.ncbi.nlm.nih.gov/articles/PMC{pid}/\n\n{body}",
                encoding="utf-8")
            saved += 1
            safe = title[:55].encode("ascii", errors="replace").decode()
            print(f"  [PMC{pid}] Saved: {safe}")
            time.sleep(0.6)
    return saved


# ── 1. RED FLAGS / REFERRAL CRITERIA ────────────────────────────────────────

RED_PMC = [
    '"red flags" musculoskeletal exercise professional referral',
    'red flags contraindications exercise assessment referral medical',
    '"scope of practice" fitness trainer referral medical conditions',
    'musculoskeletal red flags low back pain exercise referral',
    'absolute contraindications exercise screening PAR-Q',
    'exercise preparticipation screening red flags cardiovascular',
    'when to refer exercise professional physician specialist',
    '"absolute contraindications" exercise resistance training',
]

# ACE specific article slugs known to cover red flags / scope of practice
ACE_RED_SLUGS = [
    "https://www.acefitness.org/resources/pros/expert-articles/5497/red-flags-a-trainers-guide-to-knowing-when-to-refer/",
    "https://www.acefitness.org/resources/pros/expert-articles/5496/musculoskeletal-red-flags-movement-assessment/",
    "https://www.acefitness.org/resources/pros/expert-articles/6751/when-to-refer-a-client/",
    "https://www.acefitness.org/resources/pros/expert-articles/6011/scope-of-practice-for-fitness-professionals/",
    "https://www.acefitness.org/resources/pros/expert-articles/5807/contraindications-for-exercise/",
    "https://www.acefitness.org/resources/pros/expert-articles/7402/red-flags-in-movement-assessment/",
    "https://www.acefitness.org/resources/pros/expert-articles/7219/recognizing-red-flags-during-fitness-assessments/",
]


def scrape_ace_url(url):
    try:
        r = S.get(url, timeout=15)
        if r.status_code != 200: return None
    except: return None
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", r.text, re.DOTALL | re.IGNORECASE)
    title = strip(h1.group(1)) if h1 else url.split("/")[-2].replace("-"," ").title()
    lines = extract_blocks(r.text)
    return (title, lines) if lines else None


def scrape_red_flags():
    print("\n=== RED FLAGS ===")
    saved = 0

    print("[ACE] Trying known article URLs...")
    for url in ACE_RED_SLUGS:
        slug = url.rstrip("/").split("/")[-1]
        result = scrape_ace_url(url)
        if result:
            title, lines = result
            if save(KB_RED, title, url, lines):
                saved += 1
                print(f"  Saved: {title[:65]}")
            else:
                print(f"  Skip (short): {slug}")
        else:
            # Try to find via listing search
            print(f"  404: {slug}")
        time.sleep(0.5)

    # Also search ACE listing for red-flag related articles
    print("[ACE] Searching listing pages...")
    for page in range(1, 8):
        url = f"https://www.acefitness.org/resources/pros/expert-articles/?page={page}"
        try:
            r = S.get(url, timeout=15)
            links = re.findall(r'href="(/resources/pros/expert-articles/\d+/[a-z0-9\-]+/)"', r.text)
            titles_raw = re.findall(r'<h[23][^>]*>(.*?)</h[23]>', r.text, re.DOTALL)
            for link in links:
                title_text = link.split("/")[-2].replace("-"," ").lower()
                if any(kw in title_text for kw in ["red flag","referral","scope","contraindic","when to refer","medical clear"]):
                    full = "https://www.acefitness.org" + link
                    result = scrape_ace_url(full)
                    if result:
                        t, lines = result
                        if save(KB_RED, t, full, lines):
                            saved += 1
                            print(f"  Found+Saved: {t[:65]}")
        except: pass
        time.sleep(0.4)
        if not links: break

    print(f"[PMC] Fetching red flag papers...")
    saved += pmc_fetch_batch(RED_PMC, KB_RED, target=8)
    print(f"Red flags total: {saved} docs")
    return saved


# ── 2. PRI SUPPLEMENT ────────────────────────────────────────────────────────

PRI_PMC = [
    '"Postural Restoration" breathing asymmetry pelvis exercise',
    '"zone of apposition" diaphragm breathing posture',
    '"left AIC" OR "right BC" postural pattern asymmetry',
    'diaphragm respiratory postural stability lumbar spine pelvis',
    'respiratory muscle training postural control imbalance',
    '"thoraco-pelvic" breathing mechanical asymmetry',
    'rib cage mobility breathing exercise rehabilitation dysfunction',
    'femoral acetabular impingement breathing pattern hip asymmetry',
    'hamstring inhibition breathing pattern pelvic position',
    '"respiratory diaphragm" position lumbar pelvic stabilization',
]

PRI_WEBSITE_URLS = [
    "https://www.posturalrestoration.com/the-science/",
    "https://www.posturalrestoration.com/faqs/",
    "https://www.posturalrestoration.com/community/pri-conversations-by-bill-hartman/",
    "https://www.posturalrestoration.com/community/blog-entry-shared-from-integrative-human-performance/",
    "https://www.posturalrestoration.com/community/part-two-blog-by-peter-nelson-at-integrative-human-performance/",
]


def scrape_pri_website():
    print("[PRI Website] Scraping official content...")
    saved = 0
    for url in PRI_WEBSITE_URLS:
        slug = url.rstrip("/").split("/")[-1]
        try:
            r = S.get(url, timeout=15)
            if r.status_code != 200:
                print(f"  404: {slug}")
                continue
        except Exception as e:
            print(f"  Error {slug}: {e}")
            continue
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", r.text, re.DOTALL | re.IGNORECASE)
        title = strip(h1.group(1)) if h1 else slug.replace("-"," ").title()
        if not title or len(title) < 3:
            title = slug.replace("-"," ").title()
        lines = extract_blocks(r.text, min_len=50)
        if save(KB_PRI, title, url, lines):
            saved += 1
            print(f"  Saved: {title[:65]}")
        else:
            print(f"  Skip (short): {slug}")
        time.sleep(0.6)
    return saved


def scrape_pri():
    print("\n=== PRI SUPPLEMENT ===")
    n1 = scrape_pri_website()
    print(f"[PMC] Fetching PRI-specific papers...")
    n2 = pmc_fetch_batch(PRI_PMC, KB_PRI, target=15)
    print(f"PRI total new: website={n1}, PMC={n2}")
    return n1 + n2


# ── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n1 = scrape_red_flags()
    n2 = scrape_pri()

    print("\n=== FINAL KNOWLEDGE BASE ===")
    for folder in ["FMS_SFMA", "NASM_CES", "PRI", "Red_Flags"]:
        files = list((KB / folder).glob("*.txt"))
        kb = sum(f.stat().st_size for f in files) // 1024
        print(f"  {folder:15s}: {len(files):3d} files  {kb:5d}KB")
