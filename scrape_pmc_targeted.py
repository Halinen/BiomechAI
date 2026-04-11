"""
Targeted PMC scraper with exact-phrase queries for FMS/SFMA and PRI.
"""

import requests
import time
import re
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
session = requests.Session()
session.headers.update(HEADERS)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
KB = Path("e:/Fitting/knowledge_base")


def strip_tags(html):
    html = re.sub(r'<[^>]+>', ' ', html)
    for e, c in [('&amp;','&'),('&lt;','<'),('&gt;','>'),('&nbsp;',' '),
                 ('&mdash;','—'),('&#39;',"'"),('&rsquo;',"'"),('&rdquo;','"'),('\xa0',' ')]:
        html = html.replace(e, c)
    return re.sub(r'[ \t]{2,}', ' ', html).strip()


def safe_filename(title, max_len=80):
    s = re.sub(r'[^\w\s-]', '', title).strip()
    return re.sub(r'[\s-]+', '_', s)[:max_len] + '.txt'


def parse_pmc_xml(xml):
    lines = []
    abstract = re.search(r'<abstract>(.*?)</abstract>', xml, re.DOTALL | re.IGNORECASE)
    if abstract:
        lines.append('## Abstract')
        for p in re.findall(r'<p>(.*?)</p>', abstract.group(1), re.DOTALL):
            t = strip_tags(p)
            if len(t) > 40:
                lines.append(t)
    body = re.search(r'<body>(.*?)</body>', xml, re.DOTALL | re.IGNORECASE)
    if body:
        for sec in re.split(r'<sec[^>]*>', body.group(1))[1:]:
            sec_title = re.search(r'<title>(.*?)</title>', sec, re.DOTALL)
            if sec_title:
                lines.append(f'\n## {strip_tags(sec_title.group(1))}')
            for p in re.findall(r'<p>(.*?)</p>', sec, re.DOTALL):
                t = strip_tags(p)
                if len(t) > 40:
                    lines.append(t)
    return lines


def fetch_and_save(pmcid_num, out_dir, saved_ids):
    if pmcid_num in saved_ids:
        return False, "duplicate"
    saved_ids.add(pmcid_num)

    fetch_url = f"{EUTILS}/efetch.fcgi?db=pmc&id={pmcid_num}&rettype=full&retmode=xml"
    try:
        r = session.get(fetch_url, timeout=20)
        xml = r.text
    except Exception as e:
        return False, f"fetch error: {e}"

    title_m = re.search(r'<article-title>(.*?)</article-title>', xml, re.DOTALL)
    title = strip_tags(title_m.group(1)) if title_m else f"PMC{pmcid_num}"

    lines = parse_pmc_xml(xml)
    body = '\n'.join(lines).strip()
    if len(body) < 400:
        return False, "too short"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / safe_filename(title)).write_text(
        f"# {title}\nSource: https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid_num}/\n\n{body}",
        encoding='utf-8')
    return True, title


def search_and_fetch(queries, out_dir, target=12):
    saved_ids = set()
    saved = 0
    for query in queries:
        if saved >= target:
            break
        search_url = (
            f"{EUTILS}/esearch.fcgi?db=pmc"
            f"&term={requests.utils.quote(query)}+AND+open+access[filter]"
            f"&retmax=10&retmode=json&sort=relevance"
        )
        try:
            data = session.get(search_url, timeout=15).json()
            ids = data.get("esearchresult", {}).get("idlist", [])
        except:
            continue

        for pmcid_num in ids:
            if saved >= target:
                break
            print(f"  [PMC{pmcid_num}] ", end='', flush=True)
            ok, msg = fetch_and_save(pmcid_num, out_dir, saved_ids)
            if ok:
                saved += 1
                safe_msg = msg[:55].encode('ascii', errors='replace').decode('ascii')
                print(f"Saved: {safe_msg}")
            else:
                print(f"Skip ({msg})")
            time.sleep(0.5)
    return saved


# ── FMS / SFMA ──────────────────────────────────────────
FMS_QUERIES = [
    '"Functional Movement Screen" reliability athletes',
    '"Functional Movement Screen" corrective exercise injury',
    '"FMS" movement quality screening musculoskeletal',
    '"Selective Functional Movement Assessment" SFMA clinical',
    '"SFMA" movement dysfunction rehabilitation',
    '"Gray Cook" movement screen assessment',
    'movement screening overhead squat assessment athletes',
    '"FMS score" injury prevention prediction',
]

# ── PRI ──────────────────────────────────────────────────
PRI_QUERIES = [
    '"Postural Restoration Institute" breathing asymmetry',
    'zone of apposition diaphragm breathing mechanics posture',
    'respiratory diaphragm postural control lumbar spine',
    'pelvic asymmetry breathing rehabilitation exercise',
    'left AIC right BC postural pattern',
    'rib cage position breathing neuromuscular control',
    'diaphragm postural stability breathing pattern disorder',
    'hip asymmetry breathing mechanics rehabilitation',
]

# ── NASM CES supplement ──────────────────────────────────
NASM_SUPPLEMENT_QUERIES = [
    '"upper crossed syndrome" corrective exercise intervention',
    '"lower crossed syndrome" hip flexor muscle imbalance',
    'movement compensation pattern corrective exercise',
    '"overhead squat assessment" movement dysfunction',
    'muscle inhibition activation corrective exercise',
    'forward head posture corrective exercise cervical',
]


if __name__ == "__main__":
    print("=== Targeted PMC Scraper ===\n")

    print("── FMS/SFMA ──")
    n1 = search_and_fetch(FMS_QUERIES, KB / "FMS_SFMA", target=12)
    print(f"FMS/SFMA: +{n1} papers\n")

    print("── PRI ──")
    n2 = search_and_fetch(PRI_QUERIES, KB / "PRI", target=12)
    print(f"PRI: +{n2} papers\n")

    print("── NASM CES supplement ──")
    n3 = search_and_fetch(NASM_SUPPLEMENT_QUERIES, KB / "NASM_CES", target=10)
    print(f"NASM_CES: +{n3} papers\n")

    print("=== Final count ===")
    for folder in ["FMS_SFMA", "NASM_CES", "PRI"]:
        files = list((KB / folder).glob("*.txt"))
        kb = sum(f.stat().st_size for f in files) // 1024
        print(f"  {folder}: {len(files)} files, {kb}KB")
