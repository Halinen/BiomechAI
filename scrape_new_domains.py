"""
Scraper for three new knowledge base domains:
  1. Visceral_Fascia   — Barral VM, post-surgical adhesion, viscerosomatic referral
  2. Pain_Neuroscience — Moseley PNE, central sensitization, Butler neurodynamics
  3. Breathing_Retraining — DNS, PRI diaphragm, PNF breathing, respiratory retraining

Uses the same PMC E-utils XML API pattern as existing scrapers.
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
    for e, c in [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&nbsp;', ' '),
                 ('&mdash;', '—'), ('&#39;', "'"), ('&rsquo;', "'"), ('&rdquo;', '"'), ('\xa0', ' ')]:
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
        except Exception as e:
            print(f"  Search error: {e}")
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


# ── 1. Visceral Fascia / Visceral Manipulation ───────────────────────────────
VISCERAL_QUERIES = [
    'visceral manipulation manual therapy abdominal adhesion',
    'viscerosomatic referral abdominal organ musculoskeletal pain',
    'peritoneal adhesion postoperative breathing diaphragm',
    'visceral fascia organ mobility respiratory mechanics',
    'abdominal adhesion chronic pain musculoskeletal rehabilitation',
    'visceral peritoneum fascial restriction body wall',
    'post-surgical abdominal adhesion lumbar spine dysfunction',
    'mesenteric restriction diaphragm breathing compensation',
    'osteopathic visceral technique organ mobility evidence',
    'visceral afferent sensitization chronic abdominal pain',
]

# ── 2. Pain Neuroscience / Central Sensitization ─────────────────────────────
PAIN_NEURO_QUERIES = [
    '"pain neuroscience education" chronic musculoskeletal pain',
    '"central sensitization" musculoskeletal pain mechanism',
    'Moseley pain neuroscience education randomized controlled trial',
    '"central sensitization" fear avoidance catastrophizing treatment',
    'neurodynamics neural mobilization Butler nerve tension',
    'pain catastrophizing kinesiophobia chronic low back pain',
    '"explain pain" education intervention chronic pain',
    'central sensitization inventory chronic pain rehabilitation',
    'descending pain modulation central sensitization spinal cord',
    'nociceptive sensitization dorsal horn wind-up mechanism',
    'graded motor imagery pain neuroscience chronic pain',
    'pain education neuroscience rehabilitation systematic review',
]

# ── 3. Breathing Retraining / DNS / PRI diaphragm ────────────────────────────
BREATHING_QUERIES = [
    '"Dynamic Neuromuscular Stabilization" DNS breathing diaphragm',
    'diaphragm dysfunction breathing pattern disorder rehabilitation',
    'breathing retraining respiratory muscle training chronic pain',
    '"postural restoration" diaphragm breathing asymmetry treatment',
    'dysfunctional breathing pattern musculoskeletal pain exercise',
    'respiratory retraining diaphragm activation lumbar stability',
    'breathing pattern disorder assessment treatment physiotherapy',
    'diaphragm intraabdominal pressure postural stability',
    'nasal breathing mouth breathing musculoskeletal performance',
    'low load breathing exercise chronic low back pain',
    'respiratory physiotherapy breathing exercise systematic review',
    'thoracic breathing accessory muscle overactivation treatment',
]


if __name__ == "__main__":
    print("=== New Domain PMC Scraper ===\n")

    print("── 1. Visceral Fascia ──")
    n1 = search_and_fetch(VISCERAL_QUERIES, KB / "Visceral_Fascia", target=12)
    print(f"Visceral_Fascia: +{n1} papers\n")

    print("── 2. Pain Neuroscience ──")
    n2 = search_and_fetch(PAIN_NEURO_QUERIES, KB / "Pain_Neuroscience", target=14)
    print(f"Pain_Neuroscience: +{n2} papers\n")

    print("── 3. Breathing Retraining ──")
    n3 = search_and_fetch(BREATHING_QUERIES, KB / "Breathing_Retraining", target=12)
    print(f"Breathing_Retraining: +{n3} papers\n")

    print("=== Final count ===")
    for folder in ["Visceral_Fascia", "Pain_Neuroscience", "Breathing_Retraining"]:
        files = list((KB / folder).glob("*.txt"))
        kb = sum(f.stat().st_size for f in files) // 1024
        print(f"  {folder}: {len(files)} files, {kb}KB")
