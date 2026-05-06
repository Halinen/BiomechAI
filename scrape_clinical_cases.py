"""
scrape_clinical_cases.py — 临床决策流程 + 案例报告精准爬取

策略：
  1. 用 PubMed esearch（db=pubmed）加 publication type filter [pt] 搜 case report
     再用 elink 找对应的 PMC 全文 ID，最后 efetch 拿 XML
  2. 用精准 query + 领域限定词避免跑题（加 AND (physical therapy OR rehabilitation OR physiotherapy)）
  3. Protocol 类文章同样加领域限定词
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

# 领域限定词，防止跑题
DOMAIN_FILTER = " AND (physical therapy[tiab] OR physiotherapy[tiab] OR rehabilitation[tiab] OR manual therapy[tiab] OR exercise therapy[tiab])"


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


def fetch_pmc_and_save(pmcid_num, out_dir, saved_ids):
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


def pubmed_to_pmc_ids(pubmed_ids: list[str]) -> list[str]:
    """用 elink 把 PubMed ID 转换为 PMC ID（只有有全文的才返回）"""
    if not pubmed_ids:
        return []
    id_str = ",".join(pubmed_ids)
    url = (f"{EUTILS}/elink.fcgi?dbfrom=pubmed&db=pmc"
           f"&id={id_str}&retmode=json")
    try:
        data = session.get(url, timeout=15).json()
        pmc_ids = []
        for linkset in data.get("linksets", []):
            for lsd in linkset.get("linksetdbs", []):
                if lsd.get("linkname") == "pubmed_pmc":
                    pmc_ids.extend(str(i) for i in lsd.get("links", []))
        return pmc_ids
    except Exception as e:
        print(f"  elink error: {e}")
        return []


def search_case_reports(queries, out_dir, target=8):
    """通过 PubMed [pt] case report filter → elink → PMC 全文"""
    saved_ids = set()
    saved = 0

    for query in queries:
        if saved >= target:
            break
        full_query = f'({query}){DOMAIN_FILTER} AND "case reports"[pt]'
        search_url = (
            f"{EUTILS}/esearch.fcgi?db=pubmed"
            f"&term={requests.utils.quote(full_query)}"
            f"&retmax=15&retmode=json&sort=relevance"
        )
        try:
            data = session.get(search_url, timeout=15).json()
            pubmed_ids = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"  Search error: {e}")
            continue

        if not pubmed_ids:
            continue

        pmc_ids = pubmed_to_pmc_ids(pubmed_ids)
        time.sleep(0.3)

        for pmcid in pmc_ids:
            if saved >= target:
                break
            print(f"  [PMC{pmcid}] ", end='', flush=True)
            ok, msg = fetch_pmc_and_save(pmcid, out_dir, saved_ids)
            if ok:
                saved += 1
                safe_msg = msg[:55].encode('ascii', errors='replace').decode('ascii')
                print(f"Saved: {safe_msg}")
            else:
                print(f"Skip ({msg})")
            time.sleep(0.5)

    return saved


def search_protocols(queries, out_dir, target=8):
    """直接搜 PMC，query 含领域限定词，避免跑题"""
    saved_ids = set()
    saved = 0

    for query in queries:
        if saved >= target:
            break
        full_query = f"{query}{DOMAIN_FILTER}"
        search_url = (
            f"{EUTILS}/esearch.fcgi?db=pmc"
            f"&term={requests.utils.quote(full_query)}+AND+open+access[filter]"
            f"&retmax=10&retmode=json&sort=relevance"
        )
        try:
            data = session.get(search_url, timeout=15).json()
            ids = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"  Search error: {e}")
            continue

        for pmcid in ids:
            if saved >= target:
                break
            print(f"  [PMC{pmcid}] ", end='', flush=True)
            ok, msg = fetch_pmc_and_save(pmcid, out_dir, saved_ids)
            if ok:
                saved += 1
                safe_msg = msg[:55].encode('ascii', errors='replace').decode('ascii')
                print(f"Saved: {safe_msg}")
            else:
                print(f"Skip ({msg})")
            time.sleep(0.5)

    return saved


# ═══════════════════════════════════════════════════════════════
# Query 列表
# ═══════════════════════════════════════════════════════════════

# ── Visceral Fascia ──────────────────────────────────────────
VF_PROTOCOL_QUERIES = [
    'post-surgical abdominal breathing dysfunction assessment treatment',
    'visceral adhesion diaphragm function clinical evaluation',
    'abdominal surgery respiratory rehabilitation step protocol',
    'postoperative breathing exercise diaphragm retraining protocol',
    'scar tissue abdominal wall mobility assessment treatment',
]

VF_CASE_QUERIES = [
    'visceral manipulation abdominal adhesion chronic pain',
    'post-surgical abdominal adhesion musculoskeletal pain treatment',
    'abdominal scar restriction breathing dysfunction',
    'post-appendectomy chronic pain manual therapy',
    'viscerosomatic pain referral abdominal surgery',
    'peritoneal adhesion low back pain treatment outcome',
]

# ── Pain Neuroscience ────────────────────────────────────────
PN_PROTOCOL_QUERIES = [
    'central sensitization assessment clinical decision framework',
    'pain neuroscience education implementation protocol chronic pain',
    'graded exposure fear avoidance chronic pain step-by-step',
    'central sensitization inventory clinical use assessment',
    'neurodynamic assessment clinical protocol nerve tension',
]

PN_CASE_QUERIES = [
    'pain neuroscience education chronic musculoskeletal pain outcome',
    'central sensitization physiotherapy treatment outcome',
    'fear avoidance chronic low back pain graded exposure',
    'post-surgical chronic pain sensitization treatment',
    'neurodynamics nerve mobilization chronic pain outcome',
    'chronic abdominal pain central sensitization treatment',
]

# ── Breathing Retraining ─────────────────────────────────────
BR_PROTOCOL_QUERIES = [
    'diaphragm breathing dysfunction assessment clinical protocol',
    'breathing pattern disorder evaluation treatment steps',
    'DNS dynamic neuromuscular stabilization assessment clinical',
    'postural restoration breathing evaluation protocol',
    'respiratory muscle retraining chronic low back pain protocol',
]

BR_CASE_QUERIES = [
    'diaphragm breathing retraining chronic low back pain outcome',
    'DNS dynamic neuromuscular stabilization treatment outcome',
    'breathing pattern disorder physiotherapy treatment outcome',
    'asymmetric diaphragm function breathing retraining',
    'post-surgical breathing dysfunction rehabilitation outcome',
    'respiratory physiotherapy diaphragm dysfunction treatment',
]


if __name__ == "__main__":
    print("=== Clinical Protocol + Case Report Scraper (v2) ===\n")
    total = 0

    print("── Visceral_Fascia: protocols ──")
    n = search_protocols(VF_PROTOCOL_QUERIES, KB / "Visceral_Fascia", target=8)
    total += n; print(f"  +{n}\n")

    print("── Visceral_Fascia: case reports ──")
    n = search_case_reports(VF_CASE_QUERIES, KB / "Visceral_Fascia", target=8)
    total += n; print(f"  +{n}\n")

    print("── Pain_Neuroscience: protocols ──")
    n = search_protocols(PN_PROTOCOL_QUERIES, KB / "Pain_Neuroscience", target=8)
    total += n; print(f"  +{n}\n")

    print("── Pain_Neuroscience: case reports ──")
    n = search_case_reports(PN_CASE_QUERIES, KB / "Pain_Neuroscience", target=8)
    total += n; print(f"  +{n}\n")

    print("── Breathing_Retraining: protocols ──")
    n = search_protocols(BR_PROTOCOL_QUERIES, KB / "Breathing_Retraining", target=8)
    total += n; print(f"  +{n}\n")

    print("── Breathing_Retraining: case reports ──")
    n = search_case_reports(BR_CASE_QUERIES, KB / "Breathing_Retraining", target=8)
    total += n; print(f"  +{n}\n")

    print("=== Final Summary ===")
    for folder in ["Visceral_Fascia", "Pain_Neuroscience", "Breathing_Retraining"]:
        files = list((KB / folder).glob("*.txt"))
        kb = sum(f.stat().st_size for f in files) // 1024
        print(f"  {folder}: {len(files)} files, {kb}KB")
    print(f"\n  Total new this run: {total} papers")
