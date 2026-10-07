"""Identify the original assay behind each collapsed-consensus target, by evidence.

The V3 corpus supplied 25 `gate_consensus` rows as opaque 0-1 targets with no constituent
assay identifiers. A rank-preserving rescale destroys values but not *order*, so a candidate
source can be tested directly: if a deposited score set covers the same variants and its
scores are a monotone function of the supplied targets, it is the source.

The test is Spearman rho on the overlapping variants. rho = 1.0 with high coverage is proof
of a rank-preserving relabel; anything less is reported, not assumed.

Read-only with respect to the corpus. Writes a provenance map for review.

Run:  python scripts/recover_consensus_provenance.py
Out:  v4/audit/consensus_provenance_map.csv  +  v4/sources/consensus_candidates/
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Bio.SeqUtils import seq1
from scipy.stats import spearmanr

CACHE = Path('v4/sources/consensus_candidates')
OUT = Path('v4/audit/consensus_provenance_map.csv')
API = 'https://api.mavedb.org/api/v1'
SESSION = requests.Session()
PRO = re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
# Offsets to try when the corpus reference and the deposited target use different numbering.
OFFSETS = range(-60, 61)


def search(gene):
    """Score sets whose target gene name matches exactly."""
    path = CACHE/f'search_{gene}.json'
    if not path.exists():
        response = SESSION.post(f'{API}/score-sets/search', json={'text': gene}, timeout=60)
        response.raise_for_status()
        path.write_text(json.dumps(response.json(), indent=1), encoding='utf-8')
        time.sleep(0.4)
    payload = json.loads(path.read_text(encoding='utf-8'))
    hits = []
    for record in payload.get('scoreSets', []):
        for target in record.get('targetGenes') or []:
            if str(target.get('name', '')).upper() == gene.upper():
                hits.append(record)
                break
    return hits


def scores(urn):
    """Deposited scores for one set, parsed to (wt, pos, mut, score)."""
    path = CACHE/(urn.replace(':', '_')+'.csv')
    if not path.exists():
        response = SESSION.get(f'{API}/score-sets/{urn}/scores', timeout=180)
        response.raise_for_status()
        path.write_text(response.text, encoding='utf-8')
        time.sleep(0.4)
    frame = pd.read_csv(path, low_memory=False)
    if 'hgvs_pro' not in frame or 'score' not in frame:
        return pd.DataFrame()
    parts = frame.hgvs_pro.fillna('').str.extract(PRO)
    frame = frame.assign(
        wt=parts[0].map(lambda x: seq1(x) if isinstance(x, str) and x else None),
        pos=pd.to_numeric(parts[1], errors='coerce'),
        mut=parts[2].map(lambda x: seq1(x) if isinstance(x, str) and x else None))
    frame = frame.dropna(subset=['wt', 'pos', 'mut', 'score'])
    frame['pos'] = frame.pos.astype(int)
    return frame[['wt', 'pos', 'mut', 'score']].drop_duplicates(['wt', 'pos', 'mut'])


def best_alignment(target, candidate):
    """Best (offset, overlap, rho) between supplied targets and a candidate score set."""
    best = (0, 0, float('nan'))
    for offset in OFFSETS:
        shifted = candidate.assign(ref_pos=candidate.pos+offset)
        merged = target.merge(shifted, on=['wt', 'ref_pos', 'mut'], how='inner')
        if len(merged) <= max(best[1], 20):
            continue
        rho = spearmanr(merged.score_value.astype(float), merged.score.astype(float)).statistic
        best = (offset, len(merged), float(rho))
        if best[1] == len(target) and abs(rho) > 0.999:
            break            # exact, complete cover: no better alignment exists
    return best


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv('v4/data/ndd_training_corpus_v4.csv.gz', low_memory=False,
                         usecols=['gene', 'assay_id', 'wt', 'ref_pos', 'mut', 'score_value', 'task'])
    consensus = corpus[corpus.task == 'consensus']
    rows = []
    for gene, target in consensus.groupby('gene'):
        target = target[['wt', 'ref_pos', 'mut', 'score_value']].drop_duplicates(['wt', 'ref_pos', 'mut'])
        try:
            hits = search(gene)
        except Exception as error:
            rows.append({'gene': gene, 'consensus_rows': len(target), 'status': 'search_failed',
                         'note': str(error)[:120]})
            continue
        if not hits:
            rows.append({'gene': gene, 'consensus_rows': len(target),
                         'status': 'no_matching_target_gene_in_mavedb'})
            continue
        found = []
        for record in hits:
            urn = record['urn']
            try:
                candidate = scores(urn)
            except Exception:
                continue
            if candidate.empty:
                continue
            offset, overlap, rho = best_alignment(target, candidate)
            if overlap >= 20:
                found.append({'urn': urn, 'title': record.get('title'), 'offset': offset,
                              'overlap': overlap, 'rho': rho,
                              'coverage': overlap/len(target),
                              'doi': ';'.join(p.get('doi') or '' for p in
                                              (record.get('primaryPublicationIdentifiers') or []))})
        if not found:
            rows.append({'gene': gene, 'consensus_rows': len(target),
                         'status': 'candidates_found_but_no_variant_overlap',
                         'note': f'{len(hits)} score sets searched'})
            continue
        found.sort(key=lambda r: (-(abs(r['rho']) if r['rho'] == r['rho'] else 0), -r['overlap']))
        for rank, record in enumerate(found[:4]):
            verdict = ('rank_identical_source' if abs(record['rho']) > 0.9999 and record['coverage'] > 0.5
                       else 'strong_partial' if abs(record['rho']) > 0.99
                       else 'weak_or_unrelated')
            rows.append({'gene': gene, 'consensus_rows': len(target), 'rank': rank,
                         'status': verdict, **record})
        print(f'{gene:<8} {len(target):>6} rows | best rho={found[0]["rho"]:.4f} '
              f'cov={found[0]["coverage"]:.0%} offset={found[0]["offset"]} {found[0]["urn"]}', flush=True)
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print('\nwritten:', OUT, len(frame), 'rows')


if __name__ == '__main__':
    main()
