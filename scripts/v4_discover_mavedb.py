"""Snapshot the published Human Domainome collection and selected NDD searches."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import requests
import pandas as pd

ROOT = Path('v4/sources/mavedb_search')
ROOT.mkdir(parents=True, exist_ok=True)
API = 'https://api.mavedb.org/api/v1'


def search(term, offset=0):
    path = ROOT/f'{term}_{offset}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    response = requests.post(API+'/score-sets/search', json={'text': term, 'limit': 100, 'offset': offset}, timeout=120)
    response.raise_for_status()
    path.write_text(response.text, encoding='utf-8')
    return response.json()


if __name__ == '__main__':
    first = search('domainome')
    offsets = list(range(100, first['numScoreSets'], 100))
    with ThreadPoolExecutor(max_workers=3) as pool:
        pages = [first]+list(pool.map(lambda offset: search('domainome', offset), offsets))
    records = {r['urn']: r for page in pages for r in page['scoreSets']}
    Path('v4/sources/domainome_index.json').write_text(json.dumps(list(records.values()), ensure_ascii=True), encoding='utf-8')
    print('Domainome score sets', len(records), flush=True)
    manifest = pd.read_csv('ndd_training_corpus_v3_manifest.csv')
    genes = sorted(set(manifest.loc[manifest.corpus_layer.str.contains('WATCHLIST'), 'gene']))
    with ThreadPoolExecutor(max_workers=3) as pool:
        found = list(pool.map(search, genes))
    report = []
    for gene, result in zip(genes, found):
        report.append({'gene': gene, 'search_hits': result['numScoreSets'], 'urns': ';'.join(x['urn'] for x in result['scoreSets']),
                       'status': 'candidate_requires_variant_and_scope_validation' if result['numScoreSets'] else 'no_published_keyword_search_hit'})
    pd.DataFrame(report).to_csv('v4/sources/watchlist_search.csv', index=False)
    print(pd.DataFrame(report)[['gene', 'search_hits']].to_string(index=False))
