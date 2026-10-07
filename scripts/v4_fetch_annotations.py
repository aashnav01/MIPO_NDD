"""Fetch official gene/evidence mappings used by V4; no title-only gene guessing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import io
import json
import re
import requests
import pandas as pd

ROOT = Path('v4/sources')


def fetch_uniprot(batch):
    query = '('+' OR '.join('accession:'+a.split('-')[0] for a in batch)+')'
    response = requests.get('https://rest.uniprot.org/uniprotkb/search', params={
        'query': query, 'format': 'tsv', 'fields': 'accession,id,gene_primary,organism_name,sequence', 'size': 500}, timeout=120)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), sep='\t', keep_default_na=False)


if __name__ == '__main__':
    pgmap = pd.read_csv(ROOT/'uniprot_mappings.tsv', sep='\t', keep_default_na=False)
    ids = set(pgmap.loc[pgmap['Gene Names (primary)'] != '', 'Entry'])
    domainome = json.loads((ROOT/'domainome_index.json').read_text(encoding='utf-8'))
    for record in domainome:
        for target in record['targetGenes']:
            for ext in target['externalIdentifiers']:
                if ext['identifier']['dbName'] == 'UniProt':
                    ids.add(ext['identifier']['identifier'])
    ids = sorted(ids)
    with ThreadPoolExecutor(max_workers=3) as pool:
        frames = list(pool.map(fetch_uniprot, [ids[i:i+40] for i in range(0, len(ids), 40)]))
    result = pd.concat(frames, ignore_index=True).drop_duplicates('Entry')
    result.to_csv(ROOT/'uniprot_sequences.tsv', sep='\t', index=False)
    print('UniProt references', len(result), flush=True)
    base = 'https://ftp.ebi.ac.uk/pub/databases/gene2phenotype/G2P_data_downloads/2026_08_28/'
    response = requests.get(base, timeout=60)
    response.raise_for_status()
    links = re.findall(r'href="([^"]*DD[^"]*\.csv\.gz)"', response.text)
    if len(links) != 1:
        raise ValueError(f'Expected one DDG2P release file: {links}')
    response = requests.get(base+links[0], timeout=90)
    response.raise_for_status()
    (ROOT/links[0]).write_bytes(response.content)
    print('DDG2P', links[0], flush=True)
