"""Cache primary-paper bibliographic records and available Europe PMC full texts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import pandas as pd
import requests

out = Path('v4/sources/literature')
out.mkdir(parents=True, exist_ok=True)
metadata = pd.read_csv('v4/data/assay_metadata.csv').fillna('')
dois = sorted(set(metadata.loc[metadata.source == 'ProteinGym_v1.3', 'publication_doi']))

def fetch(doi):
    key = doi.replace('/', '_').replace(':', '_').rstrip('.')
    record_file = out/(key+'.json')
    if not record_file.exists():
        response = requests.get('https://www.ebi.ac.uk/europepmc/webservices/rest/search',
                                params={'query': f'DOI:"{doi.rstrip(chr(46))}"', 'format':'json'}, timeout=40)
        response.raise_for_status()
        record_file.write_text(json.dumps(response.json(), indent=2), encoding='utf-8')
    result = json.loads(record_file.read_text())['resultList']['result']
    pmcid = next((r.get('pmcid') for r in result if r.get('pmcid')), None)
    if pmcid:
        xml = out/(pmcid+'.xml')
        if not xml.exists():
            response = requests.get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML', timeout=40)
            if response.status_code == 200:
                xml.write_bytes(response.content)
        if xml.exists():
            tree = ET.fromstring(xml.read_bytes())
            paragraphs = [' '.join(''.join(p.itertext()).split()) for p in tree.iter() if p.tag in ['p','title']]
            (out/(pmcid+'.txt')).write_text('\n'.join(paragraphs), encoding='utf-8')
    return {'doi': doi, 'pmcid': pmcid, 'fulltext_cached': bool(pmcid and (out/(pmcid+'.txt')).exists())}

def safe(doi):
    try:
        return fetch(doi)
    except Exception as error:
        return {'doi':doi, 'error':str(error)}

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(safe, dois))
(out/'index.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
