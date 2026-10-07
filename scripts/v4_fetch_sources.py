"""Download immutable local source snapshots for the V4 corpus build."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import requests

ROOT = Path('v4/sources')
ROOT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'proteingym_v1.3.zip': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip',
    'panelapp_285.json': 'https://panelapp.genomicsengland.co.uk/api/v1/panels/285/',
    'mavedb_openapi.json': 'https://api.mavedb.org/openapi.json',
    'proteingym_reference.csv': 'https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv',
}


def fetch(item):
    name, url = item
    path = ROOT/name
    if path.exists():
        return f'{name}: cached {path.stat().st_size} bytes'
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        temp = path.with_suffix('.part')
        with open(temp, 'wb') as f:
            for chunk in response.iter_content(1024*1024):
                f.write(chunk)
        temp.replace(path)
    return f'{name}: downloaded {path.stat().st_size} bytes'


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, SOURCES.items()):
            print(result, flush=True)
    (ROOT/'source_urls.json').write_text(json.dumps(SOURCES, indent=2))
