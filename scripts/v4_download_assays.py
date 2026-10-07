"""Download Domainome score tables and exact sequence metadata; cache each response."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time
import requests

ROOT = Path('v4/sources/mavedb')
ROOT.mkdir(parents=True, exist_ok=True)
API = 'https://api.mavedb.org/api/v1/score-sets/'


def fetch(record):
    urn = record['urn']
    stem = urn.replace(':', '_')
    for suffix, url in [('.json', API+urn), ('.csv', API+urn+'/scores')]:
        path = ROOT/(stem+suffix)
        if path.exists():
            continue
        for attempt in range(4):
            r = requests.get(url, timeout=(15, 90))
            if r.status_code in [429, 500, 502, 503, 504]:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            tmp = path.with_suffix(path.suffix+'.part')
            tmp.write_bytes(r.content)
            tmp.replace(path)
            break
        else:
            r.raise_for_status()
    return urn


if __name__ == '__main__':
    records = json.loads(Path('v4/sources/domainome_index.json').read_text(encoding='utf-8'))
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, record): record['urn'] for record in records}
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
            except (requests.RequestException, OSError) as error:
                failures.append({'urn': futures[future], 'error': str(error)})
            if (i+1) % 25 == 0 or i+1 == len(records):
                print(f'{i+1}/{len(records)} completed; failures={len(failures)}', flush=True)
    Path('v4/sources/download_failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} downloads failed; rerun resumes cached responses')
