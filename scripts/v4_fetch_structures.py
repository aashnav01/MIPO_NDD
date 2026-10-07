"""Exact reference crops from public AlphaFold models; no approximate alignment.

Run after build_v4_dataset.py. Missing or ambiguous references remain masked.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time

import numpy as np
import requests

from mipo.common import digest, save_json
from mipo.resources import download, structure_arrays

ROOT = Path('v4')


def fetch_group(item):
    accession, group = item
    results = {}
    folder = ROOT/'resources/structures'
    source = ROOT/'sources/alphafold'
    source.mkdir(exist_ok=True)
    pending = [(ref_id, rec) for ref_id, rec in group if not (folder/(rec['key']+'.npz')).exists()]
    for ref_id, rec in group:
        if (folder/(rec['key']+'.npz')).exists():
            results[ref_id] = {'status': 'exact_reference_cache', 'source': 'existing_validated_cache'}
    if not pending:
        return results
    try:
        pdb = source/(accession+'.pdb')
        metadata_path = source/(accession+'.json')
        if not pdb.exists():
            info = download('https://alphafold.ebi.ac.uk/api/prediction/'+accession).json()
            if not info:
                raise ValueError('No AlphaFold record')
            selected = info[0]
            response = download(selected['pdbUrl'])
            pdb.write_bytes(response.content)
            save_json(metadata_path, {'url': selected['pdbUrl'], 'sha256': digest(pdb), 'download_date': '2026-09-19'})
        meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {'url': 'existing_source', 'sha256': digest(pdb)}
        for ref_id, rec in pending:
            try:
                xyz, conf = structure_arrays(pdb, rec['sequence'], confidence='plddt', allow_subsequence=True)
                np.savez_compressed(folder/(rec['key']+'.npz'), coords=xyz, confidence=conf, sequence=np.array(rec['sequence']))
                results[ref_id] = {'status': 'exact_unique_sequence_match_or_crop', **meta}
            except ValueError as error:
                results[ref_id] = {'status': 'sequence_only', 'reason': str(error), **meta}
    except (requests.RequestException, ValueError, KeyError, IndexError) as error:
        for ref_id, rec in pending:
            results[ref_id] = {'status': 'sequence_only', 'reason': str(error)}
    return results


if __name__ == '__main__':
    sequences = json.loads((ROOT/'resources/sequences.json').read_text())
    groups = {}
    for ref_id, rec in sequences.items():
        groups.setdefault(rec['accession'], []).append((ref_id, rec))
    report = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_group, item) for item in groups.items()]
        for i, future in enumerate(as_completed(futures)):
            report.update(future.result())
            if (i+1) % 25 == 0 or i+1 == len(futures):
                save_json(ROOT/'resources/structures/provenance.json', report)
                print(f'{i+1}/{len(futures)} accession groups; {sum(x["status"] != "sequence_only" for x in report.values())} structure references ready', flush=True)
    save_json(ROOT/'resources/structures/provenance.json', report)
