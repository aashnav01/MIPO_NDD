"""Build a small research handoff without measurements or full-text papers."""
from pathlib import Path
import json
import zipfile
import pandas as pd
from mipo.common import digest, save_json

root = Path(__file__).resolve().parents[1]
metadata = pd.read_csv(root/'v4/data/assay_metadata.csv', keep_default_na=False)
pending = []
for row in metadata.to_dict('records'):
    for field in ['host', 'selection', 'system', 'treatment', 'region']:
        if row[field] in ['', 'unknown']:
            pending.append({key: row.get(key, '') for key in
                            ['assay_key', 'assay_id', 'gene', 'source', 'is_direct_ndd', 'publication_doi',
                             'source_raw_phenotype', 'selection_assay', 'selection_type']} |
                           {'field': field, 'current_value': row[field],
                            'priority': 'original_provenance_required' if row['source'] == 'original_v3'
                            else 'NDD_first' if str(row['is_direct_ndd']) == '1' else 'support_assay'})
pd.DataFrame(pending).to_csv(root/'v4/audit/metadata_research_queue.csv', index=False)
files = ['CLAUDE_METADATA_HANDOFF.md', 'README_ndd_training_corpus_v3.txt',
         'v4/data/assay_metadata.csv', 'v4/data/protein_references.csv', 'v4/data/gene_evidence.csv',
         'v4/sources/proteingym_reference.csv', 'v4/sources/literature/index.json',
         'v4/audit/metadata_coverage.json', 'v4/audit/metadata_research_queue.csv',
         'v4/DATA_DICTIONARY.md', 'v4/DATASET_CARD.md', 'mipo/metadata_curation.py',
         'scripts/curate_v4_metadata.py', 'scripts/research_assay_sources.py',
         'scripts/build_v4_dataset.py', 'scripts/package_claude_handoff.py']
archive = root/'dist/claude_metadata_research_bundle.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(root/name, name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert 'CLAUDE_METADATA_HANDOFF.md' in z.namelist()
report = {'files': len(files), 'unresolved_field_entries': len(pending),
          'assays_with_unknowns': len({p['assay_key'] for p in pending}),
          'archive_bytes': archive.stat().st_size, 'sha256': digest(archive)}
save_json(root/'dist/claude_handoff_release.json', report)
print(json.dumps(report, indent=2))
