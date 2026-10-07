"""Validate real V4 measurements, original-row preservation and every outer split."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from mipo.common import save_json
from mipo.corpus import read_corpus
from mipo.splits import make_splits, verify_fold

ROOT = Path('v4')
corpus = ROOT/'data/ndd_training_corpus_v4.csv.gz'
d = read_corpus(corpus)
references = json.loads((ROOT/'resources/sequences.json').read_text())
original = pd.read_csv('ndd_training_corpus_v3.csv', keep_default_na=False, low_memory=False, float_precision='round_trip')
precise = pd.read_csv(corpus, usecols=['v3_row_id', 'gene', 'wt', 'ref_pos', 'mut', 'score_value'], low_memory=False, float_precision='round_trip')
preserved = precise[precise.v3_row_id.notna()].sort_values('v3_row_id').reset_index(drop=True)
assert len(preserved) == len(original)
assert np.array_equal(preserved.v3_row_id.to_numpy(), np.arange(len(original)))
for column in ['gene', 'wt', 'ref_pos', 'mut', 'score_value']:
    assert np.array_equal(preserved[column].to_numpy(), original[column].to_numpy()), column
for ref_id, rows in d.groupby('protein_reference_id'):
    ref = references[ref_id]
    assert set(rows.gene) == {ref['gene']}
    assert all(0 < p <= len(ref['sequence']) and ref['sequence'][p-1] == w
               for p, w in rows[['ref_pos', 'wt']].drop_duplicates().itertuples(index=False, name=None))
assert not d.measurement_id.duplicated().any()
assert not d.duplicated(['assay_key', 'variant_key']).any()
assert not d.loc[d.taxon_id != 9606, 'is_direct_ndd'].astype(int).any()
assert np.isfinite(d.score_value).all()
assert d.default_sample_weight.between(0, 1, inclusive='right').all()
metadata = pd.read_csv(ROOT/'data/assay_metadata.csv', keep_default_na=False)
assert set(metadata.assay_key) == set(d.assay_key)
assert metadata.loaded_rows.sum() == len(d)
assert set(d.protein_reference_id) == set(references)
counts = {}
for name, file in [('curriculum', corpus), ('direct', ROOT/'data/ndd_direct_v4.csv.gz')]:
    frame = d if name == 'curriculum' else read_corpus(file)
    cluster = pd.read_csv(ROOT/'data/split_groups.csv')
    cluster = cluster[cluster.gene.isin(frame.gene)]
    cluster_path = ROOT/'splits'/(name+'_groups.csv')
    cluster_path.parent.mkdir(exist_ok=True)
    cluster.to_csv(cluster_path, index=False)
    split = make_splits(file, ROOT/'splits'/(name+'.json'), clusters=cluster_path,
                        test_genes=ROOT/'data/ndd_evaluation_genes.csv')
    targets = set(pd.read_csv(ROOT/'data/ndd_evaluation_genes.csv').gene)
    gene_only = frame[['gene']].drop_duplicates()
    sequences_by_gene = {gene: {references[r]['sequence'] for r in rows.protein_reference_id}
                         for gene, rows in frame[['gene', 'protein_reference_id']].drop_duplicates().groupby('gene')}
    for fold in split['folds']:
        verify_fold(gene_only, fold)
        assert set(fold['test']) & targets
        role_sequences = []
        for role in ['train', 'validation', 'calibration', 'test']:
            role_sequences.append(set().union(*(sequences_by_gene[gene] for gene in fold[role])))
        assert all(not role_sequences[i]&role_sequences[j] for i in range(4) for j in range(i))
    counts[name] = len(split['folds'])
q = d[d.task != 'consensus'].groupby(['gene', 'is_direct_ndd']).agg(rows=('gene', 'size'), tasks=('task', lambda x: '|'.join(sorted(set(x)))),
                                                                 n_tasks=('task', 'nunique'), assays=('assay_key', 'nunique'))
q[q.n_tasks > 1].to_csv(ROOT/'audit/multi_property_genes.csv')
missing_structures = [ref for ref, r in references.items() if not (ROOT/'resources/structures'/(r['key']+'.npz')).exists()]
report = {'passed': True, 'rows': len(d), 'original_scores_bitwise_equal': True, 'original_rows_preserved': len(original),
          'duplicate_measurements': 0, 'wt_reference_mismatches': 0, 'fold_counts': counts,
          'nonhuman_direct_ndd_rows': 0, 'reference_count': len(references),
          'structure_references_available_at_validation': len(references)-len(missing_structures),
          'missing_structure_references': missing_structures}
save_json(ROOT/'audit/validation.json', report)
print(json.dumps({k:v for k,v in report.items() if k != 'missing_structure_references'}, indent=2))
