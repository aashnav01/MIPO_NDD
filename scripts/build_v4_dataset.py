"""Reproducibly assemble the NDD-focused V4 measurement corpus from snapshots.

One row is one observed single-AA substitution in one assay. References and
assays are normalized tables. Original V3 rows remain recoverable by v3_row_id.
"""
from __future__ import annotations

import json
import hashlib
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils import seq1

from mipo.common import AA, TYPE_MAP, digest, save_json, seq_key
from mipo.corpus import read_corpus
from mipo.metadata_curation import apply_overrides, default_status, enrich_metadata

ROOT = Path('v4')
SRC = ROOT/'sources'
DATA = ROOT/'data'
RES = ROOT/'resources'
AUDIT = ROOT/'audit'
DATE = '2026-09-19'
PG_URL = 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip'
PG_TYPE = {'Activity': 'activity', 'Binding': 'binding', 'Expression': 'abundance',
           'Stability': 'stability', 'OrganismalFitness': 'organismal_fitness'}
HPO_NDD = {'HP:0001249', 'HP:0001263', 'HP:0000750', 'HP:0000717', 'HP:0012758'}
# Explicit family/biological-neighborhood teachers inherited from the V3 plan.
# This is a modeling relevance annotation, not clinical NDD evidence.
TEACHERS = {
    'CALM1': 'calcium_signaling', 'SCN5A': 'voltage_gated_sodium', 'KCNH2': 'voltage_gated_potassium',
    'KCNJ2': 'inward_rectifier_potassium', 'SLC6A4': 'neurotransmitter_transporter', 'SLC22A1': 'multi_pass_transporter',
    'ADRB2': 'gpcr', 'EPHB2': 'synaptic_receptor_rtk', 'SRC': 'kinase', 'MET': 'receptor_tyrosine_kinase',
    'UBE4B': 'ubiquitin_proteostasis', 'HECTD1': 'ubiquitin_proteostasis', 'NPC1': 'lysosomal_membrane_trafficking',
    'KCNE1': 'potassium_channel_auxiliary', 'KCNQ1': 'voltage_gated_potassium',
}


def evidence_table():
    panel = json.loads((SRC/'panelapp_285.json').read_text(encoding='utf-8'))
    dd = pd.read_csv(SRC/'DDG2P_2026-08-28.csv.gz', keep_default_na=False)
    panel_by_gene = {x['gene_data']['gene_symbol']: x for x in panel['genes']}
    records = {}
    for gene in set(panel_by_gene) | set(dd['gene symbol']):
        p = panel_by_gene.get(gene, {})
        associations = dd[dd['gene symbol'] == gene]
        relevant = []
        for _, row in associations.iterrows():
            hpo = set(str(row['phenotypes']).replace(' ', '').split(';'))
            named = re.search(r'neurodevelopment|intellectual|developmental delay|encephalopath', row['disease name'], re.I)
            if row.confidence in ['definitive', 'strong', 'moderate'] and (hpo & HPO_NDD or named):
                relevant.append(row.to_dict())
        green = p.get('confidence_level') == '3'
        records[gene] = {'gene': gene, 'ndd_verified': int(green or bool(relevant)),
                         'evidence_basis': 'PanelApp285_green_and_DDG2P_NDD' if green and relevant else 'PanelApp285_green' if green else 'DDG2P_NDD_phenotype' if relevant else 'not_verified_by_selected_rule',
                         'panelapp_confidence': p.get('confidence_level', ''), 'panelapp_version': panel['version'],
                         'panelapp_phenotypes': '|'.join(p.get('phenotypes', [])),
                         'inheritance': '|'.join(sorted(set([p.get('mode_of_inheritance', '')]+[r['allelic requirement'] for r in relevant]))).strip('|'),
                         'ddg2p_ids': '|'.join(r['g2p id'] for r in relevant),
                         'ddg2p_diseases': '|'.join(r['disease name'] for r in relevant),
                         'mechanisms': '|'.join(sorted({r['molecular mechanism'] for r in relevant})),
                         'evidence_date': DATE}
    return records


def tier(gene, taxon, task, evidence):
    if taxon == 9606 and evidence.get(gene, {}).get('ndd_verified'):
        return 'C_DIRECT_NDD_STABILITY' if task == 'stability' else 'B_DIRECT_NDD_FUNCTIONAL'
    if gene in TEACHERS:
        return 'D_NDD_FAMILY_TEACHER'
    if taxon == 9606:
        return 'E_GENERIC_HUMAN_DMS'
    raise ValueError(f'Nonhuman gene outside explicit teacher list: {gene}')


class Builder:
    def __init__(self):
        for path in [DATA, RES, AUDIT, RES/'structures']:
            path.mkdir(parents=True, exist_ok=True)
        self.evidence = evidence_table()
        self.uniprot = pd.read_csv(SRC/'uniprot_sequences.tsv', sep='\t', keep_default_na=False).set_index('Entry')
        mapping = pd.read_csv(SRC/'uniprot_mappings.tsv', sep='\t', keep_default_na=False)
        mapping = mapping[mapping['Gene Names (primary)'] != '']
        self.pgmap = mapping.set_index('Entry Name').to_dict('index')
        self.references, self.assays, self.frames, self.qc, self.excluded = {}, {}, [], [], []
        self.old = pd.read_csv('ndd_training_corpus_v3.csv', keep_default_na=False, low_memory=False, float_precision='round_trip')
        self.legacy_references = json.loads(Path('prepared/resources/sequences.json').read_text(encoding='utf-8'))

    def reference(self, gene, sequence, taxon, accession='', offset=None, source=''):
        if not sequence or not set(sequence) <= set(AA):
            raise ValueError('Reference contains noncanonical or missing amino acids')
        key = seq_key(sequence)
        ref_id = f'{gene}__{taxon}__{key[:12]}'
        if ref_id not in self.references:
            canonical_offset = None
            canonical = self.uniprot.loc[accession, 'Sequence'] if accession in self.uniprot.index else ''
            if canonical and canonical.count(sequence) == 1:
                canonical_offset = canonical.index(sequence)
            self.references[ref_id] = {'gene': gene, 'protein_reference_id': ref_id, 'sequence': sequence, 'key': key,
                                       'length': len(sequence), 'taxon_id': taxon, 'accession': accession, 'source': source,
                                       'reported_offset': offset, 'canonical_offset': canonical_offset,
                                       'canonical_mapping_status': 'exact_unique_subsequence' if canonical_offset is not None else 'not_established'}
        return ref_id

    def assay(self, gene, assay_id, ref_id, assay_type, score_kind, supervision_tier, **kwargs):
        key = gene+'::'+assay_id
        if key not in self.assays:
            record = self.references[ref_id]
            self.assays[key] = {'gene': gene, 'assay_id': assay_id, 'assay_key': key, 'protein_reference_id': ref_id,
                                'task': TYPE_MAP[assay_type], 'assay_type': assay_type, 'score_kind': score_kind,
                                'supervision_tier': supervision_tier, 'taxon_id': record['taxon_id'],
                                'host': 'unknown', 'selection': 'unknown', 'system': 'unknown', 'treatment': 'unknown',
                                'region': 'unknown', 'orientation': 'as_supplied', 'score_direction': 'as_supplied_unknown_biological_direction',
                                'is_direct_ndd': int(record['taxon_id'] == 9606 and self.evidence.get(gene, {}).get('ndd_verified', 0)),
                                'ndd_evidence_class': self.evidence.get(gene, {}).get('evidence_basis', 'not_verified_by_selected_rule'),
                                'protein_group': TEACHERS.get(gene, 'unassigned'),
                                'mechanism_relevance': 'measurement_is_not_a_pathogenicity_label', **kwargs}
        else:
            if self.assays[key]['protein_reference_id'] != ref_id:
                raise ValueError(f'{key}: existing measurement and new source use different references')
            self.assays[key].update(kwargs)
        return key

    def attach(self, frame, key, source, source_rows):
        a = self.assays[key]
        f = frame.copy()
        for name in ['gene', 'assay_id', 'protein_reference_id', 'assay_type', 'score_kind', 'supervision_tier',
                     'is_direct_ndd', 'ndd_evidence_class', 'protein_group']:
            f[name] = a[name]
        f['assay_key'] = key
        f['task'] = a['task']
        f['taxon_id'] = a['taxon_id']
        f['species'] = 'Homo sapiens' if a['taxon_id'] == 9606 else 'Mus musculus'
        f['source'] = source
        f['source_row_id'] = source_rows
        f['assay_level_available'] = int(a['task'] != 'consensus')
        f['score_orientation'] = a['score_direction']
        f['measurement_type'] = a.get('measurement_type', a['task'])
        f['measurement_id'] = [seq_key(source+'|'+key+'|'+str(row)) for row in source_rows]
        ref = self.references[a['protein_reference_id']]
        f['canonical_pos'] = f.ref_pos+ref['canonical_offset'] if ref['canonical_offset'] is not None else pd.NA
        f['position_system'] = 'experimental_reference_1_based'
        f['training_role'] = {'A_LEGACY_CONSENSUS': 'legacy_consensus_auxiliary', 'B_DIRECT_NDD_FUNCTIONAL': 'primary_ndd_functional',
                              'C_DIRECT_NDD_STABILITY': 'primary_ndd_stability_or_proxy', 'D_NDD_FAMILY_TEACHER': 'family_teacher',
                              'E_GENERIC_HUMAN_DMS': 'generic_pretraining'}[a['supervision_tier']]
        f['score_value'] = pd.to_numeric(f.score_value, errors='raise')
        f['ref_pos'] = f.ref_pos.astype(int)
        seq = ref['sequence']
        good = np.asarray([1 <= p <= len(seq) and seq[p-1] == wt for p, wt in zip(f.ref_pos, f.wt)])
        if not good.all():
            raise ValueError(f'{key}: {int((~good).sum())} WT/reference mismatches')
        if not np.isfinite(f.score_value).all():
            raise ValueError('Nonfinite score passed filtering')
        self.frames.append(f)

    def legacy(self):
        old = self.old.copy()
        old['v3_row_id'] = np.arange(len(old))
        for col in ['supervision_tier', 'is_direct_ndd', 'ndd_evidence_class', 'train_target_0_1']:
            old['legacy_'+col] = old[col]
        for (gene, assay_id), frame in old.groupby(['gene', 'assay_id'], sort=False):
            original = self.legacy_references[gene]
            ref_id = self.reference(gene, original['sequence'], 9606, original.get('accession', ''), source=original['source'])
            assay_type = frame.assay_type.iloc[0]
            task = TYPE_MAP[assay_type]
            t = 'A_LEGACY_CONSENSUS' if task == 'consensus' else tier(gene, 9606, task, self.evidence)
            kind = frame.score_kind.iloc[0]
            key = self.assay(gene, assay_id, ref_id, assay_type, kind, t, source='original_v3',
                             source_url='user_supplied:ndd_training_corpus_v3.csv', source_sha256=digest('ndd_training_corpus_v3.csv'),
                             measurement_type='collapsed_consensus' if task == 'consensus' else task,
                             publication_doi='', license='original_dataset_terms_unprovided')
            frame = frame.drop(columns=['train_target_0_1'])
            self.attach(frame, key, 'original_v3', frame.v3_row_id.to_numpy())
        print('Preserved original rows:', len(old), flush=True)

    def proteingym(self):
        reference = pd.read_csv(SRC/'proteingym_reference.csv', keep_default_na=False)
        selected = reference[(reference.source_organism == 'Homo sapiens') | reference.UniProt_ID.isin(['KCNJ2_MOUSE', 'UBE4B_MOUSE'])]
        with zipfile.ZipFile(SRC/'proteingym_v1.3.zip') as z:
            members = {Path(name).name: name for name in z.namelist() if name.endswith('.csv')}
            for r in selected.itertuples():
                mapping = self.pgmap[r.UniProt_ID]
                gene = mapping['Gene Names (primary)'].upper()
                taxon = 9606 if r.source_organism == 'Homo sapiens' else 10090
                ref_id = self.reference(gene, r.target_seq, taxon, mapping['Entry'], source='ProteinGym v1.3 experimental target_seq')
                assay_type = PG_TYPE[r.coarse_selection_type]
                task = TYPE_MAP[assay_type]
                t = tier(gene, taxon, task, self.evidence)
                raw = pd.read_csv(z.open(members[r.DMS_filename]), float_precision='round_trip')
                raw['_source_index'] = np.arange(len(raw))
                single = raw.mutant.astype(str).str.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY][1-9][0-9]*[ACDEFGHIKLMNPQRSTVWY]')
                finite = pd.to_numeric(raw.DMS_score, errors='coerce').map(np.isfinite)
                f = raw.loc[single & finite].copy()
                parts = f.mutant.str.extract(r'([A-Z])(\d+)([A-Z])')
                f['wt'], f['ref_pos'], f['mut'] = parts[0], parts[1].astype(int), parts[2]
                f = f[f.wt != f.mut]
                # Check the full supplied mutant sequence, not only the WT letter.
                if 'mutated_sequence' in f:
                    verified = [s == r.target_seq[:p-1]+m+r.target_seq[p:] for s, p, m in zip(f.mutated_sequence, f.ref_pos, f.mut)]
                    if not all(verified):
                        raise ValueError(f'{r.DMS_id}: supplied mutant/reference sequence disagreement')
                key = self.assay(gene, r.DMS_id, ref_id, assay_type, 'ProteinGym_DMS_score', t,
                                 source='ProteinGym_v1.3', source_url=PG_URL, source_file=r.DMS_filename,
                                 source_sha256=hashlib.sha256(z.read(members[r.DMS_filename])).hexdigest(), publication_doi=r.jo,
                                 publication_title=r.title, publication_year=r.year, selection_assay=r.selection_assay,
                                 selection_type=r.selection_type, region='domain' if len(r.target_seq) < 100 else 'assay_reference',
                                 score_direction='higher_source_assay_phenotype_not_health', orientation='ProteinGym_oriented',
                                 measurement_type='proteolysis_stability' if 'Tsuboyama' in r.DMS_id else 'aggregation_proxy' if 'Seuma' in r.DMS_id else task,
                                 source_raw_phenotype=r.raw_DMS_phenotype_name, source_raw_directionality=r.raw_DMS_directionality,
                                 license='ProteinGym_repository_and_original_study_terms')
                # Upgrade source details for the seven already-present ProteinGym assays.
                self.assays[key].update(source='ProteinGym_v1.3', source_url=PG_URL, publication_doi=r.jo,
                                         selection_assay=r.selection_assay, selection_type=r.selection_type)
                existing = self.old[(self.old.gene == gene) & (self.old.assay_id == r.DMS_id)]
                overlap = set(zip(existing.wt, existing.ref_pos, existing.mut))
                duplicate_mask = np.asarray([(w, p, m) in overlap for w, p, m in zip(f.wt, f.ref_pos, f.mut)])
                if duplicate_mask.any():
                    comparison = f.loc[duplicate_mask].merge(existing[['wt', 'ref_pos', 'mut', 'score_value']], on=['wt', 'ref_pos', 'mut'])
                    if not np.allclose(comparison.DMS_score, comparison.score_value.astype(float), rtol=1e-9, atol=1e-10):
                        raise ValueError(f'{r.DMS_id}: conflicting score for an existing measurement')
                self.qc.append({'source': 'ProteinGym_v1.3', 'assay_id': r.DMS_id, 'source_rows': len(raw),
                                'single_finite_rows': len(f), 'already_in_v3': int(duplicate_mask.sum()),
                                'excluded_non_single_or_nonfinite': int(len(raw)-len(f)), 'added_rows': int((~duplicate_mask).sum())})
                f = f.loc[~duplicate_mask]
                columns = pd.DataFrame({'wt': f.wt, 'ref_pos': f.ref_pos, 'mut': f.mut,
                                        'score_value': f.DMS_score, 'raw_dms_score': f.DMS_score,
                                        'dms_score_bin_source': f.DMS_score_bin if 'DMS_score_bin' in f else pd.NA,
                                        'source_variant': f.mutant})
                if len(columns):
                    self.attach(columns, key, 'ProteinGym_v1.3', f['_source_index'].to_numpy())
        print('ProteinGym assays processed:', len(selected), flush=True)

    def domainome(self):
        records = json.loads((SRC/'domainome_index.json').read_text(encoding='utf-8'))
        for record in records:
            urn = record['urn']
            stem = urn.replace(':', '_')
            detail = json.loads((SRC/'mavedb'/(stem+'.json')).read_text(encoding='utf-8'))
            targets = detail['targetGenes']
            if len(targets) != 1 or targets[0].get('targetSequence', {}).get('taxonomy', {}).get('code') != 9606:
                self.excluded.append({'source_id': urn, 'reason': 'not_one_human_protein_target', 'title': detail['title']})
                continue
            target = targets[0]
            accessions = [x for x in target['externalIdentifiers'] if x['identifier']['dbName'] == 'UniProt']
            accession = accessions[0]['identifier']['identifier'] if accessions else ''
            accession = accession.split('-')[0]
            if accession not in self.uniprot.index or not self.uniprot.loc[accession, 'Gene Names (primary)']:
                self.excluded.append({'source_id': urn, 'reason': 'no_verified_UniProt_gene_mapping', 'title': detail['title']})
                continue
            gene = self.uniprot.loc[accession, 'Gene Names (primary)'].upper()
            if gene != detail['title'].split()[0].upper():
                raise ValueError(f'{urn}: study/UniProt gene-name conflict')
            sequence = target['targetSequence']['sequence']
            ref_id = self.reference(gene, sequence, 9606, accession, accessions[0].get('offset'), source='MaveDB Human Domainome 1.0 exact target sequence')
            t = tier(gene, 9606, 'stability', self.evidence)
            # Growth-coupled domain abundance is a stability proxy, not thermodynamic ddG.
            key = self.assay(gene, urn, ref_id, 'stability', 'MaveDB_domainome_normalized_score', t,
                             source='MaveDB_Domainome1', source_url='https://api.mavedb.org/api/v1/score-sets/'+urn+'/scores',
                             source_sha256=digest(SRC/'mavedb'/(stem+'.csv')),
                             publication_doi='10.1038/s41586-024-08370-4',
                             source_publication_doi='|'.join(x.get('doi', '') or '' for x in detail['primaryPublicationIdentifiers']),
                             publication_title=detail['title'], license=detail['license']['shortName'],
                             host='yeast', selection='growth', system='DHFR_PCA', treatment='methotrexate', region='isolated_domain',
                             score_direction='higher_domain_abundance_proxy_not_health', orientation='published_normalized_score',
                             measurement_type='growth_coupled_domain_abundance_stability_proxy',
                             mechanism_relevance='domain_stability_proxy_not_full_length_function')
            raw = pd.read_csv(SRC/'mavedb'/(stem+'.csv'), float_precision='round_trip')
            raw['_source_index'] = np.arange(len(raw))
            parts = raw.hgvs_pro.fillna('').str.extract(r'^p\.([A-Z][a-z]{2})([1-9][0-9]*)([A-Z][a-z]{2})$')
            wt = parts[0].fillna('').map(lambda x: seq1(x) if x else '')
            mut = parts[2].fillna('').map(lambda x: seq1(x) if x else '')
            valid_variant = wt.isin(list(AA)) & mut.isin(list(AA)) & (wt != mut)
            finite = pd.to_numeric(raw.score, errors='coerce').map(np.isfinite)
            good = valid_variant & finite
            f = pd.DataFrame({'wt': wt[good], 'ref_pos': parts.loc[good, 1].astype(int), 'mut': mut[good],
                              'score_value': raw.loc[good, 'score'], 'source_variant': raw.loc[good, 'hgvs_pro'],
                              'measurement_sigma': raw.loc[good, 'sigma'], 'source_raw_score': raw.loc[good, 'raw_score'],
                              'source_raw_sigma': raw.loc[good, 'raw_sigma']})
            self.qc.append({'source': 'MaveDB_Domainome1', 'assay_id': urn, 'source_rows': len(raw),
                            'single_finite_rows': len(f), 'already_in_v3': 0,
                            'excluded_non_single_or_nonfinite': int((~good).sum()), 'added_rows': len(f),
                            'non_missense_rows': int((~valid_variant).sum()), 'missense_missing_score': int((valid_variant & ~finite).sum())})
            if len(f):
                self.attach(f, key, 'MaveDB_Domainome1', raw.loc[good, '_source_index'].to_numpy())
        print('Domainome records processed:', len(records), 'excluded assays:', len(self.excluded), flush=True)

    def write(self):
        d = pd.concat(self.frames, ignore_index=True)
        d['score_orientation'] = d.assay_key.map({k: a['score_direction'] for k, a in self.assays.items()})
        d['measurement_type'] = d.assay_key.map({k: a.get('measurement_type', a['task']) for k, a in self.assays.items()})
        d['variant_key'] = d.protein_reference_id+':'+d.wt+d.ref_pos.astype(str)+d.mut
        if d.measurement_id.duplicated().any() or d.duplicated(['assay_key', 'variant_key']).any():
            raise ValueError('Duplicate measurement key')
        raw_variants = set(d.loc[d.task != 'consensus', 'variant_key'])
        d['possible_consensus_constituent_overlap'] = ((d.task == 'consensus') & d.variant_key.isin(raw_variants)).astype(int)
        d['default_sample_weight'] = np.where(d.possible_consensus_constituent_overlap == 1, .25, 1.)
        d['is_direct_ndd'] = d.is_direct_ndd.astype(int)
        d['v3_row_id'] = pd.to_numeric(d.get('v3_row_id'), errors='coerce').astype('Int64')
        d['canonical_pos'] = pd.to_numeric(d.canonical_pos, errors='coerce').astype('Int64')
        # Never reuse historic whole-assay target normalization in a fresh holdout.
        priority = ['measurement_id', 'gene', 'protein_reference_id', 'wt', 'ref_pos', 'mut', 'canonical_pos', 'position_system',
                    'assay_id', 'assay_key', 'task', 'assay_type', 'measurement_type', 'score_value', 'score_kind',
                    'score_orientation', 'measurement_sigma', 'supervision_tier', 'training_role', 'is_direct_ndd',
                    'ndd_evidence_class', 'taxon_id', 'species', 'source', 'source_row_id', 'v3_row_id',
                    'default_sample_weight', 'possible_consensus_constituent_overlap']
        d = d[[x for x in priority if x in d]+[x for x in d if x not in priority]]
        # Stable measurement order, with original V3 rows first.
        d.to_csv(DATA/'ndd_training_corpus_v4.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0, 'compresslevel': 4})
        parquet = d.copy()
        for col in parquet.select_dtypes('object'):
            parquet[col] = parquet[col].fillna('').astype(str)
        parquet.to_parquet(DATA/'ndd_training_corpus_v4.parquet', index=False, compression='zstd')
        direct = d[d.is_direct_ndd == 1]
        direct.to_csv(DATA/'ndd_direct_v4.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0, 'compresslevel': 4})
        for tier_name, frame in d.groupby('supervision_tier'):
            frame.to_csv(DATA/(tier_name+'.csv.gz'), index=False, compression={'method': 'gzip', 'mtime': 0, 'compresslevel': 4})
        assays = pd.DataFrame(self.assays.values())
        counts = d.groupby('assay_key').size()
        assays['loaded_rows'] = assays.assay_key.map(counts).fillna(0).astype(int)
        assays = assays[assays.loaded_rows > 0]
        # Curated context is replayed after the automatic pass, which would otherwise
        # re-derive these fields from ProteinGym descriptions and discard the research.
        assays = default_status(apply_overrides(enrich_metadata(assays)))
        assays.to_csv(DATA/'assay_metadata.csv', index=False)
        watchlist_path = SRC/'watchlist_search.csv'
        if watchlist_path.exists():
            watchlist = pd.read_csv(watchlist_path, keep_default_na=False)
            loaded = assays.groupby('gene').size()
            watchlist['loaded_assays'] = watchlist.gene.map(loaded).fillna(0).astype(int)
            watchlist['status'] = [
                'loaded_domain_stability_proxy' if n else
                'candidate_DNA_SGE_requires_splicing_aware_protein_mapping' if gene == 'DDX3X' and hits else
                'candidate_requires_manual_review' if hits else 'not_located_in_this_keyword_search'
                for gene, hits, n in watchlist[['gene', 'search_hits', 'loaded_assays']].itertuples(index=False, name=None)]
            watchlist.to_csv(DATA/'research_watchlist.csv', index=False)
        pd.DataFrame(self.references.values()).to_csv(DATA/'protein_references.csv', index=False)
        genes = sorted(set(d.gene))
        evidence = [self.evidence.get(g, {'gene': g, 'ndd_verified': 0, 'evidence_basis': 'not_verified_by_selected_rule'}) for g in genes]
        pd.DataFrame(evidence).to_csv(DATA/'gene_evidence.csv', index=False)
        pd.DataFrame({'gene': sorted(direct.gene.unique())}).to_csv(DATA/'ndd_evaluation_genes.csv', index=False)
        parent = {gene: gene for gene in genes}
        def find(gene):
            while parent[gene] != gene:
                parent[gene] = parent[parent[gene]]
                gene = parent[gene]
            return gene
        sequence_gene = {}
        for record in self.references.values():
            first = sequence_gene.setdefault(record['sequence'], record['gene'])
            a, b = find(first), find(record['gene'])
            parent[max(a,b)] = min(a,b)
        pd.DataFrame({'gene': genes, 'cluster': [find(g) for g in genes]}).to_csv(DATA/'split_groups.csv', index=False)
        save_json(RES/'sequences.json', self.references)
        (RES/'proteins.fasta').write_text(''.join(f'>{ref}\n{r["sequence"]}\n' for ref, r in self.references.items()), encoding='utf-8')
        # Preserve already verified coordinate caches whenever the exact sequence matches.
        structures = {}
        for ref_id, r in self.references.items():
            source = Path('prepared/resources/structures')/(r['key']+'.npz')
            if source.exists():
                shutil.copyfile(source, RES/'structures'/source.name)
                structures[ref_id] = 'reused_exact_reference_v3_structure'
            else:
                structures[ref_id] = 'not_yet_downloaded_sequence_fallback_available'
        save_json(RES/'structures/report.json', structures)
        pd.DataFrame(self.qc).to_csv(AUDIT/'source_row_accounting.csv', index=False)
        pd.DataFrame(self.excluded).to_csv(AUDIT/'excluded_assays.csv', index=False)
        coverage = d.groupby(['supervision_tier', 'gene', 'task']).size().rename('rows').reset_index()
        coverage.to_csv(AUDIT/'coverage.csv', index=False)
        overlaps = d[d.task != 'consensus'].groupby('variant_key').agg(
            gene=('gene', 'first'), protein_reference_id=('protein_reference_id', 'first'), ref_pos=('ref_pos', 'first'),
            wt=('wt', 'first'), mut=('mut', 'first'), n_assays=('assay_key', 'nunique'), n_tasks=('task', 'nunique'),
            tasks=('task', lambda x: '|'.join(sorted(set(x)))))
        overlaps[overlaps.n_tasks > 1].to_csv(DATA/'multi_property_variant_index.csv.gz', compression={'method': 'gzip', 'mtime': 0})
        summary = {'rows': len(d), 'genes': d.gene.nunique(), 'references': d.protein_reference_id.nunique(), 'assays': d.assay_key.nunique(),
                   'direct_ndd_rows': len(direct), 'direct_ndd_genes': direct.gene.nunique(),
                   'tier_rows': d.supervision_tier.value_counts().to_dict(),
                   'tier_genes': d.groupby('supervision_tier').gene.nunique().to_dict(),
                   'task_rows': d.task.value_counts().to_dict(),
                   'raw_multi_property_genes': int((d[d.task != 'consensus'].groupby('gene').task.nunique() > 1).sum()),
                   'paired_multi_property_variants': int((overlaps.n_tasks > 1).sum()),
                   'original_rows_preserved': int(d.v3_row_id.notna().sum()),
                   'legacy_ndd_unverified_genes': sorted(set(self.old.gene)-{g for g in self.evidence if self.evidence[g]['ndd_verified']}),
                   'possible_consensus_overlap_rows': int(d.possible_consensus_constituent_overlap.sum()),
                   'excluded_assays': self.excluded, 'build_date': DATE,
                   'csv_sha256': digest(DATA/'ndd_training_corpus_v4.csv.gz'),
                   'source_v3_sha256': digest('ndd_training_corpus_v3.csv')}
        save_json(AUDIT/'summary.json', summary)
        source_files = [SRC/'proteingym_v1.3.zip', SRC/'proteingym_reference.csv', SRC/'panelapp_285.json',
                        SRC/'DDG2P_2026-08-28.csv.gz', SRC/'uniprot_sequences.tsv', SRC/'domainome_index.json']
        save_json(AUDIT/'source_hashes.json', {str(path): digest(path) for path in source_files})
        print(json.dumps({k:v for k,v in summary.items() if k != 'excluded_assays'}, indent=2), flush=True)


if __name__ == '__main__':
    builder = Builder()
    builder.legacy()
    builder.proteingym()
    builder.domainome()
    builder.write()
