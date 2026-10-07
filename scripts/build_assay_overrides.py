"""Author the curated assay-context override file from source-backed research notes.

This script is the human-auditable source of truth. Every field it writes carries the URL,
the locator inside that source, a paraphrased summary, and whether the evidence is
assay-specific or a genuinely shared study protocol. Nothing here is inferred from a gene
name, a protein's species, or a plausible-sounding guess: fields the reviewed sources did
not settle stay out of the override and are reported as unresolved instead.

Run:  python scripts/build_assay_overrides.py
Out:  v4/data/assay_context_overrides.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

OUT = Path('v4/data/assay_context_overrides.json')
REVIEW_DATE = '2026-09-20'

# Fields an override may touch. Anything outside this list is rejected by the replay code,
# so a curation mistake cannot silently rewrite scores or identifiers.
ALLOWED_FIELDS = [
    'host', 'selection', 'system', 'treatment', 'region',           # model-conditioning fields
    'assay_environment', 'expression_host', 'cell_line', 'strain',   # documentation columns
    'treatment_dose', 'treatment_duration', 'treatment_role',
    'expression_induction', 'region_coordinates', 'construct_notes',
    # Provenance recovered for collapsed-consensus targets (see recover_consensus_provenance.py)
    'recovered_source_urn', 'recovered_source_doi', 'recovered_rank_correlation',
    'recovered_orientation', 'recovered_match_status', 'recovered_coverage',
]
MODEL_FIELDS = ['host', 'selection', 'system', 'treatment', 'region']

STATUS_VOCABULARY = [
    'verified_assay_specific',        # source describes this exact assay arm
    'verified_shared_protocol',       # one protocol demonstrably covers this assay set
    'not_applicable',                 # the field genuinely does not apply to this design
    'condition_mapping_unresolved',   # study has several arms; imported arm not identified
    'not_reported_in_reviewed_sources',
    'original_provenance_missing',    # collapsed consensus: constituent assays not supplied
    'source_table_derived',           # ProteinGym reference-table wording, not re-verified
]

TSUBOYAMA = 'https://www.nature.com/articles/s41586-023-06328-6'
PG = 'https://proteingym.org/'
PG_TABLE = 'ProteinGym v1.3 reference table, region_mutated and seq_len columns'
MAVE = 'https://mavedb.org/score-sets/'

entries: dict[str, dict] = {}


def put(assay_key, field, value, status, url, locator, summary, scope):
    """Record one field-level, source-backed override for one exact assay."""
    if field not in ALLOWED_FIELDS:
        raise ValueError(f'{field} is not an allowed override field')
    if status not in STATUS_VOCABULARY:
        raise ValueError(f'{status} is not in the status vocabulary')
    entry = entries.setdefault(assay_key, {'fields': {}})
    entry['fields'][field] = {'value': value, 'status': status, 'source_url': url,
                              'locator': locator, 'evidence_summary': summary,
                              'evidence_scope': scope, 'review_date': REVIEW_DATE}


def put_many(assay_key, url, locator, summary, scope, status, **fields):
    for field, value in fields.items():
        put(assay_key, field, value, status, url, locator, summary, scope)


# ======================================================================================
# Shared protocol 1: Tsuboyama 2023 cDNA display proteolysis (all *_Tsuboyama_2023_* rows)
# ======================================================================================
def tsuboyama(metadata):
    rows = metadata[metadata.assay_id.str.contains('Tsuboyama_2023')]
    note = ('Folding stability of isolated PDB-derived domains measured entirely in vitro: the '
            'library is transcribed and translated with a cell-free cDNA display system and '
            'challenged with protease. No cell hosts the phenotype, so a cellular host value '
            'would be wrong; the E. coli-derived in vitro translation kit is expression '
            'chemistry, not an assay host.')
    for row in rows.itertuples():
        pdb = row.assay_id.rsplit('_', 1)[-1]
        put_many(row.assay_key, TSUBOYAMA, 'Methods, "cDNA display proteolysis"; PMC10412457',
                 note, 'shared_protocol', 'verified_shared_protocol',
                 host='not_applicable', assay_environment='cell_free',
                 expression_host='cell_free_E_coli_derived_translation_extract',
                 system='cDNA_display_proteolysis', selection='proteolytic_resistance',
                 region='isolated_domain')
        put(row.assay_key, 'treatment', 'trypsin|chymotrypsin', 'verified_shared_protocol',
            TSUBOYAMA, 'Methods, "Proteolysis"; Fig. 1a',
            'Two orthogonal proteases act as independent folding probes (trypsin targets basic '
            'residues, chymotrypsin aromatic). They are the measurement itself, not a biological '
            'perturbation of a living system.', 'shared_protocol')
        put(row.assay_key, 'treatment_role', 'proteolytic_probe', 'verified_shared_protocol',
            TSUBOYAMA, 'Methods, "Proteolysis"',
            'Protease challenge is the readout rather than a treatment arm.', 'shared_protocol')
        put(row.assay_key, 'region_coordinates', f'PDB:{pdb} domain construct (40-72 aa)',
            'verified_shared_protocol', TSUBOYAMA, 'Abstract; Supplementary domain table',
            f'The assayed construct is the isolated domain taken from PDB entry {pdb}; the curated '
            'set covers natural domains 40-72 residues long.', 'assay_specific')
        put(row.assay_key, 'construct_notes',
            'N-terminal 14-aa PA tag; C-terminal covalent cDNA linkage', 'verified_shared_protocol',
            TSUBOYAMA, 'Fig. 1a', 'cDNA display construct architecture.', 'shared_protocol')
    return len(rows)


# ======================================================================================
# Shared protocol 2: MaveDB Human Domainome 1.0 (501 assays)
# ======================================================================================
def domainome(metadata):
    """Confirm the already-populated shared protocol and add exact Pfam domain coordinates."""
    pattern = re.compile(r'identifier used in the original study is ([A-Z0-9]+)_(PF\d+)_(\d+)')
    rows = metadata[metadata.source == 'MaveDB_Domainome1']
    protocol = ('Domain variants are fused to a DHFR fragment and expressed in yeast; unstable '
                'domains are degraded, lowering DHFR fragment concentration and slowing growth '
                'under methotrexate. Growth is an abundance/stability proxy readout.')
    hits = 0
    for row in rows.itertuples():
        record = json.loads((Path('v4/sources/mavedb')/(row.assay_id.replace(':', '_')+'.json'))
                            .read_text(encoding='utf-8'))
        url = MAVE+row.assay_id
        locator = 'MaveDB score-set methodText and abstractText; study DOI 10.1038/s41586-024-08370-4'
        put_many(row.assay_key, url, locator, protocol, 'shared_protocol',
                 'verified_shared_protocol',
                 host='yeast', selection='growth', system='DHFR_PCA', treatment='methotrexate',
                 region='isolated_domain', assay_environment='cellular')
        put(row.assay_key, 'treatment_role', 'selection', 'verified_shared_protocol', url, locator,
            'Methotrexate imposes the DHFR-dependent growth selection.', 'shared_protocol')
        match = pattern.search(str(record.get('abstractText') or ''))
        if match:
            accession, pfam, start = match.groups()
            length = len(record['targetGenes'][0]['targetSequence']['sequence'])
            hits += 1
            put(row.assay_key, 'region_coordinates',
                f'{accession}:{start}-{int(start)+length-1} ({pfam})', 'verified_assay_specific',
                url, 'MaveDB abstractText original-study identifier; target sequence length',
                f'The score set names its source-study domain identifier {accession}_{pfam}_{start}; '
                f'with the {length}-residue target sequence this gives the exact domain span in '
                f'UniProt {accession}, replacing a length-based guess with the study\'s own Pfam '
                'domain definition.', 'assay_specific')
    return len(rows), hits


# ======================================================================================
# Individually researched ProteinGym assays
# ======================================================================================
def proteingym(metadata):
    key = dict(zip(metadata.assay_id, metadata.assay_key))

    def add(assay_id, url, locator, summary, scope='assay_specific',
            status='verified_assay_specific', **fields):
        if assay_id not in key:
            raise KeyError(assay_id)
        put_many(key[assay_id], url, locator, summary, scope, status, **fields)

    # ---------------- yeast functional complementation ---------------------------------
    add('CBS_HUMAN_Sun_2020', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6993387/',
        'Methods, "High-throughput complementation screening"; PMC6993387',
        'Human CBS variants complement a S. cerevisiae cys4-delta deletion, with GAL1-driven '
        'expression on galactose.',
        host='yeast', selection='growth', system='yeast_functional_complementation',
        region='full_length', assay_environment='cellular',
        strain='S. cerevisiae MATalpha cys4delta::KanMX his3d1 leu2d0 lys2d0 ura3d0',
        expression_induction='2% galactose (GAL1 promoter)')

    add('MTHR_HUMAN_Weile_2021', MAVE+'urn:mavedb:00000049-a-6',
        'MaveDB score-set title for urn:mavedb:00000049-a-6, the set named by ProteinGym raw_DMS_filename',
        'ProteinGym imported urn:mavedb:00000049-a-6, titled "MTHFR at 25ug/ml folate in WT '
        'background". That identifies both the folate dose and the Ala222 (WT) background among '
        'the study\'s several arms.',
        host='yeast', selection='growth', system='yeast_functional_complementation',
        treatment='folate', treatment_dose='25 ug/mL', treatment_role='supplement',
        region='full_length', assay_environment='cellular',
        construct_notes='WT (Ala222) MTHFR background, not the Val222 background')

    url = MAVE+'urn:mavedb:00000112-a-1'
    add('OTC_HUMAN_Lo_2023', url,
        'MaveDB experiment urn:mavedb:00000112-a methodText; score set urn:mavedb:00000112-a-1',
        'A yeast-codon-optimised OTC (yOTC) is integrated into an arg3-delta strain under the '
        'native ARG3 promoter and pinned onto solid SD minimal medium lacking arginine; colonies '
        'grow 72 h at 30 C, are imaged, and are scored with null controls at 0 and wild type at 1.',
        host='yeast', selection='growth', system='solid_growth_complementation',
        treatment='arginine_dropout_medium', treatment_role='selection',
        treatment_duration='72 h at 30 C', region='full_length', assay_environment='cellular',
        strain='S. cerevisiae arg3-delta0')
    put(key['OTC_HUMAN_Lo_2023'], 'construct_notes',
        'yOTC: yeast-codon-optimised OTC with the mitochondrial leader (aa 2-32) removed; '
        'positions use native full-length OTC numbering', 'verified_assay_specific', url,
        'MaveDB experiment urn:mavedb:00000112-a, methodText',
        'The assayed construct is the cytoplasmic mature protein, not the full precursor, which '
        'is exactly why mutagenesis starts at residue 33.', 'assay_specific')

    add('SERC_HUMAN_Xie_2023', MAVE+'urn:mavedb:00000107-b-1',
        'MaveDB score-set methodText for urn:mavedb:00000107-b-1',
        'Human PSAT1 (UniProt SERC_HUMAN) variants complement yeast on solid minimal medium '
        'lacking serine; growth is imaged and normalised so null controls are 0 and wild type 1.',
        host='yeast', selection='growth', system='solid_growth_complementation',
        treatment='serine_dropout_medium', treatment_role='selection',
        region='full_length', assay_environment='cellular')

    add('HMDH_HUMAN_Jiang_2019', MAVE+'urn:mavedb:00000035-a-1',
        'MaveDB score-set shortDescription for urn:mavedb:00000035-a-1',
        'HMGCR functional complementation in yeast by DMS-TileSeq in rosuvastatin media.',
        host='yeast', selection='growth', system='yeast_functional_complementation',
        treatment='rosuvastatin', treatment_role='selection', region='full_length',
        assay_environment='cellular')

    add('HEM3_HUMAN_Loggerenberg_2023', MAVE+'urn:mavedb:00000108-a-1',
        'MaveDB score sets urn:mavedb:00000108-a-1 and -b-1; study DOI 10.1016/j.ajhg.2023.08.012',
        'Human HMBS variant effect maps were produced by functional complementation in yeast. The '
        'study deposited two isoform maps (ubiquitous and erythroid-specific); host, selection and '
        'system are common to both, so they are recorded while the isoform arm stays unresolved.',
        scope='shared_protocol', status='verified_shared_protocol',
        host='yeast', selection='growth', system='yeast_functional_complementation',
        region='full_length', assay_environment='cellular')

    add('HXK4_HUMAN_Gersing_2022_activity', MAVE+'urn:mavedb:00000096-a-1',
        'MaveDB score-set urn:mavedb:00000096-a-1; study DOI 10.1186/s13059-023-02935-8',
        'GCK variant activity measured by functional complementation of an hxk1-delta hxk2-delta '
        'glk1-delta yeast strain, scored by TileSeq. This is a different assay and a different '
        'publication from the 2023 abundance map, despite ProteinGym listing the same raw file '
        'name for both.',
        host='yeast', selection='growth', system='yeast_functional_complementation',
        strain='S. cerevisiae hxk1-delta hxk2-delta glk1-delta', region='full_length',
        assay_environment='cellular', treatment='glucose_medium', treatment_role='selection',
        construct_notes='Complementation of reduced growth on glucose medium in a strain lacking '
                        'all three endogenous hexokinases')

    add('HXK4_HUMAN_Gersing_2023_abundance', MAVE+'urn:mavedb:00000096-b-1',
        'MaveDB score-set urn:mavedb:00000096-b-1; study DOI 10.1186/s13059-024-03238-2',
        'GCK cellular protein abundance measured by DHFR-PCA in yeast, sequencing the library '
        'before and after selection on methotrexate medium across 14 tiles of the ORF.',
        host='yeast', selection='abundance', system='DHFR_PCA', treatment='methotrexate',
        treatment_role='selection', region='full_length', assay_environment='cellular')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5986715/'
    add('PTEN_HUMAN_Mighell_2018', url,
        'Material and Methods, "PTEN Saturation Mutagenesis" and "Yeast Selection Experiments"; PMC5986715',
        'Humanized yeast growth rescue: S. cerevisiae YPH-499 carrying YCpLG-p110alpha-CAAX is '
        'poisoned by PIP3 accumulation and PTEN lipid-phosphatase activity restores growth. '
        'Expression is GAL1/10-driven on SC-galactose -leu -ura.',
        host='yeast', strain='S. cerevisiae YPH-499 (YCpLG-p110alpha-CAAX)', selection='growth',
        system='yeast_functional_complementation', region='full_length',
        assay_environment='cellular',
        expression_induction='galactose (GAL1/10 promoter), SC-galactose -leu -ura')
    put(key['PTEN_HUMAN_Mighell_2018'], 'treatment', 'not_applicable', 'not_applicable', url,
        'Material and Methods, "Yeast Selection Experiments"; PMC5986715',
        'The selective pressure is genetic -- PIP3 toxicity from the p110alpha-CAAX construct the '
        'strain already carries -- and galactose is the carbon source inducing expression, '
        'recorded in expression_induction. No compound perturbs the scored arm and no '
        'treated/untreated comparison is defined.', 'assay_specific')

    add('GDIA_HUMAN_Silverstein_2021',
        'https://www.biorxiv.org/content/10.1101/2021.10.06.463360v1.full',
        'Methods, yeast complementation and library construction',
        'Human GDI1 variants complement a temperature-sensitive yeast GDI1 allele (strain TSA64); '
        'the pooled library is grown competitively at the restrictive temperature. Expression is '
        'constitutive from an ADH1-promoter CEN/ARS vector, not galactose-induced.',
        host='yeast', strain='S. cerevisiae TSA64 (temperature-sensitive GDI1)', selection='growth',
        system='yeast_functional_complementation', treatment='restrictive_temperature',
        treatment_role='selection', region='full_length', assay_environment='cellular',
        expression_induction='constitutive ADH1 promoter (CEN/ARS, pHYC-NatMX)')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5740498/'
    note = ('A modular yeast functional-complementation framework: human variant libraries rescue '
            'growth of a temperature-sensitive yeast strain at an otherwise-lethal restrictive '
            'temperature, scored so 0 is the null allele and 1 is wild-type complementation. The '
            'framework is shared; the text names the ts strain only for UBE2I/UBC9 and gives no '
            'per-gene restrictive temperature, so neither was propagated.')
    for assay_id in ['TPK1_HUMAN_Weile_2017', 'CALM1_HUMAN_Weile_2017',
                     'SUMO1_HUMAN_Weile_2017', 'UBC9_HUMAN_Weile_2017']:
        add(assay_id, url, 'Materials and Methods, DMS-BarSeq/TileSeq complementation; PMC5740498',
            note, scope='shared_protocol', status='verified_shared_protocol',
            host='yeast', selection='growth', system='yeast_functional_complementation',
            treatment='restrictive_temperature', treatment_role='selection',
            region='full_length', assay_environment='cellular')
    put(key['UBC9_HUMAN_Weile_2017'], 'strain', 'S. cerevisiae ubc9-ts', 'verified_assay_specific',
        url, 'Results, UBE2I complementation; PMC5740498',
        'The UBE2I/UBC9 library was transformed into the ubc9-ts strain.', 'assay_specific')

    # ---------------- yeast growth / toxicity selections --------------------------------
    add('TADBP_HUMAN_Bolognesi_2019', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6744496/',
        'Fig. 1a, DMS experimental protocol; PMC6744496',
        'Mutations are restricted to the prion-like domain of TDP-43 and toxicity is quantified in '
        'yeast by pooled growth selection after induction of TDP-43 expression.',
        host='yeast', selection='growth', system='yeast_growth_selection',
        region='subregion_of_full_length_construct', assay_environment='cellular',
        expression_induction='induced TDP-43 expression')

    add('SYUA_HUMAN_Newberry_2020', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7442712/',
        'Methods, pooled yeast library selection; PMC7442712',
        'A barcoded pooled library of alpha-synuclein variants is expressed in S. cerevisiae W303 '
        'from a galactose-inducible promoter; fitness is the slope of log variant frequency over '
        'time under alpha-synuclein toxicity.',
        host='yeast', strain='S. cerevisiae W303', selection='growth',
        system='yeast_growth_selection', region='full_length', assay_environment='cellular',
        expression_induction='1% galactose (SCR-Ura)')

    add('A4_HUMAN_Seuma_2022', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9674652/',
        'Fig. 1, nucleation assay schematic; PMC9674652; ProteinGym region_mutated',
        'Amyloid-beta fused to Sup35N seeds Sup35p aggregation, reading through a premature stop '
        'in the ade1 reporter and permitting growth on medium lacking adenine. The mutated span, '
        'APP 672-713, is the amyloid-beta 42 peptide rather than full-length APP.',
        host='yeast', selection='aggregation', system='sup35_prion_nucleation_assay',
        treatment='adenine_dropout_medium', treatment_role='selection',
        region='isolated_domain', assay_environment='cellular',
        region_coordinates='672-713 of 770 (APP numbering; amyloid-beta 1-42)')

    src = ('Src phosphotransferase activity limits yeast growth, so pooled growth of a barcoded '
           'library reports kinase activity. The assayed construct is the Src catalytic domain.')
    add('SRC_HUMAN_Ahler_2019', MAVE+'urn:mavedb:00000041-a-1',
        'MaveDB experiment urn:mavedb:00000041-a methodText; DOI 10.1016/j.molcel.2019.02.003',
        src+' Barcoded NNK variants on plasmids in yeast were sequenced at three time points by OD.',
        host='yeast', selection='growth', system='yeast_growth_selection',
        region='subregion_construct_scope_unresolved', assay_environment='cellular')

    add('SRC_HUMAN_Nguyen_2022', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10273359/',
        'Methods, yeast growth assay and client-score calculation; PMC10273359',
        src+' Src is expressed in S. cerevisiae BY4741 Green Monster from a galactose-inducible '
        'promoter. The imported "diffsel" column is a differential score: the calibrated '
        'radicicol activity score minus the DMSO activity score, so radicicol (an Hsp90 '
        'inhibitor) defines the scored contrast.',
        host='yeast', strain='S. cerevisiae BY4741 Green Monster', selection='growth',
        system='yeast_growth_selection', treatment='radicicol', treatment_role='competition',
        region='subregion_construct_scope_unresolved', assay_environment='cellular',
        expression_induction='galactose-inducible promoter',
        construct_notes='Score is a radicicol-minus-DMSO differential, not a single-arm measurement')

    add('SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM',
        'https://www.biorxiv.org/content/10.1101/2021.12.05.471322v1.full',
        'Methods, Src-mediated yeast toxicity with ATP-competitive inhibitors; GEO GSE190495',
        src+' Drug sensitivity is read from Src-mediated yeast toxicity in the presence of '
        'ATP-competitive inhibitors. The study used dasatinib at 25 uM and 100 uM, and both the '
        'ProteinGym assay ID and the raw file GSE190495_Src_DAS_25_Score.csv identify the 25 uM arm.',
        host='yeast', selection='growth', system='yeast_growth_selection', treatment='dasatinib',
        treatment_dose='25 uM', treatment_role='competition',
        region='subregion_construct_scope_unresolved', assay_environment='cellular')

    # ---------------- DHFR-PCA (yeast) ---------------------------------------------------
    url = 'https://www.biorxiv.org/content/10.1101/2021.09.14.460249v1.full'
    shared = ('Both the abundancePCA and bindingPCA arms of this study are DHFR protein-fragment '
              'complementation assays in S. cerevisiae BY4742 selected with methotrexate, and in '
              'both arms growth is the readout. The five conditioning fields are therefore common '
              'to the arms even though the imported generic "fitness" column does not say which '
              'arm it came from.')
    for assay_id, coords, construct in [
            ('DLG4_HUMAN_Faure_2021', '311-394 of 724 (ProteinGym region_mutated)',
             'PSD95/DLG4 third PDZ domain, given as aa 354-437 in the publication numbering; '
             'bindingPCA partner is CRIPT'),
            ('GRB2_HUMAN_Faure_2021', '159-214 of 217 (ProteinGym region_mutated)',
             'GRB2 SH3 domain, given as aa 159-224 in the publication; bindingPCA partner is a '
             'GAB2 proline-rich peptide (aa 498-530)')]:
        add(assay_id, url, 'Methods, abundancePCA/bindingPCA; published as Faure et al. 2022, '
            'Nature 604:175, DOI 10.1038/s41586-022-04586-4', shared, scope='shared_protocol',
            status='verified_shared_protocol',
            host='yeast', strain='S. cerevisiae BY4742', selection='growth', system='DHFR_PCA',
            treatment='methotrexate', treatment_role='selection', region='isolated_domain',
            assay_environment='cellular')
        put(key[assay_id], 'region_coordinates', coords, 'verified_assay_specific', PG,
            PG_TABLE, 'Coordinates use the ProteinGym full-length reference numbering, which '
            'differs from the publication numbering for the same domain.', 'assay_specific')
        put(key[assay_id], 'construct_notes', construct, 'verified_assay_specific', url,
            'Methods, domain constructs', 'Exact assayed domain and binding partner.',
            'assay_specific')

    url = 'https://www.biorxiv.org/content/10.1101/2022.12.06.519122v1.full'
    for assay_id, selection, note in [
            ('RASK_HUMAN_Weng_2022_abundance', 'abundance',
             'abundancePCA arm: KRAS is fused to one DHFR fragment while the complementary '
             'fragment is highly expressed, so growth tracks KRAS concentration.'),
            ('RASK_HUMAN_Weng_2022_binding-DARPin_K55', 'binding',
             'bindingPCA arm against DARPin K55, one of six partners profiled in the companion '
             'binding study.')]:
        add(assay_id, url, 'Methods, AbundancePCA/BindingPCA and competition medium; published as '
            'Weng et al. 2024, DOI 10.1038/s41586-023-06954-0',
            note+' Selection medium is SC -URA/MET/ADE with 200 ug/mL methotrexate in '
            'S. cerevisiae BY4742.',
            host='yeast', strain='S. cerevisiae BY4742', selection=selection, system='DHFR_PCA',
            treatment='methotrexate', treatment_dose='200 ug/mL', treatment_role='selection',
            region='full_length', assay_environment='cellular',
            construct_notes='Full-length KRAS 4B, residues 1-188')

    add('B2L11_HUMAN_Dutta_2010_binding-Mcl-1', PG,
        'ProteinGym v1.3 reference table, selection_assay/selection_type columns',
        'The source description states the BIM BH3 peptide library is yeast-displayed and '
        'antibody-stained for binding to the partner Mcl-1, read by FACS.',
        status='source_table_derived',
        host='yeast', selection='binding', system='yeast_display', assay_environment='cellular',
        region='isolated_domain')

    # ---------------- bacterial and phage systems ----------------------------------------
    add('RASH_HUMAN_Bandaru_2017', 'https://elifesciences.org/articles/27810',
        'Results, "unregulated-Ras" experiment; bacterial two-hybrid Methods; Fig. 1C; PMC5538825',
        'A bacterial two-hybrid couples the Ras-GTP:Raf-RBD interaction to chloramphenicol '
        'resistance in E. coli. ProteinGym imports the column named "unregulated", which the paper '
        'defines as the arm where H-Ras variants were expressed without the GAP or the GEF, so the '
        'regulated, attenuated and G12V arms must not be described here.',
        host='e_coli', selection='binding', system='bacterial_two_hybrid',
        treatment='chloramphenicol', treatment_role='selection',
        region='subregion_construct_scope_unresolved', assay_environment='cellular',
        construct_notes='Unregulated arm: no GAP or GEF co-expressed')

    add('RAF1_HUMAN_Zinkus-Boltz_2019', MAVE+'urn:mavedb:00000061-a-1',
        'MaveDB score-set title/methodText for urn:mavedb:00000061-a-1 ("RAF variant selected after 2h")',
        'A phage-assisted continuous selection system couples the RAF-RAS interaction to phage '
        'propagation in E. coli; the imported score set is the 2 h selection timepoint, median of '
        'three replicates.',
        host='e_coli', selection='binding', system='phage_assisted_continuous_selection',
        treatment_duration='2 h', treatment_role='selection', region='isolated_domain',
        assay_environment='cellular')

    add('AICDA_HUMAN_Gajula_2014_3cycles', MAVE+'urn:mavedb:00000106-c-1',
        'MaveDB score-set methodText for urn:mavedb:00000106-c-1; Gajula 2014 PMC4150791',
        'The library is expressed in E. coli and selected for rifampin resistance; counts come '
        'from 454 pyrosequencing after three selection cycles. The readout is DNA sequencing of a '
        'bacterial selection, not bulk RNA sequencing.',
        host='e_coli', selection='growth', system='bacterial_rifampin_selection',
        treatment='rifampin', treatment_role='selection',
        region='subregion_construct_scope_unresolved', assay_environment='cellular')

    for assay_id, urn, segment in [
            ('LYAM1_HUMAN_Elazar_2016', '00000051-a-1', 'C-terminal membrane-spanning segment of L-selectin'),
            ('ERBB2_HUMAN_Elazar_2016', '00000051-b-1', 'ErbB2 transmembrane helix'),
            ('GLPA_HUMAN_Elazar_2016', '00000051-c-1', 'Glycophorin A transmembrane helix')]:
        add(assay_id, MAVE+'urn:mavedb:'+urn,
            f'MaveDB score-set methodText for urn:mavedb:{urn}; Elazar 2016 PMC4786438',
            f'The {segment} is the membrane-spanning segment in the dsTbL assay on the E. coli '
            'inner membrane; bacterial survival on ampicillin monitors membrane integration, '
            'because beta-lactamase functions only once anchored in the inner membrane. Selection '
            'coefficients are converted to apparent insertion free energies at 310 K.',
            host='e_coli', selection='membrane_insertion', system='dsTbL_membrane_insertion',
            treatment='ampicillin', treatment_role='selection',
            region='isolated_domain', assay_environment='cellular',
            construct_notes=f'Isolated {segment} in the TOXCAT-beta-lactamase (dsTbL) scaffold')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3479514/'
    add('YAP1_HUMAN_Araya_2012', url, 'Methods, phage library construction and selection; PMC3479514',
        'A library of hYAP65 WW-domain variants is displayed on T7 bacteriophage and taken through '
        'three rounds of selection for binding to a biotinylated GTPPPPYTVG peptide immobilised on '
        'magnetic streptavidin beads. The binding selection happens in vitro on beads; E. coli only '
        'propagates the phage, so it is the expression host rather than the host of the scored '
        'phenotype.',
        host='not_applicable', expression_host='e_coli_T7_bacteriophage',
        assay_environment='in_vitro_phage_display', system='phage_display', region='isolated_domain',
        construct_notes='34-amino-acid variable region spanning most of the WW domain; ligand is '
                        'the biotinylated peptide GTPPPPYTVG')
    put(key['YAP1_HUMAN_Araya_2012'], 'treatment', 'not_applicable', 'not_applicable', url,
        'Methods, selection for binding; PMC3479514',
        'Selection is affinity capture on an immobilised peptide over three rounds. No compound is '
        'applied to a scored arm and no treated/untreated comparison exists.', 'assay_specific')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8458277/'
    add('PAI1_HUMAN_Huttinger_2021', url, 'Methods, M13 phage display and uPA selection; PMC8458277',
        'PAI-1 is displayed as a fusion on the surface of M13 filamentous phage; phage whose '
        'displayed PAI-1 forms a covalent complex with uPA are captured with an anti-uPA antibody, '
        'so the readout is inhibition of uPA rather than growth. E. coli only propagates the phage.',
        host='not_applicable', expression_host='e_coli_M13_bacteriophage',
        assay_environment='in_vitro_phage_display', system='phage_display',
        selection='enzymatic_activity', region='full_length')

    add('UBE4B_MOUSE_Starita_2013', MAVE+'urn:mavedb:00000004-a-1',
        'MaveDB score-set urn:mavedb:00000004-a-1 methodText; Starita 2013',
        'The E4B U-box domain is assayed by phage display for ubiquitin-ligase autoubiquitination '
        'activity; the imported scores use input versus round-3 selected counts.',
        host='not_applicable', expression_host='e_coli_bacteriophage',
        assay_environment='in_vitro_phage_display', system='phage_display',
        selection='enzymatic_activity', region='isolated_domain')

    for assay_id in ['CASP3_HUMAN_Roychowdhury_2020', 'CASP7_HUMAN_Roychowdhury_2020']:
        add(assay_id, 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8748541/',
            'Methods, microfluidic droplet screening; PMC8748541',
            'Variants are expressed in E. coli, the cells are lysed, and the enzymatic phenotype is '
            'read from a fluorogenic substrate inside microfluidic droplets. The phenotype is '
            'measured in vitro on lysate; E. coli is only the expression host.',
            host='not_applicable', expression_host='e_coli',
            assay_environment='in_vitro_droplet_lysate', selection='enzymatic_activity',
            system='microfluidic_droplet_fluorescence', region='full_length')

    # ---------------- human / mouse cell lines -------------------------------------------
    url = 'https://www.nature.com/articles/s41467-022-30463-9'
    add('PPM1D_HUMAN_Miller_2022', url, 'Methods, saturation mutagenesis screen; Fig. 1A; PMC9246869',
        'Engineered K562 cells carrying wild-type TP53 and a p21(CDKN1A)-GFP knock-in reporter are '
        'transduced with a lentiviral PPM1D(1-427) library, selected in puromycin, treated with '
        'daunorubicin and sorted into GFP-high and GFP-low fractions.',
        host='human_cell_line', cell_line='K562 (TP53-wild-type, CDKN1A-GFP reporter)',
        selection='transcriptional_reporter', system='landing_pad_FACS', treatment='daunorubicin',
        treatment_dose='200 nM', treatment_duration='24 h', treatment_role='stimulation',
        region='subregion_of_full_length_construct', assay_environment='cellular',
        construct_notes='Truncated PPM1D residues 1-427 library; puromycin selects transductants')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8518224/'
    add('PTEN_HUMAN_Matreyek_2021', url,
        'Methods, cell culture and library recombination; VAMP-seq results; PMC8518224',
        'EGFP-PTEN fusion abundance measured by VAMP-seq in HEK 293T LLP-iCasp9-Blast clone 12 '
        'landing-pad cells; expression is doxycycline-induced and abundance is the EGFP:mCherry '
        'ratio by FACS.',
        host='human_cell_line', cell_line='HEK 293T LLP-iCasp9-Blast clone 12', selection='abundance',
        system='VAMP_seq', region='full_length', assay_environment='cellular',
        expression_induction='doxycycline 2 ug/mL (Tet-inducible landing pad)')
    put(key['PTEN_HUMAN_Matreyek_2021'], 'treatment', 'not_applicable', 'not_applicable', url,
        'Methods, library experiments; PMC8518224',
        'Steady-state abundance is read from a single unperturbed population; the only compound in '
        'the protocol is doxycycline, which induces expression rather than perturbing the scored '
        'phenotype, and is recorded in expression_induction.', 'assay_specific')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5980760/'
    add('TPMT_HUMAN_Matreyek_2018', url,
        'Methods, landing-pad recombination and VAMP-seq; MaveDB urn:mavedb:00000013-b-1; PMC5980760',
        'VAMP-seq measures steady-state abundance of EGFP-tagged TPMT variants recombined into the '
        'Tet-on landing pad of engineered HEK 293T TetBxb1BFP Clone4 cells, sorted by FACS on the '
        'EGFP:mCherry ratio.',
        host='human_cell_line', cell_line='HEK 293T TetBxb1BFP Clone4 (Tet-on landing pad)',
        selection='abundance', system='VAMP_seq', region='full_length', assay_environment='cellular',
        expression_induction='doxycycline 2 ug/mL')
    put(key['TPMT_HUMAN_Matreyek_2018'], 'treatment', 'not_applicable', 'not_applicable', url,
        'Methods, library experiments; PMC5980760',
        'Steady-state abundance from one unperturbed population; doxycycline induces expression '
        'and is recorded in expression_induction.', 'assay_specific')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9694338/'
    add('SHOC2_HUMAN_Kwon_2022', url,
        'Methods, "SHOC2 DMS Viability Screen"; Extended Data Fig. 9b; PMC9694338',
        'The pooled SHOC2 DMS viability screen is performed in PA-TU-8902 cells treated with 10 nM '
        'trametinib or DMSO and harvested 16 days after treatment began. MIA PaCa-2 appears only in '
        'follow-up validation, so it must not be recorded as the screen line.',
        host='human_cell_line', cell_line='PA-TU-8902', selection='growth',
        system='pooled_lentiviral_proliferation_screen', treatment='trametinib',
        treatment_dose='10 nM', treatment_duration='16 days', treatment_role='selection',
        region='full_length', assay_environment='cellular')

    add('MK01_HUMAN_Brenan_2016', 'https://doi.org/10.1016/j.celrep.2016.09.061',
        'Results and Methods, MITE library and pooled proliferation screen; PMID 27760319',
        'A doxycycline-inducible, virally delivered MITE cDNA library covering ERK2 residues 2-360 '
        'is screened in a pooled proliferation/competition assay in A375 cells, quantified after '
        '96 h. The imported column DOX_Average is the doxycycline-induced arm.',
        host='human_cell_line', cell_line='A375', selection='growth',
        system='pooled_lentiviral_proliferation_screen', region='full_length',
        assay_environment='cellular', treatment_duration='96 h',
        expression_induction='doxycycline-induced cDNA library expression')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6168352/'
    locator = 'Methods, pooled selection screens, gDNA sample list; PMC6168352'
    for assay_id, line, drug, dose in [
            ('P53_HUMAN_Giacomelli_2018_WT_Nutlin', 'A549 isogenic TP53-wild-type', 'nutlin-3', '2.5 uM'),
            ('P53_HUMAN_Giacomelli_2018_Null_Nutlin', 'A549 isogenic TP53-null', 'nutlin-3', '5 uM'),
            ('P53_HUMAN_Giacomelli_2018_Null_Etoposide', 'A549 isogenic TP53-null', 'etoposide', '5 uM')]:
        add(assay_id, url, locator,
            f'Pooled positive-selection screen in {line} A549 lung carcinoma cells with {drug} at '
            f'{dose}. The Methods sample list distinguishes the arms explicitly (p53WT + nutlin-3 '
            '2.5 uM, p53NULL + nutlin-3 5 uM, p53NULL + etoposide 5 uM) and each ProteinGym column '
            'names its own arm, so the doses must not be shared across the three assays.',
            host='human_cell_line', cell_line=line, selection='growth',
            system='pooled_lentiviral_proliferation_screen', treatment=drug, treatment_dose=dose,
            treatment_role='selection', region='full_length', assay_environment='cellular')

    add('P53_HUMAN_Kotler_2018', 'https://doi.org/10.1016/j.molcel.2018.06.012',
        'Methods, library construction and screen; PMID 29979965; ProteinGym region_mutated 102-292',
        'A lentiviral library covering the p53 DNA-binding domain (residues 102-292) is transduced '
        'at MOI 0.1 into p53-null H1299 non-small-cell lung carcinoma cells, and relative abundance '
        'is tracked at 2, 6, 9 and 14 days post-infection.',
        host='human_cell_line', cell_line='H1299 (p53-null)', selection='growth',
        system='pooled_lentiviral_proliferation_screen',
        region='subregion_of_full_length_construct', assay_environment='cellular',
        construct_notes='DNA-binding domain only; mutp53 followed by an IRES-EGFP reporter')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7820631/'
    locator = 'Methods, saturation genome editing in TMD8 cells; PMC7820631'
    base = ('CARD11 variants are installed by Cas9/ssODN saturation genome editing in human TMD8 '
            'cells; cultures are split into IMDM with 50 nM ibrutinib or IMDM with DMSO vehicle.')
    add('CAR11_HUMAN_Meitlis_2020_gof', url, locator,
        base+' The gain-of-function score table is the ibrutinib-selected arm.',
        host='human_cell_line', cell_line='TMD8', selection='growth',
        system='saturation_genome_editing', treatment='ibrutinib', treatment_dose='50 nM',
        treatment_role='selection', region='subregion_of_full_length_construct',
        assay_environment='cellular')
    add('CAR11_HUMAN_Meitlis_2020_lof', url, locator,
        base+' The loss-of-function table is the vehicle arm: an explicitly documented DMSO '
        'control culture, which is why a vehicle value is recorded here rather than "unknown".',
        host='human_cell_line', cell_line='TMD8', selection='growth',
        system='saturation_genome_editing', treatment='DMSO_vehicle',
        treatment_role='vehicle_control', region='subregion_of_full_length_construct',
        assay_environment='cellular')

    add('BRCA1_HUMAN_Findlay_2018', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6181777/',
        'Methods, saturation genome editing and function-score calculation; PMC6181777',
        'Saturation genome editing in haploid HAP1 cells; variants compromising BRCA1 function are '
        'depleted, scored as the log2 ratio of day-11 frequency to the original plasmid library. '
        'Editing targets thirteen exons encoding the RING and BRCT domains (exons 2-5 and 15-23), '
        'so although ProteinGym records a 1-1855 span, the assay does not tile the whole protein.',
        host='human_cell_line', cell_line='HAP1', selection='growth',
        system='saturation_genome_editing', treatment_duration='11 days',
        region='subregion_of_full_length_construct', assay_environment='cellular',
        region_coordinates='RING and BRCT domains, exons 2-5 and 15-23 (ProteinGym records 1-1855 of 1863)')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7820803/'
    add('MSH2_HUMAN_Jia_2020', url,
        'Methods, cell culture and 6-TG selection; Fig. 1B; MaveDB urn:mavedb:00000050-a-1 ("MSH2 LOF scores (HAP1)")',
        'An MSH2 variant library complements MSH2-knockout human HAP1 cells; cultures are split '
        'into 6-thioguanine-treated and mock-treated arms and the LOF score is the log2 ratio of '
        'variant frequency after 6-TG over mock.',
        host='human_cell_line', cell_line='HAP1', selection='growth',
        system='pooled_lentiviral_proliferation_screen', treatment='6-thioguanine',
        treatment_role='selection', region='full_length', assay_environment='cellular',
        construct_notes='Score is a ratio against a mock-treated arm of the same library')

    erwood = ('Saturation prime editing in locally haploidized cells. The two NPC1 arms differ in '
              'cell line and in mutagenized region, confirming they are separate experiments.')
    for assay_id, line, urn in [('NPC1_HUMAN_Erwood_2022_HEK293T', 'HEK293T (locally haploidized)', '00001232-a-1'),
                                ('NPC1_HUMAN_Erwood_2022_RPE1', 'RPE1 (locally haploidized)', '00001232-b-1')]:
        add(assay_id, MAVE+'urn:mavedb:'+urn,
            f'MaveDB score-set urn:mavedb:{urn} methodText; study DOI 10.1038/s41587-021-01201-1',
            erwood+' NPC1 function is read from a fluorescence-based lysosomal cholesterol '
            'accumulation assay: cells are stained with LysoTracker and scored as '
            'log2(frequency_high/frequency_low) between gated populations.',
            host='human_cell_line', cell_line=line, selection='fluorescent_reporter',
            system='saturation_prime_editing_FACS',
            region='subregion_of_full_length_construct', assay_environment='cellular',
            construct_notes='LysoTracker staining reports lysosomal accumulation, not direct NPC1 activity')

    add('BRCA2_HUMAN_Erwood_2022_HEK293T', MAVE+'urn:mavedb:00001223-a-1',
        'MaveDB score-set urn:mavedb:00001223-a-1 methodText; study DOI 10.1038/s41587-021-01201-1',
        erwood+' The BRCA2 arm is an essentiality screen scored as log2(day-14/day-6 frequency).',
        host='human_cell_line', cell_line='HEK293T (locally haploidized)', selection='growth',
        system='saturation_prime_editing', treatment_duration='day 6 to day 14',
        region='subregion_of_full_length_construct', assay_environment='cellular')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7462613/'
    add('VKOR1_HUMAN_Chiasson_2020_abundance', url,
        'Results, VAMP-seq for VKOR; Fig. 1; PMC7462613',
        'Abundance arm: an eGFP-VKOR fusion is expressed from a Bxb1 landing pad in HEK293T cells '
        'and abundance is the eGFP:mCherry ratio by flow cytometry.',
        host='human_cell_line', cell_line='HEK293T (Bxb1 landing pad)', selection='abundance',
        system='VAMP_seq', region='full_length', assay_environment='cellular')
    add('VKOR1_HUMAN_Chiasson_2020_activity', url,
        'Results, gamma-glutamyl carboxylation reporter; Fig. 2a; PMC7462613',
        'Activity arm: a distinct HEK293 reporter line with VKORC1 and VKORC1L1 knocked out '
        'expresses a Factor IX Gla-domain reporter, and carboxylation-dependent surface staining '
        'with a carboxylation-specific antibody is read by flow cytometry.',
        host='human_cell_line',
        cell_line='HEK293 VKOR activity reporter (VKORC1/VKORC1L1 double knockout)',
        selection='enzymatic_activity', system='FACS', region='full_length',
        assay_environment='cellular')
    for assay_id in ['VKOR1_HUMAN_Chiasson_2020_abundance', 'VKOR1_HUMAN_Chiasson_2020_activity']:
        put(key[assay_id], 'treatment', 'not_applicable', 'not_applicable', url,
            'Results and Methods, cell culture; PMC7462613',
            'Warfarin is discussed as VKOR biology and clinical motivation, but neither multiplexed '
            'arm applies it as a screen treatment and no treated/untreated arm is defined. '
            'Recording warfarin here would invent a condition.', 'assay_specific')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9273215/'
    locator = 'Fig. 1, library generation and sorting methods; PMC9273215'
    base = ('A stable single-copy library is generated by BxBI recombination in HEK293T cells. The '
            'assayed protein is mouse Kir2.1 but the assay host is a human-derived cell line, so '
            'host must not be taken from the target protein species.')
    add('KCNJ2_MOUSE_Coyote-Maestas_2022_surface', url, locator,
        base+' The surface arm sorts on antibody labelling of an extracellular FLAG tag.',
        host='human_cell_line', cell_line='HEK293T (BxBI landing pad)', selection='surface_expression',
        system='FACS', region='full_length', assay_environment='cellular')
    add('KCNJ2_MOUSE_Coyote-Maestas_2022_function', url, locator,
        base+' The function arm sorts on potassium conductance reported by a voltage-sensitive dye.',
        host='human_cell_line', cell_line='HEK293T (BxBI landing pad)', selection='ion_conduction',
        system='FACS', region='full_length', assay_environment='cellular')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7574912/'
    add('ACE2_HUMAN_Chan_2020', url, 'Results, library selection and Fig. 1; PMC7574912',
        'The ACE2 library is transiently expressed in human Expi293F cells at roughly one variant '
        'per cell, incubated with a subsaturating dilution of SARS-CoV-2 RBD-sfGFP, and sorted by '
        'dual-colour flow cytometry into high and low RBD-binding populations.',
        host='human_cell_line', cell_line='Expi293F', selection='binding', system='FACS',
        region='subregion_of_full_length_construct', assay_environment='cellular',
        construct_notes='MYC-tagged ACE2; probe is SARS-CoV-2 RBD-sfGFP')

    add('CCR5_HUMAN_Gill_2023', 'https://www.biorxiv.org/content/10.1101/2023.03.25.534231v1.full',
        'Methods, library transfection and FACS gating',
        'X4-knockout Expi293F cells stably expressing FLAG-CCR5-VN were transfected with CCR5-VC '
        'saturation libraries and sorted either on myc-tag surface expression or on high '
        'bimolecular fluorescence complementation signal. Host and system are common to both gates; '
        'which gate the imported avg_score column holds was not established.',
        host='human_cell_line', cell_line='Expi293F (X4 knockout)', system='FACS',
        region='full_length', assay_environment='cellular')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6926029/'
    base = ('MHC-I variant libraries are assayed in human Expi293F cells with two complementary '
            'fluorescence-based selections read by FACS.')
    add('Q53Z42_HUMAN_McShan_2019_expression', url,
        'Results, deep mutagenesis selections; PMC6926029',
        base+' The expression arm sorts the HLA-A*02:01 library for MHC-I cell surface expression '
        'in wild-type cells.',
        host='human_cell_line', cell_line='Expi293F', selection='surface_expression', system='FACS',
        region='subregion_of_full_length_construct', assay_environment='cellular')
    add('Q53Z42_HUMAN_McShan_2019_binding-TAPBPR', url,
        'Results, bimolecular fluorescence complementation selections; PMC6926029',
        base+' The binding arm scores MHC-I/TAPBPR association by bimolecular fluorescence '
        'complementation of split Venus in TAPBPR-TM-VC-expressing cells.',
        host='human_cell_line', cell_line='Expi293F', selection='binding',
        system='bimolecular_fluorescence_complementation',
        region='subregion_of_full_length_construct', assay_environment='cellular')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7707821/'
    add('ADRB2_HUMAN_Jones_2020', url, 'Methods, reporter stimulation; Fig. 1B; PMC7707821',
        'A barcoded cAMP-response-element transcriptional reporter is read by RNA sequencing in a '
        'HEK293T-derived Bxb1 landing-pad line integrated at the H11 locus. Cells were stimulated '
        'for 4 h across a 0-10 uM isoproterenol range, which identifies the imported raw phenotype '
        'name "0.625" as the 0.625 uM isoproterenol condition rather than a unitless score. The '
        'luciferase experiment is a separate validation assay.',
        host='human_cell_line', cell_line='HEK293T-derived Bxb1 landing pad (H11 locus)',
        selection='transcriptional_reporter', system='barcoded_transcriptional_reporter_RNAseq',
        treatment='isoproterenol', treatment_dose='0.625 uM', treatment_duration='4 h',
        treatment_role='stimulation', region='full_length', assay_environment='cellular')

    add('OPSD_HUMAN_Wan_2019', MAVE+'urn:mavedb:00000099-a-1',
        'MaveDB score-set methodText for urn:mavedb:00000099-a-1; Wan 2019 PMC7027811',
        'Multiplexed rhodopsin surface-expression assay: variants are counted by unique 20 bp '
        'probes in antibody-stained high and low surface-expression bins.',
        host='human_cell_line', selection='surface_expression', system='FACS',
        region='full_length', assay_environment='cellular')

    add('PRKN_HUMAN_Clausen_2023', MAVE+'urn:mavedb:00000114-a-1',
        'MaveDB score-set methodText for urn:mavedb:00000114-a-1',
        'Parkin variant abundance in human cells measured by VAMP-seq, scored from FACS bin '
        'indices and normalised to nonsense and synonymous variants.',
        host='human_cell_line', selection='abundance', system='VAMP_seq', region='full_length',
        assay_environment='cellular')

    add('KCNH2_HUMAN_Kozek_2020', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7704534/',
        'Methods, library and flow-cytometry trafficking assay; PMC7704534',
        'HEK293T cells express KV11.1 variants carrying an HA tag in the first extracellular loop; '
        'live cells are stained with Alexa 647 anti-HA, which labels only surface channel, and '
        'assayed by flow cytometry. The trafficking score is a weighted average across four sorted '
        'pools normalised to wild type. The mutagenized span is residues 545-555, running from the '
        'S4-S5 linker into the S5 helix, so this measures surface trafficking rather than voltage.',
        host='human_cell_line', cell_line='HEK293T', selection='surface_expression', system='FACS',
        region='subregion_of_full_length_construct', assay_environment='cellular')

    url = 'https://www.biorxiv.org/content/10.1101/2023.04.28.538612v1'
    add('KCNE1_HUMAN_Muhammad_2023_function', MAVE+'urn:mavedb:00000674-b-1',
        'MaveDB score-set urn:mavedb:00000674-b-1 methodText; preprint DOI 10.1101/2023.04.28.538612',
        'The KCNE1-HA library is integrated into the landing pad of HEK293T cells and fitness of '
        'library cells is measured in LP-KCNQ1-S140G cells over days. The functional readout is '
        'therefore cell fitness in a KCNQ1-S140G background, not a direct conductance measurement.',
        host='human_cell_line', cell_line='HEK293T landing pad (LP-KCNQ1-S140G)', selection='growth',
        system='landing_pad_FACS', region='full_length', assay_environment='cellular')
    add('KCNE1_HUMAN_Muhammad_2023_expression', MAVE+'urn:mavedb:00000674-a-2',
        'MaveDB score-sets urn:mavedb:00000674-a-2 and -c-1 methodText; preprint DOI 10.1101/2023.04.28.538612',
        'Cell-surface trafficking of the KCNE1-HA library is measured in HEK293 landing-pad cells '
        'by staining and sorting. The study deposited two trafficking score sets, in the presence '
        '(LP-KCNQ1) and absence (LP) of KCNQ1; host, selection and system are common to both, so '
        'they are recorded while the KCNQ1 co-expression arm stays unresolved.',
        scope='shared_protocol', status='verified_shared_protocol',
        host='human_cell_line', cell_line='HEK293 landing pad', selection='surface_expression',
        system='landing_pad_FACS', region='full_length', assay_environment='cellular')

    add('NUD15_HUMAN_Suiter_2020', MAVE+'urn:mavedb:00000055-b-1',
        'MaveDB score-set urn:mavedb:00000055-b-1 ("NUDT15 activity scores"); Suiter 2020 PNAS, PMC7071893',
        'The NUDT15 library is introduced into HEK293T landing-pad cells at one variant per cell. '
        'The imported "Final NUDT15 activity Score" is the thiopurine-cytotoxicity arm: variant '
        'frequency after thioguanine treatment relative to an untreated population. The companion '
        'VAMP-seq abundance arm is a separate score set.',
        host='human_cell_line', cell_line='HEK293T landing pad', selection='growth',
        system='landing_pad_FACS', treatment='thioguanine', treatment_role='selection',
        region='full_length', assay_environment='cellular')

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5131844/'
    add('PPARG_HUMAN_Majithia_2016', url, 'Results and Methods, CD36 FACS screen; PMC5131844',
        'The PPARG construct library is introduced into THP-1 monocytes edited to lack endogenous '
        'PPARG, one construct per cell; cells are differentiated to macrophages, stimulated with '
        'PPARgamma agonists, and FACS-sorted on expression of the target gene CD36.',
        host='human_cell_line', cell_line='THP-1 (PPARG-null, differentiated to macrophages)',
        system='FACS', treatment='rosiglitazone|prostaglandin_J2', treatment_role='stimulation',
        region='full_length', assay_environment='cellular',
        construct_notes='CD36 is an endogenous PPARgamma target gene read by antibody staining')

    add('CP2C9_HUMAN_Amorosi_2021_abundance', MAVE+'urn:mavedb:00000095-b-1',
        'MaveDB score-set urn:mavedb:00000095-b-1 shortDescription',
        'The abundance arm of this study is VAMP-seq in HEK293T cells.',
        host='human_cell_line', cell_line='HEK293T', selection='abundance', system='VAMP_seq',
        region='full_length', assay_environment='cellular')
    add('CP2C9_HUMAN_Amorosi_2021_activity', MAVE+'urn:mavedb:00000095-a-1',
        'MaveDB score-set urn:mavedb:00000095-a-1 shortDescription',
        'The activity arm is a yeast activity-based protein profiling assay (Click-seq) in '
        'humanized yeast cells -- a different host from the abundance arm of the same study, so '
        'the two must not share a host value.',
        host='yeast', selection='enzymatic_activity',
        system='activity_based_protein_profiling_Click_seq', region='full_length',
        assay_environment='cellular')

    url = 'https://www.biorxiv.org/content/10.1101/2023.06.06.543963v1.full'
    add('S22A1_HUMAN_Yee_2023_abundance', url, 'Methods, split-fluorescent-protein abundance screen',
        'OCT1 (SLC22A1) is C-terminally tagged with a split mNeonGreen fragment in HEK293T cells '
        'and variants are sorted by fluorescence intensity.',
        host='human_cell_line', cell_line='HEK293T', selection='abundance', system='VAMP_seq',
        region='full_length', assay_environment='cellular')
    add('S22A1_HUMAN_Yee_2023_activity', url, 'Methods, SM73 cytotoxicity screen',
        'The activity arm is a cytotoxicity screen: SM73, a cisplatin analogue, kills cells in an '
        'OCT1-dependent manner, and cells were treated with or without SM73 at 1 uM. The imported '
        'column name SM73_1_score corresponds to that 1 uM arm.',
        host='human_cell_line', cell_line='HEK293T', selection='growth',
        system='pooled_cytotoxicity_screen', treatment='SM73', treatment_dose='1 uM',
        treatment_role='selection', region='full_length', assay_environment='cellular')

    add('SC6A4_HUMAN_Young_2021', 'https://www.biorxiv.org/content/10.1101/2021.04.19.440442v1.full',
        'Methods, library selection and FACS gating',
        'The SERT library is expressed in Expi293F cells and two selections are run: surface '
        'expression via a fluorescent anti-myc antibody, and transport of the fluorescent '
        'monoamine analogue APP+. The imported column avg_MYC is the myc surface-expression arm, '
        'in which the top 50% of Alexa 647-positive cells were collected.',
        host='human_cell_line', cell_line='Expi293F', selection='surface_expression', system='FACS',
        region='full_length', assay_environment='cellular')

    add('SCN5A_HUMAN_Glazer_2019', MAVE+'urn:mavedb:00000098-a-2',
        'MaveDB score-set urn:mavedb:00000098-a-2 methodText; Methods, library integration and '
        'drug challenge, PMC7031040; DOI 10.1161/CIRCGEN.119.002786',
        'The barcoded SCN5A library is integrated into HEK TetBxb1BFP landing-pad cells, one '
        'variant per cell, then challenged with 25 uM veratridine, 250 ng/mL brevetoxin and 10 uM '
        'ouabain for 5 h. Scores are log10(post-drug / initial frequency), rescaled so mean '
        'nonsense is 0 and mean synonymous is 100; the triple-drug challenge selects against '
        'channel function.',
        host='human_cell_line', cell_line='HEK TetBxb1BFP (Bxb1 landing pad)',
        selection='growth', system='pooled_cytotoxicity_screen',
        treatment='veratridine|brevetoxin|ouabain',
        treatment_dose='25 uM veratridine, 250 ng/mL brevetoxin, 10 uM ouabain',
        treatment_duration='5 h', treatment_role='selection',
        region='subregion_of_full_length_construct', assay_environment='cellular')

    add('MET_HUMAN_Estevam_2023', 'https://www.biorxiv.org/content/10.1101/2023.08.03.551866v1.full',
        'Methods, Ba/F3 cell culture and IL-3 withdrawal selection',
        'MET variant libraries are screened in murine Ba/F3 pro-B cells: one plate is kept free of '
        'IL-3 as the experimental withdrawal condition while a control plate receives 10 ng/mL '
        'IL-3. The imported file ex14_scores.csv is the exon-14-skipped TPR-MET library. The '
        'protein is human but the assay host is a mouse cell line.',
        host='mouse_cell_line', cell_line='Ba/F3', selection='growth',
        system='pooled_retroviral_proliferation_screen', treatment='IL3_withdrawal',
        treatment_role='selection', region='full_length', assay_environment='cellular',
        construct_notes='MSCV TPR-MET(delta-exon14)-IRES-eGFP fusion construct')

    add('TPOR_HUMAN_Bridgford_2020', 'https://doi.org/10.1182/blood.2019002561',
        'Results and Methods, Ba/F3 cytokine-independent growth screen; PMID 31697803',
        'Saturation mutagenesis of the TpoR juxtamembrane-transmembrane region (residues 488-516) '
        'is screened for the ability to confer cytokine-independent growth in murine Ba/F3 cells. '
        'The imported file names the S505N background, i.e. the screen for second-site mutations '
        'that modify S505N-driven activation.',
        host='mouse_cell_line', cell_line='Ba/F3', selection='growth',
        system='pooled_retroviral_proliferation_screen', treatment='IL3_withdrawal',
        treatment_role='selection', region='subregion_of_full_length_construct',
        assay_environment='cellular', construct_notes='S505N TpoR background')

    add('CD19_HUMAN_Klesmith_2019_FMC_singles', PG,
        'ProteinGym raw_DMS_filename single-site/Clinical_FMC_T1_Fitness.tsv; Klesmith 2019 '
        'CD19 ECD saturation library, DOI 10.1021/acs.molpharmaceut.9b00418',
        'The CD19 extracellular-domain single-site saturation library is screened by yeast surface '
        'display with flow-cytometric selection for binding of the clinical anti-CD19 antibody '
        'FMC63; the imported table is the FMC63 T1 selection.',
        status='source_table_derived',
        host='yeast', selection='binding', system='yeast_display',
        region='subregion_of_full_length_construct', assay_environment='cellular')

    # ---- Designs with no perturbation arm at all --------------------------------------
    # `not_applicable` is used only where the reviewed Methods show a single population read
    # on a reporter, with no treated/untreated contrast. Where the assay applies a reagent
    # that *is* the measurement -- a protease, a staining antibody, a fluorogenic substrate,
    # an immobilised ligand -- that reagent is the probe, not a treatment of the scored arm,
    # and it is recorded in construct_notes or system instead. Anything less clear stays
    # `unknown` rather than being tidied away.
    no_treatment_arm = {
        'OPSD_HUMAN_Wan_2019': ('surface-expression bins read by antibody staining',
                                MAVE+'urn:mavedb:00000099-a-1', 'MaveDB score-set methodText'),
        'PRKN_HUMAN_Clausen_2023': ('VAMP-seq FACS bin indices',
                                    MAVE+'urn:mavedb:00000114-a-1', 'MaveDB score-set methodText'),
        'CP2C9_HUMAN_Amorosi_2021_abundance': ('VAMP-seq abundance in HEK293T',
                                               MAVE+'urn:mavedb:00000095-b-1', 'MaveDB score-set shortDescription'),
        'S22A1_HUMAN_Yee_2023_abundance': ('split-fluorescent-protein abundance sorting',
                                           'https://www.biorxiv.org/content/10.1101/2023.06.06.543963v1.full', 'Methods, abundance screen'),
        'Q53Z42_HUMAN_McShan_2019_expression': ('surface-expression FACS in wild-type cells',
                                                'https://pmc.ncbi.nlm.nih.gov/articles/PMC6926029/', 'Results, deep mutagenesis selections'),
        'Q53Z42_HUMAN_McShan_2019_binding-TAPBPR': ('BiFC signal sorting',
                                                    'https://pmc.ncbi.nlm.nih.gov/articles/PMC6926029/', 'Results, BiFC selections'),
        'KCNJ2_MOUSE_Coyote-Maestas_2022_surface': ('anti-FLAG surface staining and sorting',
                                                    'https://pmc.ncbi.nlm.nih.gov/articles/PMC9273215/', 'Fig. 1, sorting methods'),
        'KCNJ2_MOUSE_Coyote-Maestas_2022_function': ('voltage-sensitive-dye conductance sorting',
                                                     'https://pmc.ncbi.nlm.nih.gov/articles/PMC9273215/', 'Fig. 1, sorting methods'),
        'KCNH2_HUMAN_Kozek_2020': ('anti-HA surface staining and sorting',
                                   'https://pmc.ncbi.nlm.nih.gov/articles/PMC7704534/', 'Methods, trafficking assay'),
        'KCNE1_HUMAN_Muhammad_2023_expression': ('surface staining and sorting of the KCNE1-HA library',
                                                 MAVE+'urn:mavedb:00000674-a-2', 'MaveDB score-set methodText'),
        'ACE2_HUMAN_Chan_2020': ('sorting on binding of an RBD-sfGFP probe',
                                 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7574912/', 'Results, library selection'),
        'SC6A4_HUMAN_Young_2021': ('anti-myc surface-expression sorting',
                                   'https://www.biorxiv.org/content/10.1101/2021.04.19.440442v1.full', 'Methods, FACS gating'),
        'CASP3_HUMAN_Roychowdhury_2020': ('a fluorogenic substrate read inside droplets',
                                          'https://pmc.ncbi.nlm.nih.gov/articles/PMC8748541/', 'Methods, droplet screening'),
        'CASP7_HUMAN_Roychowdhury_2020': ('a fluorogenic substrate read inside droplets',
                                          'https://pmc.ncbi.nlm.nih.gov/articles/PMC8748541/', 'Methods, droplet screening'),
        'PAI1_HUMAN_Huttinger_2021': ('capture of phage whose displayed PAI-1 complexed with uPA',
                                      'https://pmc.ncbi.nlm.nih.gov/articles/PMC8458277/', 'Methods, phage display selection'),
        'UBE4B_MOUSE_Starita_2013': ('input versus round-3 phage-display selection counts',
                                     MAVE+'urn:mavedb:00000004-a-1', 'MaveDB score-set methodText'),
        'SYUA_HUMAN_Newberry_2020': ('galactose-induced alpha-synuclein toxicity alone',
                                     'https://pmc.ncbi.nlm.nih.gov/articles/PMC7442712/', 'Methods, pooled selection'),
        'TADBP_HUMAN_Bolognesi_2019': ('induced TDP-43 toxicity alone',
                                       'https://pmc.ncbi.nlm.nih.gov/articles/PMC6744496/', 'Fig. 1a, DMS protocol'),
        'MK01_HUMAN_Brenan_2016': ('doxycycline-induced proliferation competition alone',
                                   'https://doi.org/10.1016/j.celrep.2016.09.061', 'Methods, pooled proliferation screen'),
        'P53_HUMAN_Kotler_2018': ('abundance tracking over 2-14 days with no drug',
                                  'https://doi.org/10.1016/j.molcel.2018.06.012', 'Methods, screen design'),
        'BRCA1_HUMAN_Findlay_2018': ('day-11 versus plasmid-library depletion with no drug',
                                     'https://pmc.ncbi.nlm.nih.gov/articles/PMC6181777/', 'Methods, function-score calculation'),
        'BRCA2_HUMAN_Erwood_2022_HEK293T': ('a day-14 versus day-6 essentiality contrast with no drug',
                                            MAVE+'urn:mavedb:00001223-a-1', 'MaveDB score-set methodText'),
        'NPC1_HUMAN_Erwood_2022_HEK293T': ('LysoTracker staining and fluorescence gating',
                                           MAVE+'urn:mavedb:00001232-a-1', 'MaveDB score-set methodText'),
        'NPC1_HUMAN_Erwood_2022_RPE1': ('LysoTracker staining and fluorescence gating',
                                        MAVE+'urn:mavedb:00001232-b-1', 'MaveDB score-set methodText'),
        'SRC_HUMAN_Ahler_2019': ('pooled yeast growth over three timepoints with no inhibitor',
                                 MAVE+'urn:mavedb:00000041-a-1', 'MaveDB experiment methodText'),
    }
    for assay_id, (what, url, locator) in no_treatment_arm.items():
        put(key[assay_id], 'treatment', 'not_applicable', 'not_applicable', url, locator,
            f'The scored measurement is {what}. The reviewed Methods describe a single population '
            'with no perturbation arm and no treated/untreated contrast, so no treatment applies '
            'to the scored arm.', 'assay_specific')


# ======================================================================================
# Collapsed-consensus targets whose original assay was recovered by rank matching
# ======================================================================================
# A rank-preserving rescale destroys values but not order, so a deposited score set whose
# scores are a monotone function of the supplied 0-1 targets *is* the source. Established by
# scripts/recover_consensus_provenance.py; |rho| and coverage are recorded per assay so a
# reader can see how strong each identification is. Two genes match at rho = -1: the corpus
# target is rank-INVERTED relative to the deposit, which is recorded, not silently corrected.
CONSENSUS_SOURCES = {
    'ASPA': dict(urn='urn:mavedb:00000657-a-1', doi='10.1038/s41467-024-48481-0',
                 note='Site-saturation ASPA library fused to GFP with an IRES-mCherry control, '
                      'integrated into a HEK293T landing-pad line and FACS-sorted: VAMP-seq abundance.',
                 host='human_cell_line', cell_line='HEK293T landing pad', selection='abundance',
                 system='VAMP_seq', treatment='not_applicable', region='full_length'),
    'CRX': dict(urn='urn:mavedb:00001227-a-2', doi='10.1101/gr.279415.124',
                note='Deep mutational scan of CRX in HEK 293-derived cells carrying a synthetic '
                     'fluorescent reporter; activity is a bin-weighted fluorescence average.',
                host='human_cell_line', cell_line='HEK293-derived reporter line',
                selection='transcriptional_reporter',
                system='barcoded_fluorescent_reporter_FACS', treatment='not_applicable',
                region='full_length'),
    'MAPK1': dict(urn='urn:mavedb:00000103-b-1', doi='10.1016/j.celrep.2016.09.061',
                  note='The "MAPK1 Induced with DOX" score set, i.e. the same doxycycline-induced '
                       'A375 proliferation screen already present as MK01_HUMAN_Brenan_2016.',
                  host='human_cell_line', cell_line='A375', selection='growth',
                  system='pooled_lentiviral_proliferation_screen', region='full_length',
                  expression_induction='doxycycline-induced cDNA library expression'),
    'OTC': dict(urn='urn:mavedb:00000112-a-1', doi='10.1016/j.ajhg.2023.03.019',
                note='The same yeast arg3-delta complementation score set already present as '
                     'OTC_HUMAN_Lo_2023; the consensus row duplicates that measurement.',
                host='yeast', strain='S. cerevisiae arg3-delta0', selection='growth',
                system='solid_growth_complementation', treatment='arginine_dropout_medium',
                region='full_length'),
    'PSAT1': dict(urn='urn:mavedb:00000107-b-1', doi='',
                  note='The same yeast solid-growth complementation score set already present as '
                       'SERC_HUMAN_Xie_2023.',
                  host='yeast', selection='growth', system='solid_growth_complementation',
                  treatment='serine_dropout_medium', region='full_length'),
    'SUMO1': dict(urn='urn:mavedb:00000001-b-1', doi='10.15252/msb.20177908',
                  note='Deep mutational scan of human SUMO1 by functional complementation in yeast '
                       'via DMS-TileSeq.',
                  host='yeast', selection='growth', system='yeast_functional_complementation',
                  treatment='restrictive_temperature', region='full_length'),
    'SPOP': dict(urn='urn:mavedb:00001258-a-1', doi='',
                 note='Amino-acid-level scan of human SPOP by a yeast survival assay; enrichment '
                      'between SPOP-expressed and non-expressed pools as z-scores.',
                 host='yeast', selection='growth', system='yeast_growth_selection',
                 region='full_length'),
    'BRCA2': dict(urn='urn:mavedb:00001224-a-1', doi='10.1016/j.ajhg.2024.02.002',
                  note='Arrayed homology-directed repair assay: full-length BRCA2 cDNA carrying '
                       'each variant is co-transfected with an iSce1 vector into brca2-deficient '
                       'V-C8 cells bearing a DR-GFP reporter, scored by flow cytometry at 72 h. '
                       'V-C8 is a Chinese hamster line, so the host is not human.',
                  host='hamster_cell_line', cell_line='V-C8 (brca2-deficient, DR-GFP reporter)',
                  selection='dna_repair_reporter', system='arrayed_HDR_FACS',
                  treatment_duration='72 h', region='full_length'),
    'PALB2': dict(urn='urn:mavedb:00001278-a-1', doi='10.1038/s41467-025-67252-z',
                  note='Site-saturation screen of PALB2 missense variants in the coiled-coil and '
                       'WD40 domains in mouse embryonic stem cells, using PARP-inhibitor '
                       'sensitivity as a homologous-recombination readout. Human protein, mouse host.',
                  host='mouse_cell_line', cell_line='mouse embryonic stem cells',
                  selection='growth', system='site_saturation_PARPi_screen',
                  treatment='PARP_inhibitor', treatment_role='selection',
                  region='subregion_of_full_length_construct',
                  region_coordinates='coiled-coil and WD40 domains'),
    'FKRP': dict(urn='urn:mavedb:00001197-a-5', doi='10.1101/2023.07.12.548370',
                 note='SMuRF (Saturation Mutagenesis-Reinforced Functional assays) measuring FKRP '
                      'variant effects on alpha-dystroglycan glycosylation. The assay host was not '
                      'established from the reviewed record.',
                 selection='glycosylation_reporter', system='SMuRF_glycosylation_assay',
                 region='full_length'),
    'LARGE1': dict(urn='urn:mavedb:00001254-a-1', doi='10.1101/2023.07.12.548370',
                   note='SMuRF assay measuring LARGE1 variant effects on alpha-dystroglycan '
                        'glycosylation. Host not established from the reviewed record.',
                   selection='glycosylation_reporter', system='SMuRF_glycosylation_assay',
                   region='full_length'),
    'KCNQ1': dict(urn='urn:mavedb:00000094-a-15', doi='',
                  note='Normalised currents of 62 KCNQ1 missense SNVs measured in the homozygous '
                       'state: an electrophysiological readout. No method text was deposited, so '
                       'host and recording system were not established.',
                  selection='ion_conduction', system='electrophysiology'),
    'TSC2': dict(urn='urn:mavedb:00001267-0-1', doi='',
                 note='TSC2 combined scores across the Tuberin and RapGAP domains. This deposit is '
                      'itself a meta-analysis of two domain-level analyses, so the consensus target '
                      'may legitimately be an aggregate rather than a single relabelled assay.',
                 region='subregion_of_full_length_construct',
                 region_coordinates='Tuberin and RapGAP domains'),
}


def consensus_recovery(metadata):
    """Attach recovered provenance to the collapsed-consensus assays, by evidence."""
    table = pd.read_csv('v4/audit/consensus_provenance_map.csv', keep_default_na=False)
    table['rho'] = pd.to_numeric(table.rho, errors='coerce')
    table['coverage'] = pd.to_numeric(table.coverage, errors='coerce')
    table['overlap'] = pd.to_numeric(table.overlap, errors='coerce')
    best = (table[table.status.isin(['rank_identical_source', 'strong_partial'])]
            .assign(absrho=lambda f: f.rho.abs())
            .sort_values('absrho', ascending=False).drop_duplicates('gene').set_index('gene'))
    keys = dict(zip(metadata.gene+'::'+metadata.assay_id, metadata.assay_key))
    count = 0
    for gene, spec in CONSENSUS_SOURCES.items():
        key = keys.get(gene+'::gate_consensus')
        if key is None or gene not in best.index:
            continue
        row = best.loc[gene]
        inverted = row.rho < 0
        url = MAVE+spec['urn']
        locator = (f"MaveDB score set {spec['urn']}"
                   + (f"; publication DOI {spec['doi']}" if spec['doi'] else '')
                   + '; rank match established by scripts/recover_consensus_provenance.py')
        evidence = (f"The supplied 0-1 consensus target is a rank-preserving relabel of this score "
                    f"set: Spearman rho = {row.rho:+.4f} over {int(row.overlap)} shared variants "
                    f"({row.coverage:.0%} of the consensus rows). " + spec['note'])
        if inverted:
            evidence += (' The rank correlation is NEGATIVE, so the supplied target is ordered '
                         'opposite to the deposited scores. Measurements were left untouched; the '
                         'inversion is recorded so a later join against the raw deposit cannot '
                         'silently mix orientations.')
        status = ('verified_assay_specific' if row.status == 'rank_identical_source'
                  else 'condition_mapping_unresolved')
        for field in ['host', 'cell_line', 'strain', 'selection', 'system', 'treatment',
                      'treatment_role', 'treatment_duration', 'region', 'region_coordinates',
                      'expression_induction']:
            if field in spec:
                put(key, field, spec[field], status, url, locator, evidence, 'assay_specific')
        put(key, 'recovered_source_urn', spec['urn'], status, url, locator, evidence, 'assay_specific')
        put(key, 'recovered_source_doi', spec['doi'] or 'not_deposited', status, url, locator,
            evidence, 'assay_specific')
        put(key, 'recovered_rank_correlation', f'{row.rho:+.6f}', status, url, locator,
            evidence, 'assay_specific')
        put(key, 'recovered_coverage', f'{row.coverage:.4f}', status, url, locator,
            evidence, 'assay_specific')
        put(key, 'recovered_orientation',
            'inverted_relative_to_source' if inverted else 'preserved_relative_to_source',
            status, url, locator, evidence, 'assay_specific')
        put(key, 'recovered_match_status', row.status, status, url, locator, evidence, 'assay_specific')
        put(key, 'assay_environment', 'cellular', status, url, locator, evidence, 'assay_specific')
        count += 1
    return count


# ======================================================================================
# Controlled vocabulary for `selection`, from the source table where it is unambiguous
# ======================================================================================
# The assay-conditioned head consumes `selection` as a token: left raw, semantically identical
# assays ("Yeast growth", "growth enrichment", "Fitness") become distinct tokens. Verbatim
# source wording stays untouched in `selection_assay`, so nothing is lost.
SELECTION_FROM_SOURCE_TABLE = {
    'BRCA1_HUMAN_Findlay_2018': 'growth',
    'BRCA2_HUMAN_Erwood_2022_HEK293T': 'growth',
}
# Descriptions naming two properties at once, or none: standardising these would be a guess
# about which property the imported column actually holds.
SELECTION_AMBIGUOUS = {
    'CCR5_HUMAN_Gill_2023': 'source names both binding affinity and surface expression, and the study ran both gates',
    'NUD15_HUMAN_Suiter_2020': 'source selection_assay is empty; resolved instead from the matched MaveDB score set',
    'PPARG_HUMAN_Majithia_2016': 'CD36 is a downstream target readout and the imported score is an integrated score',
}


def standardise_selection(metadata):
    lookup = dict(zip(metadata.assay_id, metadata.assay_key))
    wording = dict(zip(metadata.assay_id, metadata.selection_assay))
    count = 0
    for assay_id, value in SELECTION_FROM_SOURCE_TABLE.items():
        key = lookup[assay_id]
        if 'selection' in entries.get(key, {}).get('fields', {}):
            continue
        put(key, 'selection', value, 'source_table_derived', PG,
            'ProteinGym v1.3 reference table, selection_assay/selection_type columns',
            f'Source description "{wording[assay_id]}" standardised to "{value}"; the verbatim '
            'wording is preserved in the selection_assay column.', 'assay_specific')
        count += 1
    return count


def region_from_reference(metadata):
    """Fill region/region_coordinates from ProteinGym's own mutagenized span where not curated."""
    reference = pd.read_csv('v4/sources/proteingym_reference.csv', keep_default_na=False).set_index('DMS_id')
    added = 0
    for row in metadata[metadata.source == 'ProteinGym_v1.3'].itertuples():
        if row.assay_id not in reference.index or 'Tsuboyama' in row.assay_id:
            continue
        record = reference.loc[row.assay_id]
        match = re.fullmatch(r'(\d+)-(\d+)', str(record.region_mutated).strip())
        if not match:
            continue
        start, end, length = int(match.group(1)), int(match.group(2)), int(record.seq_len)
        coverage = (end-start+1)/length
        existing = entries.get(row.assay_key, {}).get('fields', {})
        if 'region_coordinates' not in existing:
            put(row.assay_key, 'region_coordinates', f'{start}-{end} of {length} (ProteinGym region_mutated)',
                'verified_assay_specific', PG, PG_TABLE,
                'Exact mutagenized span reported by the source table for this assay.', 'assay_specific')
        if 'region' not in existing:
            added += 1
            if coverage >= 0.9:
                put(row.assay_key, 'region', 'full_length', 'verified_assay_specific', PG, PG_TABLE,
                    f'Mutagenesis spans {start}-{end} of a {length}-residue reference '
                    f'({coverage:.0%}), i.e. effectively the whole assayed reference.', 'assay_specific')
            else:
                put(row.assay_key, 'region', 'subregion_construct_scope_unresolved',
                    'verified_assay_specific', PG, PG_TABLE,
                    f'Only {start}-{end} of {length} residues ({coverage:.0%}) was mutagenized. '
                    'Whether the assayed construct was the isolated fragment or the full-length '
                    'protein was not established from the reviewed sources.', 'assay_specific')
    return added


def main():
    metadata = pd.read_csv('v4/data/assay_metadata.csv', keep_default_na=False)
    n_tsu = tsuboyama(metadata)
    n_dom, n_pfam = domainome(metadata)
    proteingym(metadata)
    n_cons = consensus_recovery(metadata)
    n_sel = standardise_selection(metadata)
    n_region = region_from_reference(metadata)

    unknown = sorted(set(entries)-set(metadata.assay_key))
    if unknown:
        raise KeyError(f'override keys not present in metadata: {unknown[:5]}')

    payload = {
        'schema_version': '1.1',
        'review_date': REVIEW_DATE,
        'description': 'Exact-assay, field-level, source-backed assay context overrides for '
                       'MIPO-NDD V4. Replayed after automatic enrichment.',
        'baseline_metadata_sha256': 'bb98120b8b25e78d8b83f5f94515668d7c23f5cb3f9233e3e9f3566793784095',
        'allowed_fields': ALLOWED_FIELDS,
        'model_conditioning_fields': MODEL_FIELDS,
        'status_vocabulary': STATUS_VOCABULARY,
        'selection_left_unstandardised': SELECTION_AMBIGUOUS,
        'overrides': dict(sorted(entries.items())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    cells = sum(len(e['fields']) for e in entries.values())
    print(f'assays with overrides: {len(entries)}  field entries: {cells}')
    print(f'  Tsuboyama: {n_tsu}  Domainome: {n_dom} (Pfam coordinates {n_pfam})')
    print(f'  consensus assays with recovered provenance: {n_cons}')
    print(f'  selection standardised: {n_sel}  region from coordinates: {n_region}')
    print('written:', OUT)


if __name__ == '__main__':
    main()
