"""Cache the MaveDB records that resolve which experimental arm each ProteinGym assay imported.

Two ways in:

1. ProteinGym's ``raw_DMS_filename`` sometimes *names* the score set it imported
   (``urn_mavedb_00000049-a-6_scores.csv``). That URN pins dose, genetic background and
   readout in a way no amount of reading the paper can.
2. For assays where it does not, a MaveDB text search by gene usually finds the same study's
   deposited score sets. ``MATCHED`` below records the ones whose publication DOI or title
   matches the ProteinGym assay, established by hand -- a search hit alone is not evidence,
   so each entry names why it matches.

Run:  python scripts/research_assay_conditions.py
Out:  v4/sources/mavedb_conditions/*.json + index.json
"""
from pathlib import Path
import json
import re
import time

import pandas as pd
import requests

OUT = Path('v4/sources/mavedb_conditions')
SCORE_SET = 'https://api.mavedb.org/api/v1/score-sets/'
EXPERIMENT = 'https://api.mavedb.org/api/v1/experiments/'
KEEP = ['urn', 'title', 'shortDescription', 'abstractText', 'methodText', 'extraMetadata',
        'license', 'primaryPublicationIdentifiers', 'publishedDate', 'targetGenes']

# Score sets matched to a ProteinGym assay by publication DOI unless noted otherwise.
MATCHED = {
    'urn:mavedb:00000095-b-1': ('CP2C9_HUMAN_Amorosi_2021_abundance', 'same study, abundance arm (VAMP-seq, HEK293T)'),
    'urn:mavedb:00000095-a-1': ('CP2C9_HUMAN_Amorosi_2021_activity', 'same study, activity arm (Click-seq, humanized yeast)'),
    'urn:mavedb:00000096-b-1': ('HXK4_HUMAN_Gersing_2023_abundance', 'DOI 10.1186/s13059-024-03238-2, abundance by DHFR-PCA'),
    'urn:mavedb:00000096-a-1': ('HXK4_HUMAN_Gersing_2022_activity', 'DOI 10.1186/s13059-023-02935-8, activity by complementation'),
    'urn:mavedb:00000098-a-2': ('SCN5A_HUMAN_Glazer_2019', 'DOI 10.1161/CIRCGEN.119.002786'),
    'urn:mavedb:00000041-a-1': ('SRC_HUMAN_Ahler_2019', 'DOI 10.1016/j.molcel.2019.02.003, Src catalytic domain'),
    'urn:mavedb:00000004-a-1': ('UBE4B_MOUSE_Starita_2013', 'Starita 2013 E4B phage-display score set'),
    'urn:mavedb:00000050-a-1': ('MSH2_HUMAN_Jia_2020', 'DOI 10.1016/j.ajhg.2020.12.003, HAP1 6-TG arm'),
    'urn:mavedb:00001232-a-1': ('NPC1_HUMAN_Erwood_2022_HEK293T', 'DOI 10.1038/s41587-021-01201-1, HEK293T arm'),
    'urn:mavedb:00001232-b-1': ('NPC1_HUMAN_Erwood_2022_RPE1', 'DOI 10.1038/s41587-021-01201-1, RPE1 arm'),
    'urn:mavedb:00001223-a-1': ('BRCA2_HUMAN_Erwood_2022_HEK293T', 'DOI 10.1038/s41587-021-01201-1, BRCA2 essentiality arm'),
    'urn:mavedb:00000674-b-1': ('KCNE1_HUMAN_Muhammad_2023_function', 'DOI 10.1101/2023.04.28.538612, functional arm'),
    'urn:mavedb:00000674-a-2': ('KCNE1_HUMAN_Muhammad_2023_expression', 'DOI 10.1101/2023.04.28.538612, trafficking with KCNQ1'),
    'urn:mavedb:00000674-c-1': ('KCNE1_HUMAN_Muhammad_2023_expression', 'DOI 10.1101/2023.04.28.538612, trafficking without KCNQ1'),
    'urn:mavedb:00000055-b-1': ('NUD15_HUMAN_Suiter_2020', 'thiopurine-cytotoxicity activity arm'),
    'urn:mavedb:00000055-a-1': ('NUD15_HUMAN_Suiter_2020', 'VAMP-seq stability arm, for contrast'),
    'urn:mavedb:00000108-a-1': ('HEM3_HUMAN_Loggerenberg_2023', 'ubiquitous HMBS isoform'),
    'urn:mavedb:00000108-b-1': ('HEM3_HUMAN_Loggerenberg_2023', 'erythroid-specific HMBS isoform'),
    'urn:mavedb:00000013-b-1': ('TPMT_HUMAN_Matreyek_2018', 'DOI 10.1038/s41588-018-0122-z, TPMT VAMP-seq'),
}


def referenced_urns():
    """URNs that ProteinGym itself names in raw_DMS_filename."""
    metadata = pd.read_csv('v4/data/assay_metadata.csv', keep_default_na=False)
    ids = set(metadata.loc[metadata.source == 'ProteinGym_v1.3', 'assay_id'])
    reference = pd.read_csv('v4/sources/proteingym_reference.csv', keep_default_na=False)
    found = {}
    for row in reference[reference.DMS_id.isin(ids)].itertuples():
        match = re.search(r'urn_mavedb_(\d{8}-[a-z]-\d+)', str(row.raw_DMS_filename))
        if match:
            found.setdefault('urn:mavedb:'+match.group(1), []).append(row.DMS_id)
    return found


def fetch(urn, base=SCORE_SET):
    target = OUT/(urn.replace(':', '_')+('' if base is SCORE_SET else '_experiment')+'.json')
    if not target.exists():
        response = requests.get(base+urn, timeout=60)
        response.raise_for_status()
        target.write_text(json.dumps(response.json(), indent=2), encoding='utf-8')
        time.sleep(1)
    record = json.loads(target.read_text(encoding='utf-8'))
    return {key: record.get(key) for key in KEEP}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    for urn, assays in sorted(referenced_urns().items()):
        try:
            record = fetch(urn)
            index[urn] = {'assays': assays, 'match': 'named by ProteinGym raw_DMS_filename',
                          'title': record['title']}
            print('named   ', urn, '->', assays, '|', record['title'], flush=True)
        except Exception as error:   # availability is not curation evidence
            index[urn] = {'assays': assays, 'error': str(error)}
            print('FAILED  ', urn, error, flush=True)
    for urn, (assay, why) in sorted(MATCHED.items()):
        try:
            record = fetch(urn)
            index[urn] = {'assays': [assay], 'match': why, 'title': record['title']}
            print('matched ', urn, '->', assay, '|', record['title'], flush=True)
        except Exception as error:
            index[urn] = {'assays': [assay], 'error': str(error)}
            print('FAILED  ', urn, error, flush=True)
    # The OTC experiment record carries the growth medium the score set omits.
    for urn in ['urn:mavedb:00000112-a', 'urn:mavedb:00000041-a']:
        try:
            record = fetch(urn, EXPERIMENT)
            index[urn] = {'match': 'experiment record with protocol detail', 'title': record['title']}
            print('experiment', urn, '|', record['title'], flush=True)
        except Exception as error:
            print('FAILED  ', urn, error, flush=True)
    (OUT/'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    print('cached records:', len(index))
