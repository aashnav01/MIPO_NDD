# Rebuilding V4

The shipped dataset is ready to use; rebuilding is optional. Run from the package root. Network scripts cache source responses. Do not overwrite source snapshots and then claim the old release fingerprint.

```bash
python -m pip install -e '.[test]'
python scripts/v4_fetch_sources.py
python scripts/v4_discover_mavedb.py
python scripts/v4_fetch_annotations.py
python scripts/v4_download_assays.py
python scripts/build_v4_dataset.py
python scripts/v4_fetch_structures.py
python scripts/validate_v4_dataset.py
python scripts/build_v4_notebook.py
python -m pytest -q
```

`v4/sources/uniprot_mappings.tsv` is a pinned source mapping included in the release. It maps ProteinGym UniProt entry names to primary gene symbols and accessions; retain its fingerprint with the release. The discovery scripts query the current APIs, while the DDG2P date is explicitly pinned to 2026-08-28. Updating a source release requires a new build version and validation.

The complete archive includes raw MaveDB tables/metadata and the key source snapshots. The 43 MB upstream ProteinGym ZIP and full AlphaFold PDB downloads are omitted from the release archive to avoid redundant storage; their download URLs and hashes remain available. Run the source/structure fetchers to restore those caches. The validated structure NPZs themselves are included.

After rebuilding the dataset, regenerate splits: their fingerprints deliberately reject an altered corpus. Keep the original V3 CSV unchanged. The data card's counts belong to this released snapshot and should be regenerated if inputs change.
