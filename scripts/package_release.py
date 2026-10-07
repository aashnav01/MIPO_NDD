"""Build a self-contained Colab ZIP without caches, checkpoints or synthetic data."""
import json
import shutil
import zipfile
from pathlib import Path

from mipo.common import digest, save_json

root = Path(__file__).resolve().parents[1]
prepared = root/"prepared"
for name in ["audit", "baselines", "splits"]:
    shutil.copytree(root/"artifacts"/name, prepared/name, dirs_exist_ok=True)
resources = prepared/"resources"
resources.mkdir(parents=True, exist_ok=True)
for name in ["sequences.json", "proteins.fasta", "proteingym_reference.csv", "resolution_failures.json"]:
    shutil.copyfile(root/"artifacts/resources"/name, resources/name)
structures = resources/"structures"
structures.mkdir(exist_ok=True)
for file in (root/"artifacts/resources/structures").iterdir():
    if file.suffix in [".npz", ".json"]:
        shutil.copyfile(file, structures/file.name)
records = json.loads((resources/"sequences.json").read_text())
report = json.loads((structures/"report.json").read_text())
report["UBR5"] = "Experimental PDB 1I2T: exact first 58 residues of chain A, excluding terminal AHG tag; confidence mask=1, not pLDDT"
save_json(structures/"report.json", report)
inventory = {}
for gene, record in records.items():
    file = structures/f"{record['key']}.npz"
    inventory[gene] = {"sequence_key": record["key"], "length": record["length"],
                       "structure": file.exists(), "structure_sha256": digest(file) if file.exists() else None,
                       "source": report[gene]}
save_json(prepared/"resource_inventory.json", inventory)
dist = root/"dist"
dist.mkdir(exist_ok=True)
include = ["mipo", "tests", "scripts", "configs", "notebooks", "prepared"]
files = [p for directory in include for p in (root/directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
files += [p for p in root.iterdir() if p.is_file() and (p.suffix in [".md", ".toml"] or p.name in ["sequence_overrides.json", "README_ndd_training_corpus_v3.txt", "ndd_training_corpus_v3.csv", "ndd_training_corpus_v3_manifest.csv"])]
archive = dist/"mipo_ndd_colab_package.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(files):
        z.write(p, p.relative_to(root).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
save_json(dist/"release.json", {"archive": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size,
                              "files": len(files), "gene_references": len(records), "structures": sum(x["structure"] for x in inventory.values())})
print(json.dumps(json.loads((dist/"release.json").read_text()), indent=2))
