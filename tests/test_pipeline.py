import json

import numpy as np
import pandas as pd
import pytest
import torch

from mipo.common import save_json
from mipo.corpus import read_corpus
from mipo.data import FeatureStore, Metadata, TargetTransform, VariantDataset, collate
from mipo.features import window_starts
from mipo.losses import loss_function
from mipo.model import MIPO
from mipo.resources import matches, structure_arrays
from mipo.smoke import fixture
from mipo.splits import make_splits, verify_fold


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    paths = fixture(tmp_path)
    d = read_corpus(paths[0])
    config = json.loads(paths[-1].read_text())
    store = FeatureStore(paths[1], paths[2], allow_synthetic=True)
    transform = TargetTransform().fit(d)
    ds = VariantDataset(d, store, transform, Metadata().fit(d), config)
    return paths, d, config, ds


def test_rotation_translation_and_conformer_order(setup):
    _, _, config, ds = setup
    b = collate([ds[0], ds[1]])
    model = MIPO(32, 1, config).eval()
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    with torch.no_grad():
        original = model(b)
        moved = dict(b)
        moved["unit"] = b["unit"]@rotation
        rotated = model(moved)
        assert torch.allclose(original["mu"], rotated["mu"], atol=1e-6)
        assert torch.allclose(original["field"], rotated["field"], atol=1e-6)
        assert torch.allclose(original["vector"]@rotation, rotated["vector"], atol=1e-6)
        moved["unit"] = b["unit"].flip(1)
        moved["edge_weight"] = b["edge_weight"].flip(1)
        assert torch.allclose(original["mu"], model(moved)["mu"], atol=1e-6)
    # Graph construction itself must cancel rigid translation.
    before = ds.graph(ds.rows.iloc[0].gene, 0)
    protein = ds.store.protein(ds.rows.iloc[0].gene)
    protein["coords"] += np.array([13., -8., 2.], dtype=np.float32)
    ds.graph.cache_clear()
    after = ds.graph(ds.rows.iloc[0].gene, 0)
    assert np.allclose(before["unit"], after["unit"], atol=2e-5)


class _HalfOutput(torch.nn.Module):
    """Wrap a module so it returns fp16, as a Linear does under CUDA autocast."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x).half()


def test_forward_tolerates_autocast_dtype_mix(setup):
    """Regression: on GPU with amp, the impulse path is fp16 while project() stays fp32.

    CPU autocast uses a different op policy and never reproduced this, so the CUDA
    condition is recreated directly: impulse modules emit fp16, context stays fp32.
    """
    _, _, config, ds = setup
    b = collate([ds[0], ds[1]])
    model = MIPO(32, 1, config).eval()
    model.impulse = _HalfOutput(model.impulse)
    model.impulse_gate = _HalfOutput(model.impulse_gate)
    with torch.no_grad():
        out = model(b)
    assert torch.isfinite(out["mu"].float()).all()
    assert torch.isfinite(out["sigma"].float()).all()


def test_identity_mutation_zero_field(setup):
    _, _, config, ds = setup
    b = collate([ds[0], ds[1]])
    b["mut"] = b["wt"].clone()
    p = MIPO(32, 1, config).eval()(b)
    assert torch.count_nonzero(p["field"]) == 0
    assert torch.count_nonzero(p["vector"]) == 0


@pytest.mark.parametrize("mode", ["mipo", "esm_mlp", "fixed_heads"])
def test_loss_gradients_modes_and_missing_structure(setup, mode):
    _, _, config, ds = setup
    config = {**config, "model": mode}
    b = collate([ds[i] for i in range(4)])
    b["edge_weight"].zero_()
    b["confidence"].zero_()
    model = MIPO(32, 1, config)
    pred = model(b)
    loss, _ = loss_function(pred, b)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert (pred["sigma"] > 0).all()


def test_ranking_does_not_cross_assays(setup):
    _, _, config, ds = setup
    b = collate([ds[0], ds[1]])
    pred = MIPO(32, 1, config)(b)
    b["group"] = torch.tensor([1, 2])
    _, parts = loss_function(pred, b)
    assert parts["ranking"] == 0


def test_training_only_scaling_and_metadata(setup):
    _, d, _, _ = setup
    train = d[d.gene == "SYNTHETIC_0"].copy()
    train["task"] = "fitness"
    transform = TargetTransform().fit(train)
    assert "binding" not in transform.state
    before = dict(transform.state)
    transform.transform([1e12], ["fitness"])
    assert before == transform.state
    m = Metadata().fit(train)
    assert m.encode("held_out::unseen").tolist() == [0]*5
    assert not any("SYNTHETIC" in k for k in m.vocab)


def test_all_splits_disjoint_and_clusters(setup, tmp_path):
    paths, d, _, _ = setup
    split = json.loads(paths[-2].read_text())
    for fold in split["folds"]:
        verify_fold(d, fold)
    assert sorted(g for f in split["folds"] for g in f["test"]) == sorted(d.gene.unique())
    cluster_path = tmp_path/"clusters.csv"
    pd.DataFrame({"gene": sorted(d.gene.unique()), "cluster": ["a", "a", "b", "c", "d"]}).to_csv(cluster_path, index=False)
    clustered = make_splits(paths[0], tmp_path/"cluster_splits.json", cluster_path)
    for f in clustered["folds"]:
        for role in ["train", "validation", "calibration", "test"]:
            assert ("SYNTHETIC_0" in f[role]) == ("SYNTHETIC_1" in f[role])


def test_duplicate_and_wrong_wt_rejected(setup, tmp_path):
    paths, d, config, ds = setup
    bad = pd.concat([pd.read_csv(paths[0])]*2)
    p = tmp_path/"bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_corpus(p)
    d.loc[0, "wt"] = "A" if d.loc[0, "wt"] != "A" else "C"
    with pytest.raises(ValueError, match="WT/sequence"):
        VariantDataset(d, ds.store, ds.transform, ds.metadata, config)
    with pytest.raises(ValueError, match="Synthetic"):
        FeatureStore(paths[1], paths[2])


def test_windows_cover_long_sequences():
    for length in [1, 32, 768, 769, 3418]:
        counts = np.zeros(length)
        for start in window_starts(length, 768, 128):
            counts[start:start+768] += 1
        assert (counts > 0).all()


def test_resume_matches_uninterrupted(setup, tmp_path, monkeypatch):
    import shutil
    import importlib
    trainer = importlib.import_module("mipo.train")
    from mipo.train import train
    paths, _, _, _ = setup
    full = tmp_path/"full"
    original_save = trainer.torch_save
    def capture_first_epoch(path, value):
        original_save(path, value)
        if path.name == "last.pt" and value["epoch"] == 0:
            shutil.copyfile(path, tmp_path/"epoch0.pt")
    monkeypatch.setattr(trainer, "torch_save", capture_first_epoch)
    train(*paths, out=full, allow_synthetic=True)
    saved = torch.load(full/"last.pt", weights_only=True)
    resumed = tmp_path/"resumed"
    resumed.mkdir()
    shutil.copyfile(tmp_path/"epoch0.pt", resumed/"last.pt")
    shutil.copyfile(tmp_path/"epoch0.pt", resumed/"best.pt")
    train(*paths, out=resumed, resume=True, allow_synthetic=True)
    restored = torch.load(resumed/"last.pt", weights_only=True)
    assert restored["epoch"] == saved["epoch"]
    assert all(torch.equal(v, restored["model"][k]) for k, v in saved["model"].items())
    assert saved["history"] == restored["history"]
    assert (full/"test_predictions.csv").exists()


def test_structure_import_exact_sequence(tmp_path):
    path = tmp_path/"tiny.pdb"
    path.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C  \nATOM      2  CA  CYS A   2       3.800   0.000   0.000  1.00 80.00           C  \nTER\nEND\n")
    xyz, conf = structure_arrays(path, "AC")
    assert xyz.shape == (1, 2, 3)
    assert np.allclose(conf, [[.9, .8]])
    with pytest.raises(ValueError, match="exact sequence"):
        structure_arrays(path, "AD")


def test_auxiliary_labels_train_only(setup, tmp_path):
    from mipo.common import seq_key
    _, d, config, ds = setup
    aux = tmp_path/"teachers"
    aux.mkdir()
    r = d.iloc[0]
    seq = ds.store.records[r.gene]["sequence"]
    np.savez_compressed(aux/f"{seq_key(r.variant_key)}.npz", target=np.ones((len(seq), 2), np.float32),
                        mask=np.ones((len(seq), 2), bool), sequence=np.array(seq))
    config = {**config, "auxiliary_dir": str(aux)}
    training = VariantDataset(d, ds.store, ds.transform, ds.metadata, config, load_aux=True)
    validation = VariantDataset(d, ds.store, ds.transform, ds.metadata, config)
    assert training[0]["aux_mask"].all()
    assert not validation[0]["aux_mask"].any()
    b = collate([training[0]])
    loss, parts = loss_function(MIPO(32, 1, config)(b), b)
    assert parts["auxiliary"] > 0
    assert torch.isfinite(loss)


def test_unseen_task_and_padding(setup):
    _, _, config, ds = setup
    model = MIPO(32, 1, config, [True, False, False, False, False, False]).eval()
    b = collate([ds[0], ds[1]])
    b["task"][:] = 4
    with torch.no_grad():
        output = model(b)
        assert torch.isfinite(output["mu"]).all()
    # Padding a protein with an invalid node must not alter its readout.
    single = ds[0]
    shorter = dict(single)
    keep = len(single["ids"])-1
    for key in ["esm", "ids", "confidence", "aux_target", "aux_mask"]:
        shorter[key] = single[key][:keep]
    for key in ["edge", "edge_features"]:
        shorter[key] = single[key][:keep].copy()
    shorter["edge"] = np.minimum(shorter["edge"], keep-1)
    shorter["unit"] = single["unit"][:, :keep]
    shorter["edge_weight"] = single["edge_weight"][:, :keep]
    with torch.no_grad():
        a = model(collate([shorter]))["mu"][0]
        b = model(collate([shorter, single]))["mu"][0]
    assert torch.allclose(a, b, atol=1e-6)


def test_ensemble_graph_order_invariance(setup):
    _, _, _, ds = setup
    gene = ds.rows.iloc[0].gene
    first = ds.graph(gene, 0)
    prot = ds.store.protein(gene)
    prot["coords"] = prot["coords"][::-1].copy()
    prot["confidence"] = prot["confidence"][::-1].copy()
    ds.graph.cache_clear()
    second = ds.graph(gene, 0)
    assert np.array_equal(first["edge"], second["edge"])
    assert np.allclose(first["edge_features"], second["edge_features"])


def test_mixed_conformer_padding_does_not_change_prediction(setup):
    _, _, config, ds = setup
    a, b = dict(ds[0]), dict(ds[1])
    a["unit"] = a["unit"][:1]
    a["edge_weight"] = a["edge_weight"][:1]
    model = MIPO(32, 1, config).eval()
    with torch.no_grad():
        alone = model(collate([a]))["mu"][0]
        together = model(collate([a, b]))["mu"][0]
    assert torch.allclose(alone, together, atol=1e-6)
