# Start here: MIPO-NDD V4 on Colab

On a DGX A100 or another multi-GPU Linux server, follow [DGX_A100.md](DGX_A100.md) instead.

1. Put `dist/mipo_ndd_v4_complete.zip` in Google Drive at `MyDrive/MIPO_NDD/mipo_ndd_v4_complete.zip`.
2. Open `notebooks/MIPO_NDD_V4_Colab.ipynb` in Colab and select a GPU runtime.
3. Run cells in order. Start with `SCOPE = 'direct'`, the default 150M frozen ESM encoder, and the PTEN held-out fold.
4. Features and training outputs persist in Drive. Resume using the notebook's checkpoint option. Use `curriculum` to add the broader human DMS and family-teacher tiers.
5. Run the provided ablations and additional held-out genes before drawing conclusions about innovation or predictive performance.

The release contains 994,673 measurements across 495 genes. The verified NDD subset has 318,274 measurements across 135 genes. All 157,549 original rows are retained. Five explicit tiers distinguish legacy consensus, direct NDD function, direct NDD stability proxies, family teachers, and generic human DMS.

Read `v4/DATASET_CARD.md` for exact inclusion rules and limitations, `v4/DATA_DICTIONARY.md` for columns, and `ARCHITECTURE.md` for the architecture rationale and testable research hypotheses. Novelty remains a research hypothesis, not an established priority claim.

Many added measurements concern isolated-domain stability proxies. They do not establish whole-protein function or clinical pathogenicity. Structures are available for 562 of 598 references; the other 36 use explicit missing-geometry masks. Colab GPU training has not been executed here, and no trained V4 performance is claimed.
