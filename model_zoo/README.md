# Model catalog

- [BDD100K-FT](../BDD20K_FT.md)
- [Mobility / CMA](../Mobility.md)
- [SeaDronesSee](../SeaDronesSee.md)
- [WAID](../WAID_FT.md)

[catalog.json](catalog.json) stores full-precision metrics, configuration and
checkpoint identities, split sizes, and evaluation settings.
[results.csv](results.csv) is the flat analysis-friendly version. Markdown
tables display AP on a 0–100 scale; machine-readable files use 0–1.

All 136 evaluated recipes are in [configs](configs/). The two ongoing recipes
link to the campaign configuration directory. Checkpoint binaries are not
hosted; the catalog’s download fields remain empty.

The maintainer exporter [build_model_zoo.py](../tools/build_model_zoo.py) reads
the full-native-validation summary and metadata, verifies checkpoint hashes,
exports portable recipes, and renders the domain pages. It runs on CPU and
does not modify the source experiment.
