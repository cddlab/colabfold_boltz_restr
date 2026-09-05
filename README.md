# colabfold_boltz_restr

In `Boltz2.ipynb`, run **1a. Install Dependencies** and **1b. Download Model
Weights** before editing the YAML in step 2 and running prediction in step 3.
The weight download cell shows progress, reuses complete cached files, and can
be rerun after an interrupted download. Prediction checks the cache first and
asks you to rerun preparation if model files are missing or incomplete.
`Boltz1.ipynb` has the same separate installation and weight download steps.

Each prediction validates the YAML and ligand SMILES, identifying invalid values
and their ligand IDs before starting Boltz. It uses a new output folder, so
rerunning after a YAML edit uses the edited input. Both stdout and stderr appear
in the notebook and are saved to `prediction.log`. A run fails if the command
fails or expected final
structures are missing, even when Boltz itself returns exit code zero.

After a failure, run **Download Results** to collect the input YAML and logs
for a bug report. Successful downloads include the prediction outputs. ZIP
archives use compression and exclude `intermediate_*.cif`; prediction no longer
requests intermediate diffusion structures.

Run the local regression checks with:

```sh
uv run --no-project python -m unittest discover -s tests -v
```
