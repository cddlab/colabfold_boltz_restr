import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("Boltz1.ipynb", "Boltz2.ipynb")


def source_with(notebook, text):
    cells = json.loads((ROOT / notebook).read_text())["cells"]
    return next("".join(cell["source"]) for cell in cells if text in "".join(cell["source"]))


def definitions(source, namespace):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    exec(compile(tree, "<notebook helpers>", "exec"), namespace)


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.previous_directory = Path.cwd()
        os.chdir(self.scratch.name)
        self.addCleanup(self.scratch.cleanup)
        self.addCleanup(os.chdir, self.previous_directory)

    def namespace(self, notebook):
        namespace = {
            "BOLTZ_REVISION": "test-revision",
            "BOLTZ_MODEL": "boltz2",
            "BOLTZ_BIN": "unused",
            "INSTALL_LOG": Path("boltz_install.log"),
            "WEIGHTS_LOG": Path("weights_download.log"),
            "WEIGHTS_DIR": Path("weights"),
            "MODEL_CHECKPOINTS": ("boltz2_conf.ckpt", "boltz2_aff.ckpt"),
            "MIN_CHECKPOINT_SIZE": 1_000_000_000,
            "jobname": "sample",
            "yaml_file": Path("sample/sample.yaml"),
            "diffusion_samples": 1,
            "recycling_steps": 1,
        }
        definitions(source_with(notebook, "def run_logged("), namespace)
        definitions(source_with(notebook, "def invalid_checkpoints("), namespace)
        Path("sample").mkdir(exist_ok=True)
        Path("sample/sample.yaml").write_text("sequences: []\n")
        Path("job.yaml").write_text("sequences: []\n")
        return namespace

    def fake_boltz(self, namespace, behavior):
        interpreter = Path("fake_python")
        interpreter.write_text(f"#!{sys.executable}\nraise SystemExit(0)\n")
        interpreter.chmod(0o755)
        namespace["VENV_PYTHON"] = "./fake_python"
        executable = Path("fake_boltz")
        executable.write_text(
            f"#!{sys.executable}\n"
            "from pathlib import Path\n"
            "import sys\n"
            "args = sys.argv[1:]\n"
            "job = Path(args[1]).stem\n"
            "out = Path(args[args.index('--out_dir') + 1])\n"
            "pred = out / f'boltz_results_{job}' / 'predictions' / job\n"
            + behavior
        )
        executable.chmod(0o755)
        namespace["BOLTZ_BIN"] = "./fake_boltz"
        namespace["require_weights"] = lambda: None

    def predict(self, notebook, namespace):
        exec(source_with(notebook, "prediction_succeeded = False"), namespace)

    def download(self, notebook, namespace):
        downloaded = []
        colab = types.ModuleType("google.colab")
        colab.files = types.SimpleNamespace(download=downloaded.append)
        with patch.dict(sys.modules, {"google": types.ModuleType("google"), "google.colab": colab}):
            exec(source_with(notebook, "zip_name ="), namespace)
        self.assertEqual(downloaded, [namespace["zip_name"]])
        return zipfile.ZipFile(downloaded[0])

    def test_notebook_python_cells_compile(self):
        for notebook in NOTEBOOKS:
            for cell in json.loads((ROOT / notebook).read_text())["cells"]:
                source = "".join(cell["source"])
                if cell["cell_type"] == "code" and not source.startswith(("%%", "!")):
                    compile(source, notebook, "exec")

    def test_stdout_and_stderr_are_visible_and_saved_on_failure(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook):
                namespace = self.namespace(notebook)
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "exit code 7"):
                    namespace["run_logged"](
                        [sys.executable, "-c", "import sys; print('progress'); print('invalid SMILES', file=sys.stderr); sys.exit(7)"],
                        "command.log",
                    )
                for message in ("progress", "invalid SMILES"):
                    self.assertIn(message, output.getvalue())
                    self.assertIn(message, Path("command.log").read_text())
                self.assertIn("Exit code: 7", Path("command.log").read_text())

    def test_zero_exit_without_final_structure_fails_and_diagnostics_download(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook):
                namespace = self.namespace(notebook)
                self.fake_boltz(namespace, "print('Invalid SMILES: AAA', file=sys.stderr)\n")
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "without producing"):
                    self.predict(notebook, namespace)
                self.assertFalse(namespace["prediction_succeeded"])
                self.assertIn("Invalid SMILES: AAA", output.getvalue())
                self.assertIn("Download Results", output.getvalue())
                with contextlib.redirect_stdout(output), self.download(notebook, namespace) as archive:
                    self.assertIn("prediction.log", archive.namelist())
                    self.assertIn(Path(namespace["yaml_file"]).name, archive.namelist())
                    self.assertIn("Invalid SMILES: AAA", archive.read("prediction.log").decode())
                self.assertIn("results may be incomplete", output.getvalue())

    def test_edited_input_uses_fresh_output_and_does_not_reuse_previous_success(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook):
                namespace = self.namespace(notebook)
                self.fake_boltz(namespace, "pred.mkdir(parents=True)\n(pred / f'{job}_model_0.cif').write_text('data_model\\n')\n")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.predict(notebook, namespace)
                first_output = namespace["OUTPUT_DIR"]
                self.assertTrue(namespace["prediction_succeeded"])
                yaml_path = Path(namespace["yaml_file"])
                yaml_path.write_text("sequences: [invalid]\n")
                self.fake_boltz(namespace, "print('Input rejected')\n")
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    self.predict(notebook, namespace)
                self.assertNotEqual(namespace["OUTPUT_DIR"], first_output)
                self.assertFalse(namespace["prediction_succeeded"])
                self.assertEqual((Path(namespace["OUTPUT_DIR"]) / yaml_path.name).read_text(), "sequences: [invalid]\n")

    def test_archive_excludes_intermediate_cifs_and_keeps_final_data(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook):
                namespace = self.namespace(notebook)
                self.fake_boltz(namespace, "pred.mkdir(parents=True)\n(pred / f'{job}_model_0.cif').write_text('data_model\\n')\n")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.predict(notebook, namespace)
                prediction_dir = namespace["predictions_dir"]
                for index in range(201):
                    for kind in ("noised", "denoised"):
                        (prediction_dir / f"intermediate_{kind}_{index}.cif").write_text("unwanted\n")
                (prediction_dir / "confidence.json").write_text('{"confidence": 0.9}\n')
                with contextlib.redirect_stdout(io.StringIO()), self.download(notebook, namespace) as archive:
                    names = archive.namelist()
                    self.assertFalse(any("intermediate_" in name for name in names))
                    self.assertTrue(any(name.endswith("_model_0.cif") for name in names))
                    if notebook == "Boltz2.ipynb":
                        self.assertTrue(any(name.endswith("confidence.json") for name in names))
                    else:
                        self.assertTrue(all("/" not in name for name in names))
                    self.assertTrue(all(item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist()))

    def test_missing_weights_stop_before_starting_prediction(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook):
                namespace = self.namespace(notebook)
                with patch.dict(namespace, {"run_logged": unittest.mock.Mock()}) as patched:
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Download Model Weights"):
                        self.predict(notebook, namespace)
                    patched["run_logged"].assert_not_called()

    def test_incomplete_checkpoints_are_rejected(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook):
                namespace = self.namespace(notebook)
                weights = namespace["WEIGHTS_DIR"]
                weights.mkdir(exist_ok=True)
                for name in namespace["MODEL_CHECKPOINTS"]:
                    (weights / name).write_bytes(b"incomplete")
                self.assertEqual(len(namespace["invalid_checkpoints"]()), 2)

    def test_all_requested_models_must_exist(self):
        namespace = self.namespace("Boltz1.ipynb")
        namespace["diffusion_samples"] = 3
        self.fake_boltz(namespace, "pred.mkdir(parents=True)\n(pred / f'{job}_model_0.cif').write_text('data_model\\n')\n")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "all expected"):
            self.predict("Boltz1.ipynb", namespace)

    def test_prediction_does_not_request_intermediate_steps(self):
        for notebook in NOTEBOOKS:
            namespace = self.namespace(notebook)
            self.fake_boltz(namespace, "assert '--save_intermediate_steps' not in args\npred.mkdir(parents=True)\n(pred / f'{job}_model_0.cif').write_text('data_model\\n')\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.predict(notebook, namespace)

    def downloader(self, notebook, retrieve):
        tree = ast.parse(source_with(notebook, "WEIGHT_DOWNLOAD_CODE ="))
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign) and node.targets[0].id == "WEIGHT_DOWNLOAD_CODE"
        )
        download_tree = ast.parse(ast.literal_eval(assignment.value))
        download_tree.body = [node for node in download_tree.body if isinstance(node, ast.FunctionDef)]
        namespace = {
            "Path": Path, "time": time, "tarfile": tarfile,
            "zipfile": zipfile, "original_retrieve": retrieve,
        }
        exec(compile(download_tree, "<weight downloader>", "exec"), namespace)
        return namespace["retrieve_with_progress"]

    def test_interrupted_download_is_not_published_and_can_be_retried(self):
        for notebook in NOTEBOOKS:
            target = Path("molecules.dat")
            target.unlink(missing_ok=True)

            def interrupted(url, filename, progress, data):
                Path(filename).write_bytes(b"partial")
                raise urllib.error.ContentTooShortError("Interrupted", None)

            with self.assertRaises(urllib.error.ContentTooShortError):
                self.downloader(notebook, interrupted)("unused", target)
            self.assertFalse(target.exists())

            def completed(url, filename, progress, data):
                Path(filename).write_bytes(b"complete")
                progress(1, 8, 8)
                return filename, {}

            with contextlib.redirect_stdout(io.StringIO()):
                self.downloader(notebook, completed)("unused", target)
            self.assertEqual(target.read_bytes(), b"complete")
            self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_truncated_checkpoint_is_not_published(self):
        def retrieve(url, filename, progress, data):
            Path(filename).write_bytes(b"truncated checkpoint")
            return filename, {}

        for notebook in NOTEBOOKS:
            with self.assertRaisesRegex(RuntimeError, "checkpoint is incomplete"):
                self.downloader(notebook, retrieve)("unused", "model.ckpt")
            self.assertFalse(Path("model.ckpt").exists())

    def test_invalid_smiles_identifies_ligand_and_value(self):
        for notebook in NOTEBOOKS:
            tree = ast.parse(source_with(notebook, "VALIDATE_INPUT_CODE ="))
            assignment = next(
                node for node in tree.body
                if isinstance(node, ast.Assign) and node.targets[0].id == "VALIDATE_INPUT_CODE"
            )
            validation = ast.literal_eval(assignment.value)
            yaml = types.ModuleType("yaml")
            yaml.safe_load = lambda _: {"sequences": [{"ligand": {"id": "B", "smiles": "AAA"}}]}
            rdkit = types.ModuleType("rdkit")
            rdkit.Chem = types.SimpleNamespace(MolFromSmiles=lambda _: None)
            Path("input.yaml").write_text("invalid input")
            with patch.dict(sys.modules, {"yaml": yaml, "rdkit": rdkit}), patch.object(sys, "argv", ["validate", "input.yaml"]):
                with self.assertRaisesRegex(ValueError, "Invalid SMILES for ligand 'B': 'AAA'"):
                    exec(validation, {})


if __name__ == "__main__":
    unittest.main()
