# PhishGuard AI

Member 1, Phase 1 covers local environment setup, dataset selection, and CSV ingestion.

## Dataset

Potpelwar, R., Kulkarni, U., and Waghmare, J. (2025). *LegitPhish Dataset*, version 2. Mendeley Data. DOI: 10.17632/hx4m73v2sf.2. CC BY 4.0. [Publisher record and download](https://data.mendeley.com/datasets/hx4m73v2sf/2).

Under **Files**, download `url_features_extracted1.csv` into `data/raw/`. If you use **Download All**, extract that CSV from the archive. Keep the original filename for the default command. Treat the CSV as data and avoid opening its URLs.

The publisher documents `URL` and `ClassLabel` with 0 for phishing and 1 for legitimate. It also lists supplied features such as URL length, entropy, dot count, and an HTTPS flag. The loader selects only the URL and label so downstream code can derive features from the same URL input used in prediction.

The publisher lists 101,219 total URLs and publishes class counts of 63,678 phishing and 37,540 legitimate. The class counts add to 101,218. Local ingestion found 101,219 rows, including one missing label and zero unexpected labels. PhishTank is a separate optional feed for later work. [PhishTank developer documentation](https://www.phishtank.org/developer_info.php).

## Run in VS Code on Windows

Open this folder in VS Code. In its PowerShell terminal run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.data.ingestion
```

Select `.venv\Scripts\python.exe` with **Python: Select Interpreter**. Pandas loads the CSV and produces the inspection counts. Pytest runs the development tests. The CLI also accepts an explicit local CSV path.

The CLI prints row count, projected columns, and counts for phishing, legitimate, missing, and other labels. It prints no URL values. Tests use reserved example domains.

## Ingestion interface

`from src.data.ingestion import load_legitphish`

Call `load_legitphish(path)` from the checkout root with a local CSV file. It returns a Pandas DataFrame with nullable string columns `url` and `label`, in that order. The loader retains row order and source label values, including unexpected values, and represents empty fields as missing. Missing files and required headers raise errors. No URL values are visited. This interface can be called from the team's future ingestion task.

## Member 1 layout

```text
phishguard-ai/
├── data/
│   ├── raw/
│   │   └── url_features_extracted1.csv    Local Phase 1 dataset (ignored by Git)
│   └── processed/                    Planned for Phases 2–3
├── src/
│   ├── data/ingestion.py             Phase 1
│   ├── data/validation.py            Phase 2
│   ├── features/preprocessing.py     Phase 3
│   └── models/                       Phases 5–6
├── feature_store/feature_repo/        Phase 4
├── configs/                          As pipeline settings are added
├── tests/test_ingestion.py           Phase 1, extended in later phases
├── reports/                          From Phase 2
├── models/                           From Phase 5
├── dvc.yaml                          Phase 7
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

## Phase 2 — Clean and validate

From the project root in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m src.data.validation
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If the development dependencies are installed, the same tests run with
` .\.venv\Scripts\python.exe -m pytest -q `.

`validate_dataset(data)` accepts the two nullable string columns returned by
`load_legitphish` and returns a clean DataFrame and an aggregate report.
It trims outer whitespace, requires an explicit HTTP(S) scheme and valid
host syntax, and accepts labels 0 (phishing) and 1 (legitimate). URL checks
are local syntax checks. Repeated URL/label pairs keep their first eligible
row. If an eligible URL has both labels, all its eligible rows are excluded.
Duplicates and conflicts compare the trimmed URL text exactly.

The command writes `data/processed/urls_clean.csv` with columns `url` and
`label`, and `reports/data_quality.json` with schema and row counts.
Individual issue counts may overlap. The removal counts are sequential and
sum to the difference between input and clean rows. Both output paths are
ignored by Git. Re-running the command refreshes these files from the raw CSV.

## Phase 3 — URL features and data splits

From the project root in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.features.preprocessing
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The command reads `data/processed/urls_clean.csv` and writes `train.csv`,
`validation.csv`, and `test.csv` to the same directory. Each file contains
ten integer features followed by `label` (0 phishing, 1 legitimate).
`reports/feature_definitions.json` records feature definitions, the keyword
list, grouping rules, and per-split counts. Git ignores these outputs.

`extract_url_features(url)` returns an ordered dictionary of the same ten
features for prediction. Use `FEATURE_COLUMNS` for the model input order.
The extractor checks URL syntax locally. Features use only the supplied
URL text. The training CSVs omit the URL and domain group key.

`preprocess_dataset(data)` takes a clean DataFrame with `url` and integer
`label` columns and returns a dictionary of split DataFrames and a summary.
The fixed seeds are 42 and 43. The target 70/15/15 split applies to groups,
so row totals and class proportions can vary. DNS groups use the registered domain from the ICANN section of the Public
Suffix List bundled with the pinned tldextract release. Network fetching and
caching are disabled. For example, subdomains of example.co.uk share one
group, while another.co.uk has its own. Private hosting suffixes such as
blogspot.com stay together at the registry-level domain. Unknown suffixes
use the final two normalized labels. Bare public suffixes use their hostname,
and IP addresses use their normalized address. The shared extractor counts
subdomains using the same parser.
Keep the test split reserved for final evaluation.

## Phase 4 — Local Feast feature store

From the project root in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m feature_store.materialize
.\.venv\Scripts\python.exe -m pytest -q
```

The command expects the clean CSV and the three Phase 3 split CSVs. It
registers the feature view, retrieves all three splits from the offline
store, checks each result against the original split, materializes the
snapshot into SQLite, and compares a known URL's online vector with
`extract_url_features(url)`. Use `--skip-online` for the historical demo.

- `feature_store/feature_repo/feature_store.yaml` configures a local
  registry, File offline store, and SQLite online store.
- `entities.py` defines the `url_id` join key as SHA-256 of the exact
  trimmed URL in UTF-8. URL text is absent from the Feast artifacts.
- `features.py` declares the Parquet source and ten `Int32` fields using
  the shared Phase 3 feature definitions.
- `feature_views.py` registers the online-enabled `url_lexical_features`
  view. Its zero TTL allows retrieval of this static snapshot without expiry.
- `feature_store/materialize.py` rebuilds IDs in the original split order,
  checks every feature and label against the saved CSVs, and demonstrates
  historical and online retrieval.

The dataset has no observation timestamps. The demo uses a fixed synthetic
event timestamp of **2025-05-22 00:00 UTC**, the date used to identify the
LegitPhish v2 release. It represents a single feature availability snapshot.
A historical query before this snapshot has no matching features. This setup
cannot validate real-world temporal performance.

Generated artifacts live under `feature_store/feature_repo/data/` and are
ignored by Git. `url_features.parquet` holds the ID, timestamp, and features.
`split_entities.parquet` holds the IDs, timestamps, split assignments, and
labels for supervised joins. Labels, IDs, timestamps, and split names are
excluded from `FEATURE_COLUMNS`. `retrieve_historical(store, entities)`
defaults to training and returns the ordered feature columns plus the target.
Use the explicit validation or test split when appropriate.
`retrieve_online(store, url)` fetches the materialized vector for a known
URL. For an unseen URL, the shared extractor can compute its feature vector
straight from the input text. Keep local copies of the snapshot and SQLite
database for the demo. A deployment can use versioned Parquet in object
storage and a shared online store when multiple service replicas need access.

## Phase 5 — Model selection and final holdout

Prepare the local Feast snapshot using Phase 4 if it is missing. From the
project root in VS Code PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.models.train
```

`src/models/train.py` retrieves the existing splits from Feast's historical
store. It fits three configurations each for Logistic Regression, Random
Forest, and XGBoost. A StandardScaler is fitted inside the Logistic
Regression pipeline on the training set. Tree models consume the original
integer feature columns. Classifier random seed is 42.

`src/models/evaluation.py` maps LegitPhish 0 (phishing) and 1 (legitimate)
to classifier 1 (phishing) and 0 (legitimate). Each validation threshold
maximizes phishing recall subject to a 10% validation false-positive rate
cap. Ties prefer lower FPR, then a higher threshold for the same model.
The candidate ranking uses recall, then lower FPR, then higher PR-AUC.
PR-AUC is the trapezoidal area under the precision-recall curve. Confusion
matrix rows are actual and columns predicted, in legitimate, phishing order.

The winner is frozen and saved before one final test retrieval and scoring
pass. The training command refuses a second final run while
`reports/model_evaluation.json` exists. Keep that report with its corresponding
artifact. It contains every validation trial, the best result per model
family, and the winner's final test metrics and confusion matrix.

`src/models/prediction.py` loads the trusted local bundle
`models/phishguard_v1.joblib`, which contains the fitted pipeline, feature
order, threshold, label mapping and version metadata. Prediction derives
the ten features from the input URL using the shared extractor.

```python
from src.models.prediction import load_model, predict_url

bundle = load_model()
print(predict_url("https://example.test/account", bundle))
```

The returned probability is the model's estimate for phishing. The
validation threshold produces the returned phishing/legitimate decision.
`tests/test_models.py` exercises all three models with synthetic URLs,
scoring, label conversion, and artifact load and prediction.

## Phase 6 — MLflow tracking and registry

Install the dependencies, then import the completed Phase 5 report and bundle:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m src.models.tracking
.\.venv\Scripts\python.exe -m pytest -q
```

The import reads saved results. It performs no model fitting or test scoring.
Each validation configuration gets a run in the **PhishGuard AI** experiment,
with parameters, fit time, metrics, and a confusion-matrix artifact. The
selected run also includes the saved test metrics and evaluation report.
Only the selected fitted model was persisted in Phase 5. The other runs
contain their recorded configurations and results.

The local SQLite database and artifacts are in the ignored `mlflow_local/`
directory. View them in the MLflow UI:

```powershell
.\.venv\Scripts\mlflow.exe server --no-serve-artifacts --backend-store-uri sqlite:///mlflow_local/tracking.db --host 127.0.0.1 --port 5000
```

In the browser open `http://127.0.0.1:5000`. Compare runs by
`validation_recall`, `validation_false_positive_rate`, and `validation_pr_auc`.
Open **Models → PhishGuardURLClassifier** to inspect the registered version.
The `candidate` alias is pending approval.

For Member 3, use the versioned URI printed by the importer and the same
local SQLite registry and artifact directory. Use a fixed version for
handoff so an alias change cannot silently alter the served model:

```python
import mlflow
import pandas as pd
from src.models.tracking import TRACKING_DIR, tracking_uri

uri = tracking_uri(TRACKING_DIR)
mlflow.set_tracking_uri(uri)
mlflow.set_registry_uri(uri)
model = mlflow.pyfunc.load_model("models:/PhishGuardURLClassifier/1")
print(model.predict(pd.DataFrame({"url": ["https://example.test/account"]})))
```

The registered model includes the trusted Phase 5 bundle and URL extraction
code. For deployment from another machine, use a shared MLflow registry
and artifact store or package the downloaded version. The import records a
SHA-256 fingerprint of the raw dataset and import-time split fingerprints.
The original training commit is unavailable because Phase 5 did not record
one and the repository has no commit yet. An identical report and bundle
pair reuses completed runs.

## Phase 7 — DVC and reproducibility

From the project root in Windows PowerShell:

```powershell
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
$env:Path = "$(Split-Path -Parent $py);$env:Path"
$env:DVC_SITE_CACHE_DIR = Join-Path $PWD '.dvc\tmp\site-cache'
& $py -m pip install -r requirements-dev.txt
& $py -m dvc status prepare_feast
& $py -m dvc dag
& $py -m dvc repro prepare_feast
& $py -m pytest -q
git status --short --branch
```

The raw CSV has a Git-tracked `.dvc` pointer. `dvc.lock` records the
clean dataset, the unchanged train/validation/test CSVs, reports, and
Feast Parquet snapshot. `prepare_feast` refreshes only the Parquet:
it does not register or materialize the online store. If you need to
initialize the Feast runtime in a fresh checkout, run
`python -m feature_store.materialize --skip-online`.

Use the named **prepare_feast** target for a safe rebuild. A bare
`dvc repro` includes the validation-only candidate training stage.
Phase 5's baseline bundle, final holdout report, and local MLflow
history are separate from DVC pipeline outputs.

### Initial Git commit

From your normal VS Code PowerShell terminal, stage only source, tests,
configuration, and DVC metadata. Review the staged list before committing:

```powershell
git add -- .gitignore .dvc/.gitignore .dvc/config .dvcignore dvc.yaml dvc.lock data/raw/url_features_extracted1.csv.dvc README.md requirements.txt requirements-dev.txt src feature_store tests
git diff --cached --name-only
git diff --cached --check
git commit -m "Initialize PhishGuard Member 1 pipeline and DVC data versions"
git status --short --branch
```

Review the staged list for datasets, databases, model binaries, and local
credential files before committing. They are ignored by the repository.

### Local backup and version recovery

Configure a backup folder outside the checkout. For example:

```powershell
$backup = Join-Path (Split-Path -Parent $PWD) 'phishguard-dvc-backup'
New-Item -ItemType Directory -Force $backup | Out-Null
& $py -m dvc remote add --local -d local-backup $backup
& $py -m dvc push -r local-backup
& $py -m dvc status -c -r local-backup
```

This is same-machine storage. Configure it per checkout, using
`dvc remote modify --local local-backup url $backup` if the name already
exists. There is one DVC dataset version. Once the initial Git commit is
created, recover a revision in a separate checkout so the current files
stay in place:

```powershell
$project = $PWD.Path
$revision = git rev-parse HEAD   # Use the desired older commit when one exists.
$recovery = Join-Path (Split-Path -Parent $project) 'phishguard-recovery'
git clone --no-local $project $recovery
Set-Location $recovery
git checkout --detach $revision
$env:DVC_SITE_CACHE_DIR = Join-Path $PWD '.dvc\tmp\site-cache'
& $py -m dvc remote add --local -d local-backup $backup
& $py -m dvc pull -r local-backup data/raw/url_features_extracted1.csv.dvc validate preprocess prepare_feast
```

For a genuine new data release, update the raw input, run
`python -m dvc add data/raw/url_features_extracted1.csv` and
`python -m dvc repro prepare_feast`, review the pointer and lock changes,
then `git add dvc.lock data/raw/url_features_extracted1.csv.dvc`,
`git commit -m "Version new dataset and regenerated features"`, and
`python -m dvc push -r local-backup`. Update data attribution and version
metadata when the source release changes. When branches conflict on a
pointer, agree on the intended dataset and regenerate `dvc.lock` from it.

### Shared remote and Member 2 handoff

The shared remote is pending the team's DagsHub owner and repository.
Coordinate the project slug with Member 2. Set the DagsHub token through
a local credential manager or inject it into both `AWS_ACCESS_KEY_ID`
and `AWS_SECRET_ACCESS_KEY` for the PowerShell process. Keep the token
out of Git, chat, and command history. With these environment variables
available, use the S3 endpoint shown in the repository's Data remote tab:

```powershell
& $py -m pip install 'dvc[s3]==3.67.1'
$endpoint = Read-Host 'DagsHub S3 endpoint from the Data remote tab'
& $py -m dvc remote add --local dagshub s3://dvc
& $py -m dvc remote modify --local dagshub endpointurl $endpoint
& $py -m dvc push -r dagshub
& $py -m dvc status -c -r dagshub
```

Member 2 can pull the raw pointer and the three data stages shown above
and call `dvc repro prepare_feast` from the
orchestrated data workflow. Schedule candidate fitting explicitly when
new approved training data is ready:

```powershell
& $py -m dvc repro train_candidate
& $py -m src.models.tracking --candidate
```

The candidate trains and tunes with Feast train/validation only, records
the raw and split DVC checksums plus Git HEAD and dirty-tree status, and
writes separate candidate files. The import registers a
`retraining-candidate` alias without changing the original `candidate`
alias or logging test metrics. If a Git-only change should trigger a
fresh candidate, explicitly force that stage after reviewing the change.
Keep Phase 5's already-scored holdout closed. For a future promotion,
freeze the candidate and threshold on validation, obtain a fresh unseen
and separately versioned holdout, and run the evaluation once as a
separately authorized promotion step. Record its result and version
before deciding whether to promote the model.
