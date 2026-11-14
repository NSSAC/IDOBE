# Expert annotation app

A small web app where invited experts look at each IDOBE outbreak time series
(`raw_data/outbreaks_disease_location.csv`) and record whether it is really an outbreak.
Every answer is stored in a database, and edits are kept as history.

- **Experts** sign in, get series one at a time from a queue, and answer *Outbreak* /
  *Not an outbreak* / *Unsure* with a 1–5 confidence and an optional comment. Keyboard:
  `O` `N` `U`, `1`–`5`, `Enter` to submit, `S` to skip, `B` to go back.
- **Each series goes to up to `TARGET_ANNOTATIONS` experts** (default 3). Experts never see each
  other's answers, or the model's.
- **A model learns from the answers as they come in** (active learning, below). It scores every series,
  steers experts toward the ones it is unsure about, and fills in a label for series no expert has seen.
- **Admins** see coverage, per-expert counts and agreement (Fleiss' kappa) at `/admin`, and can
  download CSVs.

Stack: FastAPI, SQLAlchemy, Alembic and scikit-learn. It runs on SQLite locally and on Postgres when hosted,
with the same code. The frontend is plain HTML/JS with Plotly.

See [UI and deployment review](REVIEW.md) for the current assessment and remaining pilot priorities.
The proposed [annotation rubric](docs/annotation-rubric-draft.md) and
[seven-step tutorial preview](docs/tutorial/index.html) are drafts for study-team review.
Use **Tutorial & examples** at the top of the annotation page to open the guide in a new tab
(`/guide/tutorial/`), keeping the current annotation in place. The guide saves nothing.
It is served by the app, so no separate preview server is needed. The preview file also opens directly in a browser.

## Run locally

The quickest way, from any terminal:

```sh
./annotation/run_local.sh          # sets everything up on first run, then opens http://localhost:8000
./annotation/run_local.sh --lan    # also reachable from other devices on your network
```

Or step by step:

```sh
cd annotation
uv venv --python 3.12 .venv && uv pip install -r requirements-dev.txt   # or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env                      # then set SECRET_KEY

.venv/bin/alembic upgrade head            # create/upgrade the schema (annotation/annotation.db)
.venv/bin/python scripts/load_outbreaks.py
.venv/bin/python scripts/create_user.py admin --admin --generate
.venv/bin/python scripts/create_user.py jdoe --display-name "J. Doe" --affiliation UVA --generate

.venv/bin/uvicorn app.main:app --reload   # http://localhost:8000
```

`load_outbreaks.py` is safe to re-run. It inserts new series, updates changed ones, and warns if a
series that already has annotations changed.

## Managing experts

There is no public sign-up. Admins create accounts on `/admin` → **Users**: pick a username and a
password (at least 8 characters), set a new password for anyone, or deactivate an account (their labels
are kept). The same can be done from the command line:

```sh
python scripts/create_user.py alice --display-name "Alice Smith" --expertise "measles"  # prompts for a password
python scripts/create_user.py alice --reset-password --generate
python scripts/create_user.py alice --deactivate        # keeps her annotations
```

## Data model

| Table | Contents |
|---|---|
| `users` | username, argon2 password hash, display name, affiliation, expertise, admin/active flags |
| `outbreaks` | one row per CSV `unique_id`: metadata plus weekly `values` (value *i* is the week ending `start_date + 7·i`) |
| `annotations` | **append-only**: label, confidence, comment, time spent, timestamp, app version. An edit adds a row; a user's current answer is their latest row for that series |

Schema changes go through Alembic: edit `app/models.py`, then
`alembic revision --autogenerate -m "..."` and `alembic upgrade head`.

## Active learning

With `QUEUE_ORDER=active` (the default):

1. **Warm-up.** Series are served at random until there are `MODEL_MIN_LABELS` (50) Outbreak /
   Not outbreak answers, with at least `MODEL_MIN_PER_CLASS` (10) of each.
2. **Training.** Each series is described by about 20 shape features (`app/ml/features.py`): peak
   relative to the start and end, where the peak falls, how single-humped and how noisy the series is,
   permutation entropy and so on, plus measure type and disease. Each expert answer is one training row,
   weighted by confidence. Unsure and Skip answers are left out. Logistic regression and small boosted
   trees are cross-validated with folds grouped by series, and the one with the lower log loss scores all
   series. This takes about 3 seconds.
3. **Retraining.** Training reruns in the background after every `RETRAIN_EVERY` (20) new answers, so
   experts never wait. Admins can also press *Retrain now* on `/admin`, or run
   `python scripts/train_model.py`.
4. **Serving.** After warm-up, `ACTIVE_RANDOM_FRACTION` (1/3) of series are drawn at random. The rest are
   drawn at random from the `ACTIVE_TOP_K` (200) eligible series the model is least sure about.
5. **Fewer labels for easy series.** Once the model is trained on at least `EARLY_STOP_MIN_LABELS` (200)
   answers and its cross-validated AUC is at least `EARLY_STOP_MIN_AUC` (0.85),
   a series the model is confident about (uncertainty below `CONFIDENT_UNCERTAINTY`, i.e. p < 0.1 or
   p > 0.9) needs only one expert, provided that expert agrees with the model. Disagreements still go to
   up to `TARGET_ANNOTATIONS` experts.

Every annotation records how the series was served (`served_as`: `random`, `uncertain`, `revisit` or
`lookup`), and every served series is logged in `assignments`. `/admin` reports model accuracy and outbreak
share for series whose first non-skip label came from a random draw. This is a descriptive sample of the
eligible queue, not a guaranteed unbiased sample of the entire dataset: eligibility changes as series
are labeled or stopped early, and skips and unresolved votes further affect the evaluated subset.
For population estimates, reserve a fixed random evaluation sample independently of the active queue.

**Final labels** (`/admin` → *All series with final labels*, or `series_labels.csv` from the export
script): the expert majority where one exists, otherwise the model's call, with `label_source` saying which
and `p_outbreak` giving the model's probability.

## Exporting results

```sh
python scripts/export_annotations.py        # writes annotation/exports/
```

- `annotations_current.csv`: each expert's current answer per series
- `annotations_all.csv`: the full history, including edits
- `consensus.csv`: per series: vote counts, majority label, mean confidence
- `agreement.json`: Fleiss' kappa and unanimity, overall and per disease
- `series_labels.csv`: every series with its expert answers, model probability, final label and its source
- `model.json`: the latest model run's metrics

The two annotation CSVs can also be downloaded from `/admin`.

## Configuration

These are environment variables, or `annotation/.env` (see `.env.example`).

| Variable | Default | |
|---|---|---|
| `DATABASE_URL` | `sqlite:///annotation/annotation.db` | `postgresql://…` when hosted (`postgres://` is accepted too) |
| `SECRET_KEY` | random per start | **Required in production**: signs session cookies |
| `TARGET_ANNOTATIONS` | `3` | Experts per series |
| `QUEUE_ORDER` | `active` | `active`: see *Active learning*. `random`: any eligible series. `spread`: least-labeled series first. `complete`: finish series to the target first |
| `ACTIVE_RANDOM_FRACTION` | `0.33` | Share of random draws once a model exists |
| `ACTIVE_TOP_K` | `200` | Pool of most-uncertain series to draw from |
| `MODEL_MIN_LABELS` / `MODEL_MIN_PER_CLASS` | 50 / 10 | When the first model is trained |
| `RETRAIN_EVERY` | `20` | New answers between automatic retrains |
| `AUTO_RETRAIN` | `true` | Retrain in the background after answers come in |
| `CONFIDENT_UNCERTAINTY` / `EARLY_STOP_MIN_LABELS` / `EARLY_STOP_MIN_AUC` | 0.2 / 200 / 0.85 | When one agreeing expert is enough |
| `COOKIE_SECURE` | `false` (`true` in Docker) | Send cookies over HTTPS only |
| `SESSION_MAX_AGE` | 14 days | |
| `LOGIN_MAX_FAILURES` / `LOGIN_WINDOW_SECONDS` | 10 / 900 | Failed-login limit per IP and per username |

## Hosting

### GitHub Pages versus the annotation service

[GitHub Pages serves static sites and cannot run Python server code](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).
Publishing `app/static/` alone will not provide login, queues, saved annotations, or admin exports.
The current frontend expects FastAPI on the same origin (`/api/...`, `/login`, `/admin`).

The simplest deployment is to host the complete app and Postgres together using the Render Blueprint
below. A `github.io` project page can explain the study and link to the hosted annotation app.
For a scheduled expert pilot, consider an always-on web service:
[Render's free web services sleep after inactivity](https://render.com/docs/free), which delays sign-in.

Serving the interactive frontend itself from `github.io` would require a separately hosted API,
configurable API and asset paths (including a repository subpath such as `/IDOBE/`), explicit credentialed
CORS, and a reviewed cross-site authentication/CSRF strategy. The existing `SameSite=Lax` session
cookie is designed for same-origin deployment; changing URLs alone is insufficient.
Do not publish the SQLite database, `.env`, or annotation exports as static assets.

### Render / container hosting

**Render:** `render.yaml` at the repository root is a Blueprint that creates the web service and its
Postgres database (Render dashboard → New → Blueprint). After the first deploy, load the data and create
accounts from your own computer, using the database's *External Database URL* from the Render dashboard
(free web services have no shell):

```sh
cd annotation
DATABASE_URL='<External Database URL>' .venv/bin/python scripts/load_outbreaks.py
DATABASE_URL='<External Database URL>' .venv/bin/python scripts/create_user.py admin --admin --generate
```

The image is built from the **repository root**, so the loader can reach `raw_data/`:

```sh
docker build -f annotation/Dockerfile -t idobe-annotation .
```

On start the container runs `alembic upgrade head` and then serves on `$PORT` (default 8000).
To deploy, for example on Render, Fly.io or a UVA VM:

1. Create a managed Postgres database and set `DATABASE_URL`, `SECRET_KEY` and `TARGET_ANNOTATIONS`.
2. Deploy the image behind the platform's HTTPS.
3. Run the one-off commands in the container: `python scripts/load_outbreaks.py` and
   `python scripts/create_user.py admin --admin --generate`.

To try the production setup locally with Postgres:

```sh
docker compose -f annotation/compose.yaml up --build
docker compose -f annotation/compose.yaml exec app python scripts/load_outbreaks.py
docker compose -f annotation/compose.yaml exec app python scripts/create_user.py admin --admin --generate
```

Notes:
- The login rate limiter and the retraining lock are in memory, so run a single instance (one
  uvicorn worker) or move them to shared storage before scaling out.
- Back up the Postgres database regularly. The `annotations` table is the irreplaceable data.

## Tests

```sh
.venv/bin/python -m pytest
TEST_DATABASE_URL=postgresql://user:pass@localhost/idobe_test .venv/bin/python -m pytest   # against Postgres
node --test tests/frontend.test.cjs  # Node 18+; workflow checks, no npm install needed
```


## Segment review

`/segments` shows experts the **whole series** with a segmentation method's proposed outbreak
windows. For each window the expert accepts, rejects, or drags its edges (or edits its dates) to
adjust it, and can add outbreaks the method missed or mark the series unusable. Reviews are
append-only, like outbreak labels, and each series is reviewed by up to `TARGET_ANNOTATIONS` experts.

Load a run (one method's output) with `scripts/load_segments.py`:

```
python scripts/load_segments.py --legacy                       # current published windows, respiratory series
python scripts/load_segments.py --series S.csv --segments G.csv --run-name "method X v2"
python scripts/load_segments.py --series S.csv --method seasonal-threshold
```

`run_local.sh` loads the legacy respiratory run on first start. Export with
`scripts/export_segment_reviews.py` (or `/api/seg/export.csv` as an admin); it also writes the expert
consensus windows that `python -m pipeline.build_benchmark` turns into the benchmark CSV. File
formats and the method interface are in `../pipeline/README.md`.
