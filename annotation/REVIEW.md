# Expert annotation UI review

Reviewed on 2026-10-20 against `expert-annotation-ui`, starting at `e100124d`.

## Assessment

This is a promising expert-pilot application. The plain HTML/JS approach is appropriate to its scope;
a frontend framework migration would add little immediate value. The annotation screen has a clear
sequence: inspect metadata and the chart, choose a label, choose confidence, optionally explain, submit.
Keyboard shortcuts, responsive styling, a numeric table alternative, and revisitable answers are useful
for repeated expert work. Backend strengths include append-only answer history, role checks, Argon2
password hashing, schema migrations, and explicit expert/model provenance in exports.

The visual assessment here is based on source inspection. No connected browser was available, so
desktop/mobile rendering, chart interactions, and screen-reader behavior still need a browser pass.

## Small improvements included

- Keep save feedback outside the annotation card so it remains visible when the queue finishes.
- Distinguish a confirmed save from subsequent queue/history refresh failures. If the next series
  cannot load, lock the saved form and offer a retry that does not submit another answer.
- Provide **Continue annotating** when reviewing earlier answers or checking an exhausted queue.
- Preserve Back history if fetching the previous series fails.
- Lock answer controls during requests and suppress repeated keyboard shortcuts.
- Add visible keyboard focus to links, disclosure summaries, and clickable history rows.
- Qualify the dashboard's claim about the random sample; no sampling or model logic was changed.
- Explain the GitHub Pages limitation and the simpler same-origin hosting route in the README.

## Before inviting experts

| Priority | Finding | Suggested next step |
|---|---|---|
| High | The screen asks “Is this an outbreak?” without an operational definition, examples, or instructions for truncated curves/reporting artifacts. Experts can give consistent-looking answers to different questions. | Agree on a brief annotation rubric with the study team. Add expandable instructions, approved examples, and a rubric version recorded with answers. |
| High | Unsaved comments/choices can be lost through navigation or a session-expiry redirect. | Add a dirty-form warning and scoped draft recovery, with a clear distinction between a draft and a saved answer. |
| High | A POST that commits but loses its response is still ambiguous. The included fix handles refresh failures after a confirmed POST, not this network case. | Add a client submission ID and server uniqueness constraint so retries cannot create duplicate history rows. |
| Medium | Plotly loads from a CDN, and `renderChart()` assumes it exists. A blocked CDN can interrupt form initialization. | Bundle the pinned asset or add an explicit chart-error state and reliable table fallback. |
| Medium | Login, admin mutations, and history pagination have less consistent loading/error handling than the main save flow. | Standardize request handling, prevent duplicate submissions, and preserve failed pagination offsets. |
| Medium | Password reset changes the hash, but sessions identify users by ID and do not check a session version. Existing sessions remain valid. | Add session invalidation on password reset before treating reset as account recovery. |
| Medium | The local launcher ignores migration failures (`alembic ... \| grep ... \|\| true`). | Make migration failure stop startup with an actionable error. |

## Research decisions to resolve

1. **Random-sample claims.** `queue.py` samples from a changing eligible pool; early stopping and
   previous labels remove series. `train.py` then evaluates only resolved binary majorities with
   out-of-fold predictions. This does not establish unbiased population accuracy/prevalence. Also,
   `spread` and `complete` selections are currently logged as `random`. Reserve and track an independent
   random evaluation cohort before using these numbers as study outcomes.
2. **Early stopping and disagreement.** One expert agreeing with a confident model can end collection.
   In `ml/results.py`, Unsure votes are excluded from the binary majority, and ties or Unsure-only
   results can receive a model-derived final label. Decide whether an expert gold-standard dataset
   should instead require independent agreement or adjudication. Keep `label_source` in downstream work.
3. **Blinding.** Expert-only accounts cannot access model/admin APIs, which is good. Admins can see
   model probabilities and open those series for annotation. Use separate expert-only accounts and
   a study protocol if administrators also contribute blinded labels.
4. **Concurrent work.** `TARGET_ANNOTATIONS` is a queue filter, not a hard concurrency cap. Several
   experts can receive the same eligible series before anyone submits. Decide whether modest
   over-collection is acceptable or whether assignment reservations are needed.

## Deployment recommendation

Use the existing `render.yaml` and container to serve the frontend and API from one HTTPS origin,
with managed Postgres. Keep GitHub Pages for the public study/project page and link to the app.
This retains the current cookie and URL design. GitHub Pages cannot execute the FastAPI service.
For a real pilot, choose availability suitable for booked expert sessions and verify database backup
and restore. The current in-memory retraining lock and login limiter assume a single worker.

Provider references: [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site),
[FastAPI on Render](https://render.com/docs/deploy-fastapi),
[Render free-service limitations](https://render.com/docs/free).

## Validation

- Existing Python suite: 37 passed on SQLite.
- Frontend workflow regressions: 7 passed using Node's built-in test runner and DOM/API doubles.
- JavaScript syntax and whitespace checks passed.
- No live deployment, Postgres test run, or visual browser QA was performed.
