# ADR 0005: Static publishing on the existing site

**Status**: accepted 14 September 2026; open questions 3 and 4 decided 16 September 2026.

## Context

The pipeline runs locally on demand. The forecast is a weekly snapshot, not a query
surface. The repository is public and intended to support written pieces on an existing
Astro site with an S3, CloudFront, Terraform and GitHub Actions OIDC deployment path.

## Options considered

1. **Streamlit or Dash app.** Needs a running server; the pipeline is local, so the app
   would either sit stale or require the always-on infrastructure the non-goals rule out.
2. **S3 bucket plus build-time fetch.** Provisions a bucket and IAM role to serve a file
   GitHub already serves publicly, and reintroduces a build-time network dependency.
3. **Git submodule.** Pins the site to SHA bumps and complicates CI checkout.
4. **`repository_dispatch` from the forecast repo.** Immediate, but needs a cross-repo
   write credential for a pipeline that is run by hand anyway.
5. **Pull and commit, driven from the site repo.** A workflow in the site repo, on
   `workflow_dispatch` plus a weekly schedule, fetches `forecast-latest.json` from the
   forecast repo's raw URL, compares content hashes, and commits it into `src/data/` if
   changed. The existing Astro build fires on that commit.

## Decision

Option 5. The site imports a local JSON file, so a third-party outage cannot break a
site build, and the site repo's history becomes a second, independent record of what
was published and when.

Charts are static SVG rendered at build time (Observable Plot or D3 in an Astro
component). No client-side charting library, no runtime data fetching.

Visualisations, in priority order: the 20 by 20 position probability matrix; the
per-club points distribution ridgeline; forecast evolution across gameweeks from the
committed artifact history; the calibration chart from `docs/evaluation.md`.

**Open question 3 (show the market record on the front page): yes.** It is less
flattering and more honest, and design.md section 13 already commits to stating it
plainly.

**Open question 4 (publish `fixtures-latest.json` from the first forecast): yes.** The
evaluation harness exists, so per-fixture predictions can be scored as soon as results
land; publishing predictions with no scoring attached was the concern, and it no longer
applies.

## Consequences

- No new infrastructure. The site workflow is roughly twenty lines of YAML.
- Immediacy is bounded by the weekly schedule or a manual dispatch; the operator is at
  the keyboard when the pipeline runs and can trigger the site workflow in the same
  sitting.
- **This is now true (19 September 2026, story: weekly scheduled pipeline):**
  `.github/workflows/weekly-refresh.yml` runs the forecast pipeline itself on a
  schedule, unattended -- nobody is necessarily at the keyboard when a new
  `forecast-latest.json` lands. The site's own weekly-schedule pull still works (it
  just polls less promptly than a push would), but a `repository_dispatch` from this
  repo to the site repo is now a reasonable option where it wasn't before: there's a
  well-defined moment (the commit-back step at the end of the workflow) to fire it
  from, rather than needing a cross-repo credential for a pipeline that was run by
  hand anyway. Not part of this change -- revisit if the weekly-schedule pull proves
  too slow in practice.
