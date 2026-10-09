# How a release happens

**Decided 9 October 2026, by the founder: "Yes to all 3".** Nothing ships
unless the tests pass on the exact commit being shipped. Merging a pull request
is the founder's go, so merging one that bumps the version *is* the release
go. A pull request whose tests fail cannot be merged.

Until now a session ran the suite, pushed `main` and published from a laptop.
That relied on everyone running the right tests on the right commit, and on a
PyPI token living on a laptop and in a repository secret.

## How it works

| Workflow | When it runs | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | Every pull request, every push to `main` | Installs the package with every extra and runs the full suite. The job is named **`tests`**. |
| `.github/workflows/release.yml` | After CI passes on a push to `main` | See below. |

What `release.yml` does, step by step:
1. Builds the exact commit CI tested.
2. Checks whether PyPI already has the version in `pyproject.toml`. If it does,
   it stops: most merges change code without bumping the version.
3. Publishes with PyPI trusted publishing (OIDC). There is no token.
4. Tags the commit `v<version>`.

**To release:**
1. Open a pull request that bumps `version` in `pyproject.toml` and
   `__version__` in `stats_compass_core/__init__.py`.
2. Wait for `tests` to pass.
3. Merge it. The package is on PyPI a few minutes later, and the commit is
   tagged.

**If publishing fails:** re-run the `Release` workflow from the Actions tab. It
is safe to repeat: a version PyPI already has is skipped, and an existing tag
is left alone.

## One-off setup, for the founder

Do these once. Until step 1 is done, a version bump merged to `main` fails at
the publish step and ships nothing.

**1. Trusted publisher on PyPI**
1. Sign in at <https://pypi.org>.
2. Open **Your projects → stats-compass-core → Manage → Publishing**.
3. Under **Add a new publisher**, choose **GitHub** and enter exactly:
   - Owner: `oogunbiyi21`
   - Repository name: `stats-compass-core`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
4. Click **Add**.

**2. The `pypi` environment on GitHub**
1. In the repository, open **Settings → Environments → New environment**.
2. Name it `pypi`.
3. Under **Deployment branches and tags**, choose **Selected branches and tags**
   and add `main`. Then only `main` can publish.

**3. Make the tests required**
1. Open **Settings → Rules → Rulesets → New ruleset → New branch ruleset**.
2. Name it, set **Enforcement status** to **Active**, and target the
   **Default branch**.
3. Turn on:
   - **Require a pull request before merging**;
   - **Require status checks to pass**, then add the check **`tests`** (it
     appears once CI has run on a pull request);
   - **Block force pushes**.
4. Save.

After step 3, nobody, Claude sessions included, can push to `main` directly.
Every change, releases included, goes through a pull request.

**4. Old secrets.** If a `PYPI_TOKEN` (or similar) secret exists under
**Settings → Secrets and variables → Actions**, delete it once the first trusted
release has worked. Also revoke the matching token on PyPI under **Account
settings → API tokens**.
