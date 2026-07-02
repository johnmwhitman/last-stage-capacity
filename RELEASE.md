# Release Guide — last-stage-capacity

This guide documents the exact steps to publish `last-stage-capacity` to PyPI
using GitHub Actions trusted publishing (OIDC). No API tokens required.

---

## Prerequisites

- Admin access to https://github.com/johnmwhitman/last-stage-capacity
- PyPI account at https://pypi.org (create one if needed)
- A one-time manual setup on PyPI (5 minutes, described below)

---

## Step 1 — Register a Trusted Publisher on PyPI

This step is **manual and one-time**. PyPI needs to know that the GitHub
Actions workflow in this repo is authorized to publish.

### Option A — Pending publisher (project doesn't exist on PyPI yet)

If `last-stage-capacity` is not yet on PyPI, you can register a *pending*
trusted publisher that will create the project on first use.

1. Sign in to https://pypi.org
2. Go to https://pypi.org/manage/account/publishing/
3. Scroll to "Pending publisher" and click "Add a new pending publisher"
4. Fill in:
   - **PyPI project name:** `last-stage-capacity`
   - **Owner:** `johnmwhitman`
   - **Repository name:** `last-stage-capacity`
   - **Workflow filename:** `release.yml`
   - **Environment name:** `pypi`
5. Click "Add"

### Option B — Existing project (project already on PyPI)

1. Sign in to https://pypi.org
2. Go to https://pypi.org/manage/project/last-stage-capacity/settings/publishing/
3. Click "Add a new trusted publisher"
4. Select "GitHub Actions"
5. Fill in:
   - **Owner:** `johnmwhitman`
   - **Repository name:** `last-stage-capacity`
   - **Workflow filename:** `release.yml`
   - **Environment name:** `pypi` (must match the `environment: pypi` in release.yml)
6. Click "Add"

### Verification

After registering, the publisher will appear on the project's publishing page
with a subject like:

```
sub: repo:johnmwhitman/last-stage-capacity:environment:pypi
```

This is the exact claim set our release workflow requests.

---

## Step 2 — Create a GitHub Environment

For security, the release workflow uses a GitHub Actions environment named
`pypi`. This lets you require manual approval before publishing.

1. Go to https://github.com/johnmwhitman/last-stage-capacity/settings/environments
2. Click "New environment"
3. Name: `pypi`
4. (Recommended) Under "Deployment protection rules", enable
   "Required reviewers" and add yourself as a reviewer
5. Click "Save protection rules"

---

## Step 3 — Cut a Release

Once the trusted publisher is registered and the environment exists:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The release workflow at `.github/workflows/release.yml` will:
1. Checkout the repo at the tag
2. Install Python 3.11 + uv
3. Run `uv build` to produce wheel + sdist in `dist/`
4. Use `pypa/gh-action-pypi-publish` with OIDC to upload to PyPI
5. The OIDC token is automatically exchanged for a PyPI API token by
   PyPI trusting the configured publisher

If you enabled "Required reviewers" on the `pypi` environment, the workflow
will pause and wait for your approval at https://github.com/johnmwhitman/last-stage-capacity/actions
before publishing.

---

## Step 4 — Verify

After the workflow completes:

1. Check https://pypi.org/project/last-stage-capacity/ — the version should
   appear within a few minutes
2. Install from PyPI in a fresh venv to verify:

   ```bash
   uv venv /tmp/verify-pypi
   source /tmp/verify-pypi/bin/activate
   pip install last-stage-capacity
   python -c "import last_stage_capacity; print(last_stage_capacity.__all__)"
   ```

Expected output: `['BottleneckBlock', 'CapacityReductionHead', ...]`

---

## Troubleshooting

### "invalid-publisher" error

The trusted publisher is not registered, or the environment/workflow names
don't match exactly. Re-check Step 1 and verify:
- Workflow filename is exactly `release.yml` (not the full path)
- Environment name is exactly `pypi` (lowercase, no spaces)
- Repository owner/name are correct

### "environment protection rules" pause

If the workflow pauses waiting for review, go to the Actions tab and approve
the pending deployment.

### First publish fails with "project name reserved"

If another user registers `last-stage-capacity` between your pending publisher
setup and your first publish, your pending publisher is invalidated. Use
Option B above (existing project path) instead.

---

## Local Build Reference

To test the build locally without publishing:

```bash
# Build
uv build

# Verify the artifacts
ls -la dist/

# Install in a fresh venv and verify
uv venv /tmp/test-lsc
source /tmp/test-lsc/bin/activate
pip install dist/last_stage_capacity-1.0.0-py3-none-any.whl
python -c "from last_stage_capacity import BottleneckBlock; \
    import torch; \
    block = BottleneckBlock(512, 256); \
    out = block(torch.randn(1, 512, 32, 32)); \
    print(f'OK: {out.shape}')"
```

Expected: `OK: torch.Size([1, 256, 32, 32])`

---

## Current State (as of last update)

- ✅ `.github/workflows/release.yml` — builds and publishes on `v*` tag push
- ✅ `.github/workflows/ci.yml` — runs tests on Python 3.10/3.11/3.12
- ✅ `pyproject.toml` — SPDX license, authors, keywords, repository URLs
- ✅ Local build verified — wheel + sdist produced cleanly
- ✅ Fresh-venv install verified — 11 exports, forward pass OK
- ⏳ **PyPI trusted publisher registration** — user action required (Step 1 above)

Once Step 1 is complete, pushing `v1.0.0` will publish automatically.