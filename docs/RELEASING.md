# Releasing WakewordKit

Release preparation happens in a separate branch and pull request. Merging a
preparation PR does not publish a package. The next prepared release is 0.6.0;
0.5.0 already exists on PyPI, even though the latest earlier GitHub release/tag
is v0.4.0. Do not reuse either published version/tag.

## Prepare and review

1. Change `wakewordkit.__version__` (Hatch's single version source) and run
   `uv lock` to check lock consistency. uv omits the dynamic local project
   version from this lockfile, so a version-only bump may leave it unchanged.
2. Add release notes and migration guidance to `CHANGELOG.md`; keep the release
   marked unreleased until publication. Confirm intentional API changes.
3. Run the same checks as CI:

   ```bash
   uv sync --locked --dev
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy
   uv run pytest
   uv run python tools/wheel_smoke.py
   ```

4. Build the source distribution and wheel into a version-specific directory:

   ```bash
   uv build --out-dir dist/0.6.0
   uvx twine check dist/0.6.0/wakewordkit-0.6.0.tar.gz dist/0.6.0/wakewordkit-0.6.0-py3-none-any.whl
   ```

   `tools/wheel_smoke.py` checks a built wheel in a fresh environment outside
   the checkout, including version consistency, py.typed, model files and real
   inference. Also install/build the sdist in a fresh environment before release;
   this checks that its packaging inputs are complete.

5. Open the isolated release PR against main. Require the Linux/Windows,
   Python 3.12/3.13/3.14 CI matrix to pass before merging. If main changes, rerun
   checks on the final release commit. Record the artifacts' SHA256 hashes.

## Publish after release approval

These are manual maintainer actions; the preparation PR performs none of them.
Use the intended version in each path/tag and confirm it is still unused on
PyPI and GitHub. Use the repository's existing publishing credentials locally;
never put tokens in source control, shell arguments, PRs or logs.

1. Finalize the changelog date, merge the approved PR, fetch main and check out
   the exact reviewed release commit. Rebuild and recheck both distributions
   from that commit. Do not publish artifacts from another checkout/version.
2. Publish only the two explicit distribution paths with `uv publish`. Use an
   interactive or configured credential provider; retain the artifact hashes.
3. Create/push tag `v0.6.0` at the release commit and create a GitHub release
   using the corresponding changelog entry. Attach the same checked artifacts.
4. Verify PyPI reports 0.6.0, install it in a new environment and run the real
   short-frame/finish smoke test. Confirm the GitHub tag, package metadata and
   public `wakewordkit.__version__` agree.

A representative recognition corpus is still needed before making claims about
false activations, misses or superiority. See [EVALUATION.md](EVALUATION.md).
