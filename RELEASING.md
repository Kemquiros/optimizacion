# Releasing khumbu

Publishing runs on **PyPI Trusted Publishing**, so there is no API token anywhere in this
repository or in GitHub secrets. GitHub proves the repository's identity to PyPI over OIDC at the
moment of upload.

## One-time setup (on PyPI, by the maintainer)

1. Sign in at <https://pypi.org> and open **Your projects → Publishing**, or, for a name not yet
   registered, **Add a pending publisher**.
2. Register a **GitHub** publisher with exactly these values:

   | Field | Value |
   |---|---|
   | PyPI project name | `khumbu` |
   | Owner | `Kemquiros` |
   | Repository name | `khumbu` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. In this repository, open **Settings → Environments** and create an environment named `pypi`.
   Adding a required reviewer there is worth doing: it makes every publish a deliberate act rather
   than a consequence of pushing a tag.

Nothing else is needed. No token is generated, so none can leak.

## Every release

```bash
# 1. Bump the version in the three places that must agree
#    pyproject.toml, CITATION.cff, src/khumbu/__init__.py
# 2. Update the figures and the benchmark if behaviour changed
python scripts/make_figures.py
python -m khumbu.benchmark

# 3. Tag and push
git tag -a v2.1.0 -m "khumbu 2.1.0"
git push origin v2.1.0
```

The workflow then, in order: **checks that the tag matches the packaged version** and stops if it
does not, runs lint, strict typing and the full test suite, builds the wheel and sdist, validates
them with `twine check`, and only then publishes.

A release that fails any gate never reaches PyPI. That ordering is deliberate: a broken package on
the index cannot be replaced, only yanked.
