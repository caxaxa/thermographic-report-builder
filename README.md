# Thermographic Report Builder

Clean-room implementation of the container that produces the thermographic
report after masking/annotation. The legacy prototype now lives under
`LEGACY_CODE/` purely for reference—please migrate functionality into the
structured modules under `src/thermographic_report_builder/`.

## Repository Layout
- `LEGACY_CODE/` – original prototype pulled from `greta-manual`.
- `src/thermographic_report_builder/` – new production code organised by
  concern (configuration, S3 I/O, processing, reporting).
- `tests/` – unit/integration tests to accompany the new implementation.
- `pyproject.toml` – project metadata and dependencies.

## Next Steps
1. Port algorithms from `LEGACY_CODE` into self-contained processing/report
   modules.
2. Implement the real `ThermographicPipeline` download/process/upload flow for
   AWS Batch.
3. Replace the placeholder test with fixtures that exercise masking and report
   generation end-to-end.
4. Document container entry points and environment variables as they solidify.

## Local Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```
