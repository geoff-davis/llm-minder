# Regenerating the Terminal Demo

The README animation renders the standard output of the credential-free
[`example_cooldown_resume.py`](https://github.com/geoff-davis/llm-minder/blob/main/examples/example_cooldown_resume.py):
one 429 pauses every worker, the run is killed partway, and a rerun replays the
checkpointed items and calls the client only for the rest. The library's own log
lines go to standard error and aren't shown. Run the example first; if its output
changed, update `OUTPUT_LINES` in the generator, then regenerate the asset:

```bash
uv run python examples/example_cooldown_resume.py
uv run --with pillow python scripts/generate_terminal_demo.py
```

The file keeps its historical name, `v0.20-quickstart.gif`, because the README
and tests reference it.

Pillow is a maintainer-only generation tool and is not a runtime dependency.
The generated `docs/assets/v0.20-quickstart.gif` must remain below 2 MB and must
not contain machine-specific paths, credentials, or terminal history.
