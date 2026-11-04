# async-batch-llm is now llm-minder

This package was renamed to [llm-minder](https://pypi.org/project/llm-minder/) at
1.0. v0.32.0 was the last release of the library under this name.

This release only installs `llm-minder` and keeps old imports working: `import
async_batch_llm` warns with a `DeprecationWarning`, then gives you the `llm_minder`
package itself, including every submodule (`async_batch_llm.testing` is
`llm_minder.testing`). Extras carry over, so `async-batch-llm[openai]` installs
`llm-minder[openai]`.

To move over:

```bash
pip uninstall async-batch-llm
pip install llm-minder            # with the same extras you used before
```

Then replace `async_batch_llm` with `llm_minder` in imports and logging configuration.
1.0 also removes the APIs that 0.27 and 0.28 deprecated and requires Python 3.11; read
the [1.0 migration guide](https://geoff-davis.github.io/llm-minder/migration/v1.0/)
before upgrading. To stay on the old API, pin `async-batch-llm<1`.
