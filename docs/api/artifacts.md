# Artifact and Serialization API Reference

## ArtifactIdentity

::: llm_minder.ArtifactIdentity

## ResumePolicy

::: llm_minder.ResumePolicy

## ArtifactStore

::: llm_minder.ArtifactStore

## JsonlArtifactStore

::: llm_minder.JsonlArtifactStore

## SqliteArtifactStore

Indexed SQLite backend for large restartable runs (v0.21). Note that
`SqliteArtifactStore.read_results()` is an **async** classmethod, unlike the
synchronous JSONL convenience. It materializes a finite high-water snapshot
through an operationally read-only path connection. Normal reads participate
in SQLite locking and may create empty WAL/SHM coordination sidecars so a
writer can safely start during inspection; live `store.iter_results()` instead
flushes and inspects that writable store.

::: llm_minder.SqliteArtifactStore

## SqliteDurability

::: llm_minder.SqliteDurability

## ResultSerializationError

::: llm_minder.ResultSerializationError

## Artifact errors

::: llm_minder.ArtifactError

::: llm_minder.ArtifactIdentityError

::: llm_minder.ArtifactSerializationError

::: llm_minder.ArtifactIOError

::: llm_minder.ArtifactFormatError
