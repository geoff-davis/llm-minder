# Which Migration Guide to Read

Find the release you're upgrading from and follow its row: each row names a guide,
then the row to continue from. Releases without a guide record their changes in the
[changelog](https://github.com/geoff-davis/llm-minder/blob/main/CHANGELOG.md); read
its entries for each release you cross.

Within 1.x, upgrades need no guide: the [API stability promise](../stability.md)
covers every stable name.

| Upgrading from | Read |
| --- | --- |
| v0.32 | [1.0](v1.0.md) |
| v0.28 to v0.31 | Upgrade to v0.32 and clear its deprecation warnings, then [1.0](v1.0.md) |
| v0.27 | [v0.28](v0.28.md), then the v0.28 row |
| v0.26 | [v0.27](v0.27.md), then the v0.27 row |
| v0.25 | [v0.26](v0.26.md), then the v0.26 row |
| v0.24 | [v0.25](v0.25.md), then the v0.25 row |
| v0.23 | [v0.24](v0.24.md), then the v0.24 row |
| v0.21 | [v0.22](v0.22.md), then the v0.23 row |
| v0.22 | the v0.23 row |
| v0.18 to v0.20 | [v0.20](v0.20.md) if you are before v0.20 (v0.19 was never published), then the v0.21 row |
| v0.8 to v0.17 | [v0.10](v0.10.md) if you are before v0.10, then the v0.18 row |
| v0.4 to v0.7 | [v0.4](v0.4.md), then the v0.8 row |
| v0.1 to v0.3 | [v0.1](v0.1.md), then the v0.4 row |

Guides for v0.0.x to v0.3, written before the first public release, are kept in the
[repository archive](https://github.com/geoff-davis/llm-minder/blob/main/docs/archive/README.md);
they aren't part of this site.

## Release evidence

Measured evidence published with past releases:

- [v0.20 release notes](../evidence/release-notes-v0.20.md)
- [v0.22 scale evidence](../evidence/v0.22-scale-evidence.md)
- [SQLite maintenance evidence](../evidence/sqlite-maintenance-evidence.md)
