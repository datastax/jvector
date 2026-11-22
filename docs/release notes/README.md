## Release Notes

This directory collects and aggregates release notes for each feature added to the JVector library on a release by release basis.

### Guidelines

#### Structure

* Each JVector release has its own sub-directory named after its version as specified in `pom.xml` (`revision`), e.g. `4.0.0-RC.9`.
* Within that sub-directory, each PR that warrants a release note has its own file.

#### File naming

Files must follow the convention **`<PR#>.<tag>.md`**, for example:

| Filename | PR | Section in release notes |
|---|---|---|
| `668.performance.md` | [#668](https://github.com/datastax/jvector/pull/668) | Performance Improvements |
| `659.feature.md` | [#659](https://github.com/datastax/jvector/pull/659) | New Features |
| `672.bugfix.md` | [#672](https://github.com/datastax/jvector/pull/672) | Bug Fixes and Issue Resolutions |

**Valid tags** (controls which section the entry appears in):

| Tag | Section header |
|---|---|
| `feature` | New Features |
| `enhancement` | Enhancements |
| `performance` | Performance Improvements |
| `bugfix` or `fix` | Bug Fixes and Issue Resolutions |
| `docs` | Documentation and Tutorials |
| `testing` | Testing Enhancements |

The workflow groups entries by tag, orders sections as listed above, and inserts a PR link automatically under each entry's heading — you do not need to add the link manually.

#### File content

Each file should contain **only the `###`-level entry content** for that PR — no `##` section header (that is generated from the tag). Example:

```markdown
### My Feature Title

**Description**
A concise but informative description of the feature — motivation, how it works,
and any implications or risks for users.

**How to Enable**
...

**Notes**
...
```

Content requirements:
* Include the motivation for the change and how it works.
* State how the feature is enabled/configured, or explicitly state that no configuration is required.
* Reference any related issues.
* Include code examples or links to documentation where relevant.

#### Usage

When a PR for a new feature is opened, the author creates a file in the appropriate release sub-directory (e.g. `docs/release notes/4.0.0-RC.9/668.performance.md`) and populates it with the entry content.

At release time, the **Generate Release Notes** GitHub Actions workflow is triggered manually. It assembles all entries into a single `<version>.md` file — grouping by tag, sorting by PR number within each group, and injecting PR links — which becomes a publicly available release artifact on GitHub.

#### Backward compatibility

Files that do not match the `<PR#>.<tag>.md` pattern (e.g. the legacy `BugFixes.md` from earlier releases) are appended verbatim at the end of the generated notes. They are expected to carry their own `##` section header.
