# JVector Contribution Guidelines

## Pull Requests

- PRs adhere to the checklists posted in the comments.
- PRs may only be merged after:
  - PR branch state is resolved for linear merge to the target branch
  - All CI jobs are executed successfully
  - All checklist items are completed
  - All unresolved discussions are resolved to the reviewers' satisfaction
  - At least one code owner has approved the PR
- Pushes to a PR branch invalidate prior approvals.
- Area code owners have final discretion, and may add or remove reviewers to address size, complexity, or other specific questions as needed.
- Each change is tracked via a corresponding GitHub/Jira issue, and PRs are linked in the issue
- An appropriate level of unit testing is provided for new code.
- Documentation is provided in markdown form for user-facing features.
- Javadoc is provided sufficient to avoid failure by `mvn javadoc:jar`

## Assertions

Some users run JVector with assertions enabled (`-ea`) in production. An assertion on a hot path
(build, search, compaction, or quantization of each vector) must cost a small fraction of the work it
checks and must not repeat that work. Check inputs once where they are created, or check only the
result, and build failure messages only when the check fails.
