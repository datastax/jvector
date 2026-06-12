## Release Notes

This directory collects and aggregates release notes for each feature added to the JVector library on a release by release basis.

### Guidelines 
* Structure
    * Each JVector release has its own sub-directory within this directory, named according to the release version as specified in the `pom.xml` file as `revision`, e.g. `4.0.0-RC.9`
    * Within the sub-directory for each release each feature is represented by its own independent file containing the release details for that feature. 
* Content
    * Each feature file should contain a concise but informative description of the feature, including any relevant details such as the motivation for the feature, how it works, and any important implications or considerations for users, including any known risks. 
    * If applicable, the feature file should also include links to relevant documentation, code examples, or other resources that can help users understand and utilize the feature effectively.
    * The documentation for every feature *must* at a minimum include details on how that feature is enabled and configured, or, if no explicit enablement and / or configuration is necessary, this must be stated.
    * Documentation for each feature should also include reference to any existing issues that are related to the feature.
* Usage
  * When a PR for a new feature is created, the author of the pull request should create a new file for that feature in the appropriate release sub-directory and populate it with the relevant release notes content as described above.
  * At the time when the release is cut all of the release notes for that release will be aggregated into a single release notes document that will become a release artifact publicly available in github under the releases section of the JVector repository.