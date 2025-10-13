# JVector APIs Module

This module is meant to hold small units of logic which are used by other modules.

Good reasons to put an API in this module:
* It allows components from different modules to work together in a type safe way.
* It provides a well-defined unit of functionality that is needed by multiple other modules, but you don't want them to have a stronger dependency relationship to share it.
* It is foundational for other modules in the project.
* It provides necessary functionality, but external dependencies for the same come at too high of a cost.

If new functionality does not meet one of these criteria, then it probably belongs in another module.

As a foundational layer in the jvector project, code added to this module will be subject to more stringent standards. Expect to provide high test coverage, solid javadoc, and good examples for packages added here.

