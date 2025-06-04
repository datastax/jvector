# JVector API

This module is expected to be come the official API of JVector.

## Background

Presently, the jvector codebase provides a variety of ways to get to its internal functionality. 
It has evolved quickly with new features and algorithms being made available incrementally. This 
inevitably leads to different views of how jvector works, as embedding systems and testers may 
choose a different _form_ of jvector to use, by calling path, feature set, or configuration. We 
need a way to prescribe good usage patterns around a stable interface, and simplify the view of 
JVector as much as possible for common cases. That is what the API should do.

Creating a standard API for using JVector is no longer a nice-to-have. It is becoming essential 
to the success of users and developers as a way to provide a usage contract for key JVector 
features.

Since putting this API boundary in place and then adopting it would mean some disruption for
both internal and external users, some coordination and planning with current users is
essential. We will work with them to ensure there is no abrupt change and that we support 
adoption of the API boundary by helping with porting if necessary.

## Goals

* Provide a standards-friendly and well defined API for JVector.
* Work with current users to negotiate a set of API primitives and methods which they can port to.
* Establish testing patterns for the API as the canonical interface, demonstrating how it can be 
  used.
* Model API structure after JPMS modules as much as possible.
* Ensure that the API is thoroughly documented.
* Work with embedding system maintainers to deprecate and move from the non-API usage path.

## Guidelines

Some guidelines can help us make the JVector API more robust and usable. These are not set in 
stone. As the project matures around the API, some of these might become enforced.

### Modularity

👉 JPMS-like

In general, we should use the conventions of JPMS as a strict guideline, but without trying to 
make JVector itself a fully compliant JPMS module. Should it be possible to make JVector into 
a fully standard JPMS module, that would be a different scope of effort. However, the JPMS 
standards do provide the strictest model for best practices, and adhering to them as much as 
possible can go far towards interoperability and setting clear expectations.

👉 The user facing API should be contained in a single package, for which there is no other purpose.

When a user wants to know what the officially supported API looks like, they should ideally have 
to know only one thing: the package it is in.

👉 The API should be kept in its own module, to be used as a lingua-franca between other modules where needed.

Having an API module can allow for the caller runtime and the JVector implementation to be 
indirectly wired together. This is an example of "low coupling, high cohesion".  

👉 The external API should be treated distinctly from any internal APIs, since these will serve different callers and roles.

The user-facing API is "The JVector API", but there are plenty of internal APIs. There may even 
be specifically designed APIs to support internal evolution and modularity. It is important to 
think of these as distinct service boundaries. In other words, "The JVector API" should be only 
that.


### Lifecycle

👉 Semantic Versioning

We should use semantic versioning and static analysis tools to ensure it is well-maintained.

### Tooling

👉 Static analysis should be used to ensure that only the designated API boundary is used for 
"client" code.

### Type System

👉 Types used in external APIs should be kept to a minimum, such as primitives, core objects 
like Strings, collections, arrays, and supporting types like Optional.

👉 Value types should be used by default, favoring composability, functional patterns, and immutability.

👉 Domain types which are algebraic or directly representational of a key API concept can be 
included in the API.

👉 Mutable types imply their own API and should be made explicit. When possible mutability should 
be avoided.

👉 Nullability shall be avoided completely. `Optional` should be used instead, for example. 

