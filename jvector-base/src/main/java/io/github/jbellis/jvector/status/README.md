<!--
~ Copyright DataStax, Inc.
~
~ Licensed under the Apache License, Version 2.0 (the "License");
~ you may not use this file except in compliance with the License.
~ You may obtain a copy of the License at
~
~ http://www.apache.org/licenses/LICENSE-2.0
~
~ Unless required by applicable law or agreed to in writing, software
~ distributed under the License is distributed on an "AS IS" BASIS,
~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
~ See the License for the specific language governing permissions and
~ limitations under the License.
-->
# Status API

This internal API was added to solve a few problems around a common theme:

* Lack of visibility during long-running tests.
* No easy way to instrument structured tasks.
* Lack of facilities to enable user-visible task status when jvector is embedded.

## Design Requirements and Implementation Strategies

* The Status API must be minimally invasive to other code.
  * Synchronous and Asynchronous code must be supported.
  * Tracked tasks can be instrumented with a decorator API OR
  * Tracked tasks can be wrapped with functors at instrumentation time, should existing properties be sufficient to interpret task status.
* The Status API must fit naturally to non-trivial task structure.
* The Status API must not assume a particular output form. It could be the primary view for the user, or it could be a programmatic source of task information when jvector is embedded.
* The Status API must provide reliable views of task state.
  * Try-with-resources is used to align tracker instances to critical sections.

```mermaid
```