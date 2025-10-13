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

## Architectural Model: Scopes, Trackers, and Contexts

The API enforces a clear three-level hierarchy:

### 1. StatusContext (Top Level)
- **Role**: Coordinator for an entire tracking session
- **Owns**: One StatusMonitor, multiple StatusSinks, multiple TrackerScopes
- **Lifecycle**: Lives for the duration of the operation being tracked
- **Creation**: `new StatusContext("operation-name")`

### 2. TrackerScope (Middle Level)
- **Role**: Organizational container with NO progress or state
- **Purpose**: Groups related tasks hierarchically
- **Can contain**: Child scopes (nested organization) + Task trackers (actual work)
- **Cannot do**: Have its own progress/state, be tracked by monitor
- **Lifecycle**: Closed when organization is no longer needed
- **Creation**: `context.createScope("scope-name")` or `parentScope.createChildScope("name")`

### 3. StatusTracker (Leaf Level)
- **Role**: Represents actual work with progress and state
- **Purpose**: Tracks a specific task's execution
- **Can do**: Report progress (0.0-1.0), report state (PENDING/RUNNING/SUCCESS/FAILED)
- **Cannot do**: Have children (enforced - must be leaf nodes)
- **Lifecycle**: Closed when task completes
- **Creation**: `scope.trackTask(task)` or `context.track(task)` (for scopeless tasks)

### Hierarchy Example

```
StatusContext "DataPipeline"
  │
  ├─ TrackerScope "Ingestion" (organizational - no progress)
  │    ├─ StatusTracker: LoadCSV (leaf - 45% complete, RUNNING)
  │    └─ StatusTracker: ValidateSchema (leaf - 100% complete, SUCCESS)
  │
  └─ TrackerScope "Processing" (organizational - no progress)
       ├─ StatusTracker: Transform (leaf - 30% complete, RUNNING)
       └─ TrackerScope "Indexing" (nested organizational scope)
            └─ StatusTracker: BuildIndex (leaf - PENDING)
```

### Key Design Rules

1. **Scopes organize, Trackers execute**
   - Scopes have no progress/state
   - Only trackers report progress

2. **Trackers are always leaf nodes**
   - Cannot create children
   - If you need hierarchy, use nested scopes

3. **One context per operation**
   - Context owns the monitor and sinks
   - All scopes and trackers belong to one context

4. **Scopes determine completion**
   - A scope is complete when all its children (scopes + trackers) are complete
   - Provides natural aggregation without scopes needing their own state

## Status Flow Architecture

Status information flows unidirectionally from the tracked task through the monitoring infrastructure to the sinks:

```
┌─────────────┐
│ Tracked     │ (application task object)
│ Task (T)    │
└──────┬──────┘
       │
       │ StatusTracker.refreshAndGetStatus()
       │ observes via statusFunction.apply(tracked)
       ↓
┌─────────────┐
│StatusTracker│ (caches status, updates timing)
│  (Leaf)     │
└──────┬──────┘
       │
       │ StatusMonitor.pollTracker()
       │ calls tracker.refreshAndGetStatus()
       ↓
┌─────────────┐
│StatusContext│ (routes to sinks)
└──────┬──────┘
       │
       │ taskUpdate(tracker, status)
       ↓
┌─────────────┐
│   Sinks     │ (display/log/metrics)
└─────────────┘
```

**Key principles:**
- StatusTracker owns observation of its tracked object
- StatusMonitor polls trackers on a schedule (scopes are NOT polled)
- StatusContext routes status updates to all registered sinks
- Status flows one way: Task → Tracker → Monitor → Context → Sinks
- No back-flow of status information into tasks or trackers
- Scopes provide structure but don't participate in status flow

## Usage Patterns

### Basic Usage: Single Task

For simple operations with no hierarchy:

```java
try (StatusContext context = new StatusContext("simple-operation")) {
    context.addSink(new ConsoleLoggerSink());

    try (StatusTracker<MyTask> tracker = context.track(new MyTask())) {
        // Task executes and reports progress automatically
        tracker.getTracked().execute();
    }
}
```

### Recommended Pattern: Scopes for Organization

For complex operations with multiple related tasks:

```java
try (StatusContext context = new StatusContext("data-pipeline")) {
    context.addSink(ConsolePanelSink.builder().build());

    // Create organizational scopes
    try (TrackerScope ingestionScope = context.createScope("Ingestion");
         TrackerScope processingScope = context.createScope("Processing")) {

        // Add tasks as leaf nodes within scopes
        StatusTracker<LoadTask> loader = ingestionScope.trackTask(new LoadTask());
        StatusTracker<ValidateTask> validator = ingestionScope.trackTask(new ValidateTask());
        StatusTracker<TransformTask> transformer = processingScope.trackTask(new TransformTask());

        // Execute tasks...
        loader.getTracked().execute();
        validator.getTracked().execute();
        transformer.getTracked().execute();

        // Trackers close automatically via try-with-resources
    }
    // Scopes close automatically
}
// Context closes automatically
```

### Advanced Pattern: Nested Scopes

For deep organizational hierarchies:

```java
try (StatusContext context = new StatusContext("etl-pipeline");
     TrackerScope etlScope = context.createScope("ETL")) {

    // First level of organization
    TrackerScope extractScope = etlScope.createChildScope("Extract");
    TrackerScope transformScope = etlScope.createChildScope("Transform");
    TrackerScope loadScope = etlScope.createChildScope("Load");

    // Second level of organization under Transform
    TrackerScope cleaningScope = transformScope.createChildScope("Cleaning");
    TrackerScope enrichmentScope = transformScope.createChildScope("Enrichment");

    // Actual work happens at leaf level
    StatusTracker<Task> extractTask = extractScope.trackTask(new ExtractTask());
    StatusTracker<Task> cleanTask = cleaningScope.trackTask(new CleanTask());
    StatusTracker<Task> enrichTask = enrichmentScope.trackTask(new EnrichTask());
    StatusTracker<Task> loadTask = loadScope.trackTask(new LoadTask());

    // Check completion at any level
    boolean cleaningDone = cleaningScope.isComplete();
    boolean transformDone = transformScope.isComplete();
    boolean allDone = etlScope.isComplete();
}
```

### Relationship Summary

| Component | Purpose | Can Have Children? | Has Progress? | Polled by Monitor? |
|-----------|---------|-------------------|---------------|-------------------|
| **StatusContext** | Session coordinator | Yes (scopes, trackers) | No | No |
| **TrackerScope** | Organizational container | Yes (scopes, trackers) | No | No |
| **StatusTracker** | Work unit (leaf) | **No** (enforced) | **Yes** | **Yes** |

### Common Mistakes to Avoid

❌ **DON'T**: Try to create children from trackers within a scope
```java
TrackerScope scope = context.createScope("Work");
StatusTracker<Task> task = scope.trackTask(new Task());
task.createChild(new SubTask()); // ❌ THROWS IllegalStateException
```

✅ **DO**: Use nested scopes for hierarchy
```java
TrackerScope scope = context.createScope("Work");
TrackerScope subScope = scope.createChildScope("SubWork");
StatusTracker<Task> task = scope.trackTask(new Task());        // ✓ Leaf node
StatusTracker<Task> subTask = subScope.trackTask(new SubTask()); // ✓ Leaf node
```

❌ **DON'T**: Forget to close scopes
```java
TrackerScope scope = context.createScope("Work");
StatusTracker<Task> task = scope.trackTask(new Task());
task.close();
// ❌ scope never closed - memory leak!
```

✅ **DO**: Use try-with-resources
```java
try (TrackerScope scope = context.createScope("Work")) {
    StatusTracker<Task> task = scope.trackTask(new Task());
    task.close();
} // ✓ scope automatically closed
```