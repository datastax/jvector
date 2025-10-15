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

# JVector Status Tracking User Guide

**📂 Complete, runnable examples:** [package-info.java](package-info.java)

All code examples in this guide are available as complete, compilable classes in this package (`io.github.jbellis.jvector.status.userguide`). You can run them directly or copy them into your project.

---

## START HERE

This guide builds from the simplest possible usage to advanced enterprise patterns. Each section adds one new feature to the previous example.

### Level 1: Absolute Minimum - Track ONE Task

**Just want to track a single task? One line of code.**

```java
import io.github.jbellis.jvector.status.*;
import io.github.jbellis.jvector.status.eventing.*;

// Your task class (see fauxtasks/DataLoader.java for full implementation)
public class DataLoader implements StatusSource<DataLoader> {
    ...
}

// Track it - simplest possible
DataLoader loader1 = new DataLoader();
try (StatusTracker<DataLoader> tracker1 = new StatusTracker<>(loader1)) {
    loader1.load(); // Task is independent - no tight coupling to tracker

    // Access status programmatically (no console output without a sink)
    System.out.println("Tracker status: " + tracker1); // Uses toString()
    // Output example: "DataLoader [100.0%] SUCCESS (1234ms)"
} // Auto-closes everything (tracker, scope, context, monitor thread)
```

**That's it!** Everything is auto-created and auto-closed via try-with-resources.

**Note:** Without adding a sink, you won't see real-time tracking output, but you can still access the tracker programmatically to get metrics like elapsed time, progress, and final status.

---

### Level 2: Add Console Output

**Building on Level 1: Add a sink to see progress in the console.**

```java
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;

// Same DataLoader class from Level 1
DataLoader loader2 = new DataLoader();
try (StatusTracker<DataLoader> tracker2 = new StatusTracker<>(loader2)) {
    // NEW: Add a sink to display progress
    tracker2.getContext().addSink(new ConsoleLoggerSink());

    loader2.load(); // Task executes independently
}
```

**Console output:**
```
[14:32:15.123] ▶ Started: DataLoader
[14:32:15.245]   DataLoader [████████░░░░░░░░░░░░]  40.0% - RUNNING
[14:32:15.611]   DataLoader [████████████████████] 100.0% - SUCCESS
[14:32:15.733] ✓ Finished: DataLoader
```

**What changed:** Added one line to configure output visibility.

---

### Level 3: Track Multiple Tasks

**Building on Level 2: Share a StatusContext across multiple tasks.**

```java
// Same DataLoader from Level 1, plus:
// (see fauxtasks/DataValidator.java for full implementation)
public class DataValidator implements StatusSource<DataValidator> {
    ...
}

// NEW: Shared context for multiple tasks
StatusContext ctx3 = new StatusContext("batch-processing");
ctx3.addSink(new ConsoleLoggerSink());

DataLoader loader3 = new DataLoader();
DataValidator validator3 = new DataValidator();

try (StatusTracker<DataLoader> loaderTracker = ctx3.track(loader3);
     StatusTracker<DataValidator> validatorTracker = ctx3.track(validator3)) {

    // Execute tasks independently
    loader3.load();
    validator3.validate();
} finally {
    ctx3.close(); // Close context when done (shuts down monitor thread)
}
```

**Why use StatusContext for multiple tasks?**
- Share sinks across all tasks
- Single monitoring thread (better performance)
- Configure poll interval once
- Coordinated lifecycle

---

### Level 4: Organize with Scopes

**Building on Level 3: Group related tasks using explicit scopes.**

```java
// NEW: Add a processor task (see fauxtasks/DataProcessor.java)
public class DataProcessor implements StatusSource<DataProcessor> {
    ...
}

// NEW: Use explicit scopes to organize tasks
StatusContext ctx4 = new StatusContext("data-pipeline");
ctx4.addSink(new ConsoleLoggerSink());

try (StatusScope ingestionScope = ctx4.createScope("Ingestion");
     StatusScope processingScope = ctx4.createScope("Processing")) {

    // Group ingestion tasks together
    DataLoader loader4 = new DataLoader();
    DataValidator validator4 = new DataValidator();

    try (StatusTracker<DataLoader> loaderTracker4 = ingestionScope.trackTask(loader4);
         StatusTracker<DataValidator> validatorTracker4 = ingestionScope.trackTask(validator4)) {
        loader4.load();       // Tasks execute independently
        validator4.validate();
    }

    // NEW: Wait for ingestion to complete before processing
    while (!ingestionScope.isComplete()) {
        Thread.sleep(10);
    }

    // Group processing tasks together
    DataProcessor processor4 = new DataProcessor();
    try (StatusTracker<DataProcessor> processorTracker4 = processingScope.trackTask(processor4)) {
        processor4.process();
    }
} finally {
    ctx4.close();
}
```

**Why use explicit scopes?**
- Logical grouping of related tasks
- Check completion of task groups with `scope.isComplete()`
- Better visualization in hierarchical sinks
- Control execution flow between phases

---

### Level 5: Add Custom Sinks

**Building on Level 4: Add metrics collection alongside console output.**

```java
import io.github.jbellis.jvector.status.sinks.*;

// Use tasks from previous levels
StatusContext ctx5 = new StatusContext("data-pipeline");

// NEW: Add multiple sinks for different purposes
ctx5.addSink(new ConsoleLoggerSink());
MetricsStatusSink metrics5 = new MetricsStatusSink();
ctx5.addSink(metrics5);

DataLoader loader5 = new DataLoader();

try (StatusScope scope5 = ctx5.createScope("Work")) {
    try (StatusTracker<DataLoader> tracker5 = scope5.trackTask(loader5)) {
        loader5.load(); // Task executes independently
    }
}

// NEW: Access metrics after tasks complete
System.out.println("Total tasks: " + metrics5.getTotalTasksStarted());
System.out.println("Avg duration: " + metrics5.getAverageTaskDuration() + "ms");

ctx5.close();
```

**Available sinks:**
- `ConsoleLoggerSink` - Text output with progress bars
- `ConsolePanelSink` - Rich terminal UI with hierarchy
- `LoggerStatusSink` - Integration with Log4j
- `MetricsStatusSink` - Performance metrics collection
- Custom - Implement `StatusSink` interface

---

### Level 6: Configure Poll Interval

**Building on Level 5: Adjust how frequently tasks are polled.**

```java
import java.time.Duration;

// NEW: Custom poll interval for responsive updates
StatusContext ctx6 = new StatusContext(
    "data-pipeline",
    Duration.ofMillis(50)); // Poll every 50ms instead of default 100ms

ctx6.addSink(new ConsoleLoggerSink());

DataLoader loader6 = new DataLoader();
try (StatusTracker<DataLoader> tracker6 = ctx6.track(loader6)) {
    loader6.load(); // Task executes independently
} finally {
    ctx6.close();
}
```

**Poll interval trade-offs:**
- **Faster (e.g., 50ms):** More responsive UI, higher CPU usage
- **Slower (e.g., 500ms):** Lower CPU usage, less responsive
- **Default: 100ms** - Good balance for most cases

---

### Level 7: Parallel Execution

**Building on Level 6: Track tasks running in parallel threads.**

```java
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

// NEW: Use AtomicLong for thread-safe parallel progress
// (see fauxtasks/ParallelDataLoader.java for full implementation)
public class ParallelDataLoader implements StatusSource<ParallelDataLoader> {
    ...
}

StatusContext ctx7 = new StatusContext("parallel-work");
ctx7.addSink(new ConsoleLoggerSink());

ExecutorService executor7 = Executors.newFixedThreadPool(3);
try (StatusScope scope7 = ctx7.createScope("Parallel")) {

    // NEW: Submit tasks to run in parallel
    List<Future<?>> futures7 = new ArrayList<>();

    for (int i = 0; i < 3; i++) {
        ParallelDataLoader loader7 = new ParallelDataLoader();
        Future<?> future = executor7.submit(() -> {
            try (StatusTracker<ParallelDataLoader> tracker7 = scope7.trackTask(loader7)) {
                loader7.load(); // Task executes independently
            } // Tracker auto-closes
        });
        futures7.add(future);
    }

    // Wait for all tasks
    for (Future<?> f : futures7) {
        f.get();
    }
} finally {
    executor7.shutdown();
    ctx7.close();
}
```

---

### Level 8: Nested Scopes (Full Enterprise Pattern)

**Building on Level 7: Deep hierarchical organization for complex pipelines.**

This is the complete example from the beginning of the guide, now showing how it builds on all previous levels.

---

### All-In: Complex Hierarchical Tracking

**Scenario:** You have a multi-stage data pipeline with multiple parallel tasks, nested workflows, and need custom visualization.

**What you need:**
1. Multiple task classes implementing `StatusSource<T>`
2. A `StatusContext` to coordinate everything
3. `StatusScope` objects to organize tasks hierarchically
4. Multiple sinks for different outputs (console panel, metrics, logs)
5. Custom configuration for polling intervals

**Complete Example:**

```java
import io.github.jbellis.jvector.status.*;
import io.github.jbellis.jvector.status.eventing.*;
import io.github.jbellis.jvector.status.sinks.*;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.*;

// Step 1: Define all your task types
public class ExtractTask implements StatusSource<ExtractTask> {
    private volatile double progress = 0.0;
    private volatile RunState state = RunState.PENDING;
    private final String source;

    public ExtractTask(String source) {
        this.source = source;
    }

    @Override
    public StatusUpdate<ExtractTask> getTaskStatus() {
        return new StatusUpdate<>(progress, state, this);
    }

    public String getName() {
        return "Extract from " + source;
    }

    public void execute() {
        state = RunState.RUNNING;
        try {
            // Extraction logic with progress updates
            for (int i = 0; i < 100; i++) {
                extractChunk(i);
                progress = (double) (i + 1) / 100;
                Thread.sleep(10); // Simulate work
            }
            state = RunState.SUCCESS;
        } catch (Exception e) {
            state = RunState.FAILED;
            throw new RuntimeException(e);
        }
    }

    private void extractChunk(int i) {
        // Extraction logic
    }
}

public class TransformTask implements StatusSource<TransformTask> {
    private volatile double progress = 0.0;
    private volatile RunState state = RunState.PENDING;
    private final String transformation;

    public TransformTask(String transformation) {
        this.transformation = transformation;
    }

    @Override
    public StatusUpdate<TransformTask> getTaskStatus() {
        return new StatusUpdate<>(progress, state, this);
    }

    public String getName() {
        return "Transform: " + transformation;
    }

    public void execute() {
        state = RunState.RUNNING;
        try {
            for (int i = 0; i < 100; i++) {
                transformRecord(i);
                progress = (double) (i + 1) / 100;
                Thread.sleep(5);
            }
            state = RunState.SUCCESS;
        } catch (Exception e) {
            state = RunState.FAILED;
            throw new RuntimeException(e);
        }
    }

    private void transformRecord(int i) {
        // Transformation logic
    }
}

public class LoadTask implements StatusSource<LoadTask> {
    private volatile double progress = 0.0;
    private volatile RunState state = RunState.PENDING;
    private final String destination;

    public LoadTask(String destination) {
        this.destination = destination;
    }

    @Override
    public StatusUpdate<LoadTask> getTaskStatus() {
        return new StatusUpdate<>(progress, state, this);
    }

    public String getName() {
        return "Load to " + destination;
    }

    public void execute() {
        state = RunState.RUNNING;
        try {
            for (int i = 0; i < 100; i++) {
                loadRecord(i);
                progress = (double) (i + 1) / 100;
                Thread.sleep(8);
            }
            state = RunState.SUCCESS;
        } catch (Exception e) {
            state = RunState.FAILED;
            throw new RuntimeException(e);
        }
    }

    private void loadRecord(int i) {
        // Loading logic
    }
}

// Step 2: Optionally create a custom sink for specialized monitoring
public class MetricsCollector implements StatusSink {
    private final ConcurrentHashMap<String, Double> taskProgress = new ConcurrentHashMap<>();

    @Override
    public void taskStarted(StatusTracker<?> task) {
        String name = StatusTracker.extractTaskName(task);
        taskProgress.put(name, 0.0);
        System.out.println("METRICS: Task started - " + name);
    }

    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
        String name = StatusTracker.extractTaskName(task);
        taskProgress.put(name, status.progress);

        if (status.runstate == RunState.RUNNING && status.progress > 0.5) {
            System.out.println("METRICS: " + name + " crossed 50% threshold");
        }
    }

    @Override
    public void taskFinished(StatusTracker<?> task) {
        String name = StatusTracker.extractTaskName(task);
        Double finalProgress = taskProgress.remove(name);
        long runTime = task.getElapsedRunningTime();
        System.out.println("METRICS: Task completed - " + name +
                          " (progress: " + finalProgress + ", time: " + runTime + "ms)");
    }

    public Map<String, Double> getCurrentProgress() {
        return new HashMap<>(taskProgress);
    }
}

// Step 3: Build the complete pipeline with hierarchical tracking
public class DataPipeline {
    public void execute() throws Exception {
        // Create multiple sinks for different purposes
        ConsolePanelSink consolePanel = ConsolePanelSink.builder()
            .outputMode(OutputMode.FULL)
            .build();
        LoggerStatusSink logger = new LoggerStatusSink();
        MetricsCollector metrics = new MetricsCollector();

        // Create context with custom poll interval and multiple sinks
        try (StatusContext context = new StatusContext(
                "etl-pipeline",
                Duration.ofMillis(50), // Poll every 50ms for responsive UI
                List.of(consolePanel, logger, metrics))) {

            // Create organizational hierarchy with scopes
            try (StatusScope etlScope = context.createScope("ETL-Pipeline")) {

                // First-level scopes for major phases
                StatusScope extractScope = etlScope.createChildScope("Extract");
                StatusScope transformScope = etlScope.createChildScope("Transform");
                StatusScope loadScope = etlScope.createChildScope("Load");

                // Second-level scopes for parallel operations
                StatusScope parallelTransforms = transformScope.createChildScope("Parallel-Transforms");

                // Create executor for parallel tasks
                ExecutorService executor = Executors.newFixedThreadPool(3);

                try {
                    // Phase 1: Extract (sequential)
                    ExtractTask extract1 = new ExtractTask("database-1");
                    ExtractTask extract2 = new ExtractTask("database-2");

                    // Execute extractions sequentially with try-with-resources
                    try (StatusTracker<ExtractTask> tracker1 = extractScope.trackTask(extract1)) {
                        extract1.execute();
                    } // Tracker auto-closes

                    try (StatusTracker<ExtractTask> tracker2 = extractScope.trackTask(extract2)) {
                        extract2.execute();
                    } // Tracker auto-closes

                    // Wait for extract scope to complete
                    while (!extractScope.isComplete()) {
                        Thread.sleep(10);
                    }

                    // Phase 2: Transform (parallel)
                    TransformTask transform1 = new TransformTask("normalize");
                    TransformTask transform2 = new TransformTask("enrich");
                    TransformTask transform3 = new TransformTask("validate");

                    // Execute transformations in parallel with try-with-resources
                    Future<?> f1 = executor.submit(() -> {
                        try (StatusTracker<TransformTask> t1 = parallelTransforms.trackTask(transform1)) {
                            transform1.execute();
                        } // Tracker auto-closes
                    });
                    Future<?> f2 = executor.submit(() -> {
                        try (StatusTracker<TransformTask> t2 = parallelTransforms.trackTask(transform2)) {
                            transform2.execute();
                        } // Tracker auto-closes
                    });
                    Future<?> f3 = executor.submit(() -> {
                        try (StatusTracker<TransformTask> t3 = parallelTransforms.trackTask(transform3)) {
                            transform3.execute();
                        } // Tracker auto-closes
                    });

                    // Wait for all transforms to complete
                    f1.get();
                    f2.get();
                    f3.get();

                    // Verify transform scope completion
                    while (!transformScope.isComplete()) {
                        Thread.sleep(10);
                    }

                    // Phase 3: Load (sequential)
                    LoadTask load = new LoadTask("data-warehouse");
                    try (StatusTracker<LoadTask> loadTracker = loadScope.trackTask(load)) {
                        load.execute();
                    } // Tracker auto-closes

                    // Wait for entire ETL pipeline to complete
                    while (!etlScope.isComplete()) {
                        Thread.sleep(10);
                    }

                    System.out.println("\nPipeline completed!");
                    System.out.println("Final metrics: " + metrics.getCurrentProgress());

                } finally {
                    executor.shutdown();
                    executor.awaitTermination(10, TimeUnit.SECONDS);
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        new DataPipeline().execute();
    }
}
```

**What This Example Demonstrates:**

1. **Multiple Sinks:** Console panel for visual display, logger for persistent logs, custom metrics collector
2. **Custom Polling:** 50ms intervals for responsive updates
3. **Hierarchical Organization:**
   ```
   ETL-Pipeline (scope)
     ├─ Extract (scope)
     │    ├─ Extract from database-1 (tracker)
     │    └─ Extract from database-2 (tracker)
     ├─ Transform (scope)
     │    └─ Parallel-Transforms (scope)
     │         ├─ Transform: normalize (tracker)
     │         ├─ Transform: enrich (tracker)
     │         └─ Transform: validate (tracker)
     └─ Load (scope)
          └─ Load to data-warehouse (tracker)
   ```
4. **Parallel Execution:** Multiple tasks running concurrently with independent progress tracking
5. **Completion Checking:** Using `scope.isComplete()` to wait for phases
6. **Custom Sink:** `MetricsCollector` demonstrates custom monitoring logic
7. **Resource Management:** Try-with-resources ensures proper cleanup

**Key Points:**
- Use `StatusScope` to organize related tasks hierarchically
- Scopes can be nested to any depth
- Multiple sinks can run simultaneously
- Custom poll intervals balance responsiveness vs. CPU usage
- Parallel execution works seamlessly with status tracking
- Use `scope.isComplete()` to coordinate between pipeline stages
- Extract task names using `StatusTracker.extractTaskName()` for sinks

---

## Core Concepts

### The Three-Level Hierarchy

The Status API enforces a clear three-level architecture:

```
StatusContext (Session Coordinator)
    │
    ├─ StatusScope (Organizational Container - no progress)
    │    ├─ StatusTracker (Actual Work - has progress)
    │    ├─ StatusTracker (Actual Work - has progress)
    │    └─ StatusScope (Nested Organization)
    │         └─ StatusTracker (Actual Work - has progress)
    │
    └─ StatusScope (Another Organizational Container)
         └─ StatusTracker (Actual Work - has progress)
```

**Key Rules:**
1. **StatusContext** owns the monitoring infrastructure (one per operation)
2. **StatusScope** provides organization but has NO progress or state
3. **StatusTracker** represents actual work and is always a leaf node (cannot have children)

### Status Flow Architecture

Status information flows unidirectionally:

```
Task (your object with progress/state)
  ↓
StatusTracker (observes task via statusFunction)
  ↓
StatusMonitor (polls trackers periodically)
  ↓
StatusContext (routes updates)
  ↓
StatusSinks (console, logs, metrics, custom)
```

**Important:** Status flows one way only. Tasks don't know about trackers, and trackers don't push updates back to tasks.

---

## Representing Task Status

Before implementing status tracking, you need to understand how to represent progress and state in your tasks. The monitoring framework polls your task periodically, so your task must maintain thread-safe status fields.

### Status Semantics

Every task status consists of two components:

1. **Progress** (double, 0.0 to 1.0)
   - `0.0` = Not started or 0% complete
   - `0.5` = 50% complete
   - `1.0` = 100% complete
   - Should be monotonically increasing during normal execution
   - Represents fractional completion of work

2. **RunState** (enum)
   - `PENDING` - Task is queued but not yet started
   - `RUNNING` - Task is actively executing
   - `SUCCESS` - Task completed successfully
   - `FAILED` - Task completed with errors
   - `CANCELLED` - Task was cancelled before completion

### **CRITICAL: Efficient Progress Tracking in Hot Code**

When tracking progress over hot code paths or fine-grained iterations, **do the minimum necessary** to update an observable field. The golden rule:

**❌ DON'T calculate fractional progress on every iteration**
**✅ DO increment a counter, calculate fraction in getTaskStatus()**

This is critical for performance - progress calculations involve division (expensive) and should happen only when polled (every ~100ms), not on every loop iteration (potentially millions of times per second).

**Anti-Pattern (SLOW - avoid this):**
```java
public class SlowTask implements StatusSource<SlowTask> {
    private volatile double progress = 0.0;  // Fractional value
    private final long total = 1_000_000;

    public void process() {
        for (long i = 0; i < total; i++) {
            // ... hot code ...
            progress = (double) (i + 1) / total;  // ❌ SLOW: Division on EVERY iteration!
        }
    }

    @Override
    public StatusUpdate<SlowTask> getTaskStatus() {
        return new StatusUpdate<>(progress, state, this);  // Just return pre-calculated value
    }
}
```

**Efficient Pattern (FAST - do this):**
```java
public class FastTask implements StatusSource<FastTask> {
    private volatile long itemsProcessed = 0;  // Integer counter
    private final long total = 1_000_000;

    public void process() {
        for (long i = 0; i < total; i++) {
            // ... hot code ...
            itemsProcessed++;  // ✅ FAST: Just increment (single instruction)
        }
    }

    @Override
    public StatusUpdate<FastTask> getTaskStatus() {
        // ✅ Calculate fraction HERE (only when polled ~every 100ms)
        double progress = (double) itemsProcessed / total;
        return new StatusUpdate<>(progress, state, this);
    }
}
```

**Performance Impact:**
- **Anti-pattern:** 1 million divisions per second in hot loop = significant overhead
- **Efficient pattern:** 1 million increments (fast) + 10 divisions per second (polling) = negligible overhead
- **Speedup:** Can be 10-100x faster for fine-grained loops

**All examples in this guide follow the efficient pattern** - counters in loops, calculations in getTaskStatus().

---

### Thread Safety Requirements

Since the monitor thread reads status while your task thread updates it, you **must** use thread-safe mechanisms:

#### Pattern 1: Volatile Fields (Simplest)

Best for simple counters and single-threaded task execution:

```java
public class SimpleTask implements StatusSource<SimpleTask> {
    private volatile double progress = 0.0;
    private volatile RunState state = RunState.PENDING;
    private final int totalItems;

    @Override
    public StatusUpdate<SimpleTask> getTaskStatus() {
        return new StatusUpdate<>(progress, state, this);
    }

    public void execute() {
        state = RunState.RUNNING;
        for (int i = 0; i < totalItems; i++) {
            processItem(i);
            progress = (double) (i + 1) / totalItems; // Safe: single writer thread
        }
        state = RunState.SUCCESS;
    }
}
```

**When to use:**
- Single thread updates progress
- Simple sequential processing
- Frequent progress updates are acceptable
- No complex state transitions

**Pros:**
- Simplest approach
- No overhead from atomics
- Easy to understand and maintain

**Cons:**
- Only safe with single writer thread
- No atomicity guarantees for compound operations

---

#### Pattern 2: AtomicLong with CAS (Compare-And-Swap)

Best for concurrent updates from multiple threads:

```java
public class ParallelTask implements StatusSource<ParallelTask> {
    private final AtomicLong completed = new AtomicLong(0);
    private final long totalItems;
    private volatile RunState state = RunState.PENDING;

    public ParallelTask(long totalItems) {
        this.totalItems = totalItems;
    }

    @Override
    public StatusUpdate<ParallelTask> getTaskStatus() {
        double progress = (double) completed.get() / totalItems;
        return new StatusUpdate<>(progress, state, this);
    }

    public void execute() {
        state = RunState.RUNNING;

        // Multiple threads can safely update completed
        ExecutorService executor = Executors.newFixedThreadPool(4);
        IntStream.range(0, (int) totalItems)
            .parallel()
            .forEach(i -> {
                processItem(i);
                completed.incrementAndGet(); // Atomic operation
            });

        state = RunState.SUCCESS;
    }
}
```

**When to use:**
- Multiple threads update progress concurrently
- Parallel processing with work stealing
- Need exact counting without race conditions
- Counter-based progress tracking

**Pros:**
- Thread-safe for multiple writers
- Lock-free performance
- Guarantees atomic updates

**Cons:**
- Slightly more complex
- Small overhead from atomic operations
- Must convert to double for progress calculation

---

#### Pattern 3: Synchronized Progress Updates

Best for complex state that requires consistency:

```java
public class ComplexTask implements StatusSource<ComplexTask> {
    private final Object lock = new Object();
    private double progress = 0.0;
    private RunState state = RunState.PENDING;
    private String currentPhase = "initializing";

    @Override
    public StatusUpdate<ComplexTask> getTaskStatus() {
        synchronized (lock) {
            return new StatusUpdate<>(progress, state, this);
        }
    }

    public void execute() {
        updateState(RunState.RUNNING, 0.0, "loading");

        loadData();
        updateState(RunState.RUNNING, 0.33, "processing");

        processData();
        updateState(RunState.RUNNING, 0.66, "saving");

        saveResults();
        updateState(RunState.SUCCESS, 1.0, "complete");
    }

    private void updateState(RunState newState, double newProgress, String phase) {
        synchronized (lock) {
            this.state = newState;
            this.progress = newProgress;
            this.currentPhase = phase;
        }
    }
}
```

**When to use:**
- Multiple related fields must be updated together
- Need consistent snapshots of complex state
- State transitions require coordination
- Additional metadata beyond progress/state

**Pros:**
- Guarantees consistency across multiple fields
- Can protect complex state transitions
- Easy to add more tracked fields

**Cons:**
- Synchronization overhead
- Potential for contention
- Must be careful about lock scope

---

#### Pattern 4: Derived Progress from Existing State

Best when tracking legacy code you can't modify:

```java
public class LegacyBatchJob {
    // Existing code you can't change
    private final AtomicInteger processedCount = new AtomicInteger(0);
    private final int totalCount;
    private volatile boolean running = false;
    private volatile boolean failed = false;

    public int getProcessedCount() { return processedCount.get(); }
    public int getTotalCount() { return totalCount; }
    public boolean isRunning() { return running; }
    public boolean hasFailed() { return failed; }
}

// Track it with a custom status function
Function<LegacyBatchJob, StatusUpdate<LegacyBatchJob>> statusFn = job -> {
    double progress = (double) job.getProcessedCount() / job.getTotalCount();

    RunState state;
    if (job.hasFailed()) {
        state = RunState.FAILED;
    } else if (job.getProcessedCount() >= job.getTotalCount()) {
        state = RunState.SUCCESS;
    } else if (job.isRunning()) {
        state = RunState.RUNNING;
    } else {
        state = RunState.PENDING;
    }

    return new StatusUpdate<>(progress, state, job);
};

try (StatusTracker<LegacyBatchJob> tracker = context.track(job, statusFn)) {
    job.execute();
}
```

**When to use:**
- Can't modify the task class
- Working with third-party code
- Task already has suitable state tracking
- Need to adapt existing APIs

**Pros:**
- No changes to existing code required
- Can track any object
- Flexible status derivation logic

**Cons:**
- Status function called on every poll
- May need to synchronize if reading multiple fields
- More complex setup code

---

### Progress Calculation Patterns

**Simple Counter:**
```java
progress = (double) itemsProcessed / totalItems;
```

**Weighted Phases:**
```java
// Phase 1: Loading (0% - 30%)
progress = 0.0 + (0.30 * loadProgress);

// Phase 2: Processing (30% - 80%)
progress = 0.30 + (0.50 * processProgress);

// Phase 3: Saving (80% - 100%)
progress = 0.80 + (0.20 * saveProgress);
```

**Bytes Processed:**
```java
progress = (double) bytesProcessed / totalBytes;
```

**Time-Based Estimation:**
```java
long elapsed = System.currentTimeMillis() - startTime;
long estimated = (long) (elapsed / progress); // Total estimated time
progress = Math.min(1.0, (double) elapsed / estimatedTotal);
```

---

## Task Instrumentation: Decorator vs. Functor

There are two fundamental approaches to instrumenting tasks for status tracking. Understanding when to use each is crucial for clean, maintainable code.

### Approach 1: Decorator Pattern (Implement StatusSource)

**What it is:** Modify the task class to implement the `StatusSource<T>` interface, making it self-reporting.

**When to use:**
- ✅ You own the task code
- ✅ Writing new code
- ✅ Task is designed for monitoring
- ✅ Want clean, self-contained tasks
- ✅ Status tracking is a core feature

**Example:**

```java
public class DataProcessor implements StatusSource<DataProcessor> {
    private volatile double progress = 0.0;
    private volatile RunState state = RunState.PENDING;
    private final List<Data> dataset;

    public DataProcessor(List<Data> dataset) {
        this.dataset = dataset;
    }

    // Decorator: Task reports its own status
    @Override
    public StatusUpdate<DataProcessor> getTaskStatus() {
        return new StatusUpdate<>(progress, state, this);
    }

    public void process() {
        state = RunState.RUNNING;
        for (int i = 0; i < dataset.size(); i++) {
            processData(dataset.get(i));
            progress = (double) (i + 1) / dataset.size();
        }
        state = RunState.SUCCESS;
    }
}

// Usage is simple and clean
try (StatusTracker<DataProcessor> tracker = new StatusTracker<>(processor)) {
    tracker.getContext().addSink(new ConsoleLoggerSink());
    tracker.getTracked().process();
}
```

**Advantages:**
- **Type Safety:** Compiler ensures status method matches the task type
- **Encapsulation:** Status logic is part of the task
- **Discoverability:** Interface makes tracking capability obvious
- **Simplicity:** No separate status function needed
- **IDE Support:** Auto-completion works naturally

**Disadvantages:**
- **Intrusive:** Requires modifying the task class
- **Coupling:** Task becomes aware of status tracking framework
- **Not Always Possible:** Can't modify third-party or legacy code

**Best for:**
- New application code
- Tasks designed with monitoring in mind
- When you control the codebase
- Clean architecture where tasks self-report

---

### Approach 2: Functor Pattern (External Status Function)

**What it is:** Keep the task class unchanged and provide a separate function that knows how to extract status from the task.

**When to use:**
- ✅ Can't modify the task class (third-party, legacy)
- ✅ Task already has status information in different form
- ✅ Want to keep tracking concerns separate
- ✅ Multiple different status interpretations of same task
- ✅ Adapting existing APIs

**Example:**

```java
// Existing task you can't modify
public class LegacyImporter {
    private final AtomicInteger imported = new AtomicInteger(0);
    private final int total;
    private volatile boolean running = false;
    private volatile Throwable error = null;

    public int getImported() { return imported.get(); }
    public int getTotal() { return total; }
    public boolean isRunning() { return running; }
    public Throwable getError() { return error; }

    public void importData() {
        // Existing implementation
    }
}

// Functor: External function observes the task
Function<LegacyImporter, StatusUpdate<LegacyImporter>> statusFunction = task -> {
    // Derive progress from existing state
    double progress = (double) task.getImported() / task.getTotal();

    // Derive runstate from existing state
    RunState state;
    if (task.getError() != null) {
        state = RunState.FAILED;
    } else if (task.getImported() >= task.getTotal()) {
        state = RunState.SUCCESS;
    } else if (task.isRunning()) {
        state = RunState.RUNNING;
    } else {
        state = RunState.PENDING;
    }

    return new StatusUpdate<>(progress, state, task);
};

// Usage requires passing the status function
try (StatusContext context = new StatusContext("import");
     StatusTracker<LegacyImporter> tracker = context.track(importer, statusFunction)) {
    tracker.getTracked().importData();
}
```

**Advantages:**
- **Non-Intrusive:** No changes to task code required
- **Separation of Concerns:** Tracking logic separate from business logic
- **Flexibility:** Can have different status interpretations
- **Adaptability:** Works with any existing code
- **Reusability:** Same function can track multiple instances

**Disadvantages:**
- **More Verbose:** Need to write and pass status function
- **Less Type Safe:** Function signature not enforced by task
- **Duplication:** May duplicate status logic across similar tasks
- **Hidden Contract:** Not obvious that task can be tracked

**Best for:**
- Legacy or third-party code
- When you can't modify the task class
- Adapting existing frameworks
- When tracking is optional/external concern

---

### Choosing Between Decorator and Functor

**Use Decorator (StatusSource) when:**
```java
// ✅ You own the code
public class MyTask implements StatusSource<MyTask> {
    // Task knows how to report its own status
}

try (StatusTracker<MyTask> tracker = new StatusTracker<>(new MyTask())) {
    // Clean and simple
}
```

**Use Functor (Status Function) when:**
```java
// ✅ You can't modify the task
public class ThirdPartyTask {
    // Existing code, can't change
}

Function<ThirdPartyTask, StatusUpdate<ThirdPartyTask>> adaptStatus = task -> {
    // Adapt existing state to status tracking
    return new StatusUpdate<>(calculateProgress(task), deriveState(task), task);
};

try (StatusTracker<ThirdPartyTask> tracker = context.track(task, adaptStatus)) {
    // Tracking external code
}
```

**Hybrid Approach - Best of Both:**

For complex scenarios, use decorator for your code and functors for external code:

```java
// Your code uses decorator
public class MyProcessor implements StatusSource<MyProcessor> {
    @Override
    public StatusUpdate<MyProcessor> getTaskStatus() {
        return new StatusUpdate<>(progress, state, this);
    }
}

// External library uses functor
Function<ExternalTask, StatusUpdate<ExternalTask>> externalAdapter = task -> {
    return new StatusUpdate<>(task.getProgress(), mapState(task), task);
};

try (StatusContext context = new StatusContext("hybrid")) {
    context.addSink(new ConsoleLoggerSink());

    // Your code - clean decorator
    try (StatusTracker<MyProcessor> t1 = context.track(new MyProcessor())) {
        t1.getTracked().execute();
    }

    // External code - adapted with functor
    try (StatusTracker<ExternalTask> t2 = context.track(externalTask, externalAdapter)) {
        t2.getTracked().run();
    }
}
```

---

### Common Mistakes and Solutions

**Mistake 1: Non-Volatile Fields**
```java
// ❌ Wrong - monitor thread may see stale values
private double progress;
private RunState state;

// ✅ Right - monitor thread sees updates immediately
private volatile double progress;
private volatile RunState state;
```

**Mistake 2: Multiple Writers Without Atomics**
```java
// ❌ Wrong - race condition with parallel updates
private volatile int completed;
// Thread 1: completed++;
// Thread 2: completed++;
// Lost updates!

// ✅ Right - atomic updates
private final AtomicInteger completed = new AtomicInteger(0);
// Thread 1: completed.incrementAndGet();
// Thread 2: completed.incrementAndGet();
// All updates counted correctly
```

**Mistake 3: Reading Multiple Fields Without Synchronization**
```java
// ❌ Wrong - inconsistent snapshot
public StatusUpdate<Task> getTaskStatus() {
    double prog = progress;      // Read 1
    RunState st = state;         // Read 2 - may be inconsistent with Read 1
    String phase = currentPhase; // Read 3 - may be inconsistent
    return new StatusUpdate<>(prog, st, this);
}

// ✅ Right - consistent snapshot
private final Object lock = new Object();

public StatusUpdate<Task> getTaskStatus() {
    synchronized (lock) {
        return new StatusUpdate<>(progress, state, this);
    }
}
```

**Mistake 4: Progress Goes Backwards**
```java
// ❌ Wrong - confusing to users
progress = 0.5;
// ... some work ...
progress = 0.3; // Went backwards!

// ✅ Right - monotonically increasing
progress = Math.max(progress, newProgress);
```

**Mistake 5: Forgetting to Update State**
```java
// ❌ Wrong - state stays PENDING forever
public void execute() {
    for (int i = 0; i < total; i++) {
        doWork(i);
        progress = (double) i / total;
    }
}

// ✅ Right - state transitions properly
public void execute() {
    state = RunState.RUNNING; // Start
    try {
        for (int i = 0; i < total; i++) {
            doWork(i);
            progress = (double) (i + 1) / total;
        }
        state = RunState.SUCCESS; // Success
    } catch (Exception e) {
        state = RunState.FAILED; // Failure
        throw e;
    }
}
```

---

## Implementing Status in Your Tasks

### Option 1: Implement StatusSource (Recommended)

This is the cleanest approach for new code:

```java
public class MyTask implements StatusSource<MyTask> {
    private volatile double progress = 0.0;
    private volatile RunState state = RunState.PENDING;

    @Override
    public StatusUpdate<MyTask> getTaskStatus() {
        return new StatusUpdate<>(progress, state, this);
    }

    public void execute() {
        state = RunState.RUNNING;
        // Update progress as work proceeds
        progress = 0.5;
        // ...
        state = RunState.SUCCESS;
    }
}

// Track it (simplest - scope auto-created)
try (StatusContext context = new StatusContext("operation");
     StatusTracker<MyTask> tracker = context.track(new MyTask())) {
    tracker.getTracked().execute();
}

// Or with explicit scope (for organization)
try (StatusContext context = new StatusContext("operation");
     StatusScope scope = context.createScope("Work");
     StatusTracker<MyTask> tracker = scope.trackTask(new MyTask())) { // Use try-with-resources
    tracker.getTracked().execute();
} // Tracker auto-closes
```

### Option 2: Custom Status Function

Use this when you can't modify the task class:

```java
public class LegacyTask {
    private int completed = 0;
    private int total = 100;

    public int getCompleted() { return completed; }
    public int getTotal() { return total; }
    public boolean isDone() { return completed >= total; }
}

// Create a status function
Function<LegacyTask, StatusUpdate<LegacyTask>> statusFn = task -> {
    double progress = (double) task.getCompleted() / task.getTotal();
    RunState state = task.isDone() ? RunState.SUCCESS : RunState.RUNNING;
    return new StatusUpdate<>(progress, state, task);
};

// Track it (simplest - scope auto-created)
try (StatusContext context = new StatusContext("legacy-operation");
     StatusTracker<LegacyTask> tracker = context.track(new LegacyTask(), statusFn)) {
    // Execute task
}

// Or with explicit scope
try (StatusContext context = new StatusContext("legacy-operation");
     StatusScope scope = context.createScope("Work");
     StatusTracker<LegacyTask> tracker = scope.trackTask(new LegacyTask(), statusFn)) { // Use try-with-resources
    // Execute task
} // Tracker auto-closes
```

---

## Available Sinks

### ConsoleLoggerSink

Simple text-based logging to console with timestamps and progress bars.

**Usage:**
```java
context.addSink(new ConsoleLoggerSink());
```

**Example Output:**
```
[14:32:15.123] ▶ Started: data-processing
[14:32:15.245]   data-processing [████████░░░░░░░░░░░░]  40.0% - RUNNING
[14:32:15.367]   data-processing [████████████░░░░░░░░]  60.0% - RUNNING
[14:32:15.489]   data-processing [████████████████░░░░]  80.0% - RUNNING
[14:32:15.611]   data-processing [████████████████████] 100.0% - SUCCESS
[14:32:15.733] ✓ Finished: data-processing
```

**Customization:**
```java
// Without timestamps
new ConsoleLoggerSink(System.out, false, true);

// Without progress bars
new ConsoleLoggerSink(System.out, true, false);
// Output: [14:32:15.123]   data-processing [60.0%] - RUNNING

// Minimal (no timestamps, no bars)
new ConsoleLoggerSink(System.out, false, false);
// Output:   data-processing [60.0%] - RUNNING
```

---

### ConsolePanelSink

Rich visual panel with hierarchical task display, progress bars, and integrated console output.

**Usage:**
```java
ConsolePanelSink sink = ConsolePanelSink.builder()
    .outputMode(OutputMode.FULL)  // or COMPACT, MINIMAL
    .build();
context.addSink(sink);
```

**Example Output:**
```
╔═══ Task Status Monitor ═══════════════════════════════════════╗
║                                                               ║
║ ▶ [14:32:15] ETL-Pipeline [████████████░░░░░░░░]  60% (2.3s) ║
║   ├─ ✓ [14:32:16] Extract [████████████████████] 100% (1.2s) ║
║   │   ├─ ✓ Extract from database-1 (0.6s)                    ║
║   │   └─ ✓ Extract from database-2 (0.6s)                    ║
║   ├─ ▶ [14:32:17] Transform [██████░░░░░░░░░░░░]  30% (0.8s) ║
║   │   └─ Parallel-Transforms                                  ║
║   │       ├─ ▶ Transform: normalize [████░░░░░░░░] 20%       ║
║   │       ├─ ▶ Transform: enrich [██████░░░░░░░░] 30%        ║
║   │       └─ ▶ Transform: validate [████████░░░░] 40%        ║
║   └─ ○ [--:--:--] Load                                        ║
║                                                               ║
║ Active: 4 | Completed: 3 | Failed: 0                         ║
╠═══ Console Output ════════════════════════════════════════════╣
║ [INFO ] Starting ETL pipeline...                             ║
║ [INFO ] Extracting from database-1                           ║
║ [INFO ] Extracting from database-2                           ║
║ [INFO ] Beginning transformation phase                       ║
║ [DEBUG] Processing record batch 1 of 10                      ║
║ ▼ (↑/↓ to scroll, 12 more lines)                            ║
╚═══════════════════════════════════════════════════════════════╝
```

**Status Indicators:**
- `▶` Running task (cyan)
- `✓` Completed successfully (green)
- `✗` Failed (red)
- `○` Pending (white)

**Interactive Controls:**
- `↑/↓` Scroll console logs
- `[/]` Adjust split between task and log panels
- `Home` Reset view
- `s` Save snapshot to file
- `?` Show help

---

### LoggerStatusSink

Integration with Log4j 2 for production logging infrastructure.

**Usage:**
```java
context.addSink(new LoggerStatusSink());
```

**Example Output (in application logs):**
```
2025-01-15 14:32:15.123 [INFO ] io.myapp.TaskProcessor - Task started: data-processing
2025-01-15 14:32:15.245 [INFO ] io.myapp.TaskProcessor - Task update: data-processing [40.0%] - RUNNING
2025-01-15 14:32:15.367 [INFO ] io.myapp.TaskProcessor - Task update: data-processing [60.0%] - RUNNING
2025-01-15 14:32:15.489 [INFO ] io.myapp.TaskProcessor - Task update: data-processing [80.0%] - RUNNING
2025-01-15 14:32:15.611 [INFO ] io.myapp.TaskProcessor - Task finished: data-processing
```

**With Custom Logger and Level:**
```java
Logger customLogger = LogManager.getLogger("app.background.tasks");
context.addSink(new LoggerStatusSink(customLogger, Level.DEBUG));
```

Output:
```
2025-01-15 14:32:15.123 [DEBUG] app.background.tasks - Task started: background-job
2025-01-15 14:32:15.245 [DEBUG] app.background.tasks - Task update: background-job [50.0%] - RUNNING
2025-01-15 14:32:15.367 [DEBUG] app.background.tasks - Task finished: background-job
```

---

### MetricsStatusSink

Collects detailed performance metrics about task execution for monitoring and analysis.

**Usage:**
```java
MetricsStatusSink metrics = new MetricsStatusSink();
context.addSink(metrics);

// ... run tasks ...

// Access metrics programmatically
long started = metrics.getTotalTasksStarted();
long finished = metrics.getTotalTasksFinished();
double avgDuration = metrics.getAverageTaskDuration();
```

**Generating Reports:**
```java
String report = metrics.generateReport();
System.out.println(report);
```

**Example Report Output:**
```
=== Task Metrics Report ===
Total tasks started: 15
Total tasks finished: 12
Active tasks: 3
Total updates: 847
Average task duration: 2547.33 ms

Task Details:
  - Extract from database-1:
    Duration: 3200 ms
    Updates: 64
    Progress: 100.0%
    Status: Finished
  - Extract from database-2:
    Duration: 2890 ms
    Updates: 58
    Progress: 100.0%
    Status: Finished
  - Transform: normalize:
    Duration: 1650 ms
    Updates: 45
    Progress: 75.0%
    Status: Running
  - Transform: enrich:
    Duration: 1820 ms
    Updates: 52
    Progress: 80.0%
    Status: Running
```

**Accessing Individual Task Metrics:**
```java
MetricsStatusSink.TaskMetrics taskMetrics = metrics.getMetrics(tracker);
System.out.println("Duration: " + taskMetrics.getDuration() + "ms");
System.out.println("Updates: " + taskMetrics.getUpdateCount());
System.out.println("Avg Progress: " + taskMetrics.getAverageProgress());
System.out.println("Final Progress: " + taskMetrics.getLastProgress() * 100 + "%");
```

**Export to Monitoring Systems:**
```java
// Periodically export to Prometheus, CloudWatch, etc.
scheduler.scheduleAtFixedRate(() -> {
    exportMetrics(
        "tasks.started", metrics.getTotalTasksStarted(),
        "tasks.finished", metrics.getTotalTasksFinished(),
        "tasks.active", metrics.getActiveTaskCount(),
        "tasks.avg_duration_ms", metrics.getAverageTaskDuration()
    );
}, 0, 60, TimeUnit.SECONDS);
```

---

### Custom Sink

Implement the `StatusSink` interface for custom monitoring, alerting, or integration with external systems.

**Interface:**
```java
public class CustomSink implements StatusSink {
    @Override
    public void taskStarted(StatusTracker<?> task) {
        // Called when tracker is created
    }

    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
        // Called on each status change
    }

    @Override
    public void taskFinished(StatusTracker<?> task) {
        // Called when tracker is closed
    }
}
```

**Example: Alert on Slow Tasks**
```java
public class SlowTaskAlerter implements StatusSink {
    private final long thresholdMs;
    private final AlertService alertService;

    public SlowTaskAlerter(long thresholdMs, AlertService alertService) {
        this.thresholdMs = thresholdMs;
        this.alertService = alertService;
    }

    @Override
    public void taskStarted(StatusTracker<?> task) {
        // Track start time (already tracked by StatusTracker)
    }

    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
        long elapsed = task.getElapsedRunningTime();
        if (elapsed > thresholdMs && status.progress < 0.5) {
            String taskName = StatusTracker.extractTaskName(task);
            alertService.sendAlert(
                "Task taking too long",
                taskName + " has been running for " + elapsed + "ms but only " +
                (status.progress * 100) + "% complete"
            );
        }
    }

    @Override
    public void taskFinished(StatusTracker<?> task) {
        long duration = task.getElapsedRunningTime();
        if (duration > thresholdMs) {
            String taskName = StatusTracker.extractTaskName(task);
            alertService.sendAlert(
                "Slow task completed",
                taskName + " took " + duration + "ms (threshold: " + thresholdMs + "ms)"
            );
        }
    }
}
```

**Example Output (via alert service):**
```
[ALERT] Task taking too long: data-processing has been running for 35000ms but only 30.0% complete
[ALERT] Slow task completed: data-processing took 62000ms (threshold: 30000ms)
```

---

## Common Patterns

### Simple Single Task (No Explicit Scope)

The simplest pattern for tracking a single task:

```java
try (StatusContext context = new StatusContext("simple-task")) {
    context.addSink(new ConsoleLoggerSink());

    try (StatusTracker<Task> tracker = context.track(new Task())) {
        tracker.getTracked().execute();
    }
}
```

**What gets auto-created:**
- A `StatusScope` with name `"auto-scope-{taskName}"`
- Accessible via `tracker.getScope()`
- Automatically cleaned up when tracker closes

### Multiple Independent Tasks (No Explicit Scopes)

For multiple unrelated tasks:

```java
try (StatusContext context = new StatusContext("batch-work")) {
    context.addSink(new ConsoleLoggerSink());

    // Each task gets its own auto-created scope
    try (StatusTracker<Task1> t1 = context.track(new Task1());
         StatusTracker<Task2> t2 = context.track(new Task2());
         StatusTracker<Task3> t3 = context.track(new Task3())) {

        t1.getTracked().execute();
        t2.getTracked().execute();
        t3.getTracked().execute();
    }
}
```

### Sequential Workflow (With Explicit Scopes)

When you want to organize tasks explicitly:

```java
try (StatusContext context = new StatusContext("workflow");
     StatusScope scope = context.createScope("Sequential")) {

    Task1 t1 = new Task1();
    try (StatusTracker<Task1> tracker = scope.trackTask(t1)) {
        t1.execute();
    }

    Task2 t2 = new Task2();
    try (StatusTracker<Task2> tracker = scope.trackTask(t2)) {
        t2.execute();
    }
}
```

### Parallel Execution

```java
ExecutorService executor = Executors.newFixedThreadPool(4);
try (StatusContext context = new StatusContext("parallel-work");
     StatusScope scope = context.createScope("Parallel")) {

    List<Task> tasks = createTasks();

    // Submit all tasks in parallel, each with try-with-resources
    List<Future<?>> futures = tasks.stream()
        .map(task -> executor.submit(() -> {
            try (StatusTracker<Task> tracker = scope.trackTask(task)) {
                task.execute();
            } // Tracker auto-closes
        }))
        .collect(Collectors.toList());

    // Wait for all to complete
    for (Future<?> f : futures) {
        f.get();
    }
}
```

### Conditional Task Execution

```java
try (StatusContext context = new StatusContext("conditional");
     StatusScope scope = context.createScope("Work")) {

    ValidationTask validation = new ValidationTask();
    try (StatusTracker<ValidationTask> vTracker = scope.trackTask(validation)) {
        validation.execute();

        if (validation.isValid()) {
            ProcessTask process = new ProcessTask();
            try (StatusTracker<ProcessTask> pTracker = scope.trackTask(process)) {
                process.execute();
            } // pTracker auto-closes
        }
    } // vTracker auto-closes
}
```

### Long-Running Background Task

```java
try (StatusContext context = new StatusContext("background")) {
    context.addSink(ConsolePanelSink.builder().build());

    BackgroundTask task = new BackgroundTask();
    try (StatusTracker<BackgroundTask> tracker = context.track(task)) {
        // Start in background thread
        Future<?> future = executor.submit(() -> task.execute());

        // Do other work while monitoring progress
        doOtherWork();

        // Wait for completion
        future.get();
    } // Tracker auto-closes
}
```

---

### Batching for Visibility Without Sacrificing Performance

When you have a massive number of iterations (millions/billions), you can break them into batches to provide better visibility with minimal performance cost.

**The Trade-off:**
- **Fine-grained updates:** Update progress every iteration = maximum overhead
- **No updates:** Update only at end = zero visibility during execution
- **Batched updates:** Update every N iterations = good visibility, minimal overhead

**Example: Processing 1 Billion Items**

**Without batching (poor visibility):**
```java
public class MassiveTask implements StatusSource<MassiveTask> {
    private final long total = 1_000_000_000L;
    private volatile long processed = 0;
    private volatile RunState state = RunState.PENDING;

    public void process() {
        state = RunState.RUNNING;
        for (long i = 0; i < total; i++) {
            // Process item
            processItem(i);
            // No progress update - user sees 0% for hours, then suddenly 100%
        }
        processed = total; // Only update at end
        state = RunState.SUCCESS;
    }

    @Override
    public StatusUpdate<MassiveTask> getTaskStatus() {
        return new StatusUpdate<>((double) processed / total, state, this);
    }
}
```

**With batching (good visibility, minimal cost):**
```java
public class MassiveTask implements StatusSource<MassiveTask> {
    private final long total = 1_000_000_000L;
    private volatile long processed = 0;
    private volatile RunState state = RunState.PENDING;
    private final long batchSize = 10_000_000; // Update every 10M items (1% of total)

    public void process() {
        state = RunState.RUNNING;
        long batchCount = 0;

        for (long i = 0; i < total; i++) {
            // Process item
            processItem(i);

            // Batch progress updates
            batchCount++;
            if (batchCount >= batchSize) {
                processed += batchCount;  // Update progress every 10M items
                batchCount = 0;
            }
        }

        // Final update for remaining items
        processed += batchCount;
        state = RunState.SUCCESS;
    }

    @Override
    public StatusUpdate<MassiveTask> getTaskStatus() {
        return new StatusUpdate<>((double) processed / total, state, this);
    }
}
```

**Performance Analysis:**
- **Without batching:** 0 progress updates in loop, fastest but no visibility
- **With batching:** 100 progress updates (1B / 10M), provides 1% granularity
- **Performance cost:** 100 volatile writes over hours of execution = negligible
- **User experience:** Progress visible, updates every ~minute instead of never

**Choosing Batch Size:**

| Total Iterations | Suggested Batch Size | Update Frequency | Rationale |
|-----------------|---------------------|------------------|-----------|
| 1,000 | 10-100 | 10-100 updates | Fine-grained OK |
| 1,000,000 | 10,000 | 100 updates | 1% granularity |
| 1,000,000,000 | 10,000,000 | 100 updates | 1% granularity |
| 1,000,000,000,000 | 10,000,000,000 | 100 updates | 1% granularity |

**Rule of Thumb:**
- Target **100-1000 total progress updates** for the entire task
- This gives users visibility without measurable overhead
- For 1 billion iterations, updating every 10 million items = 100 updates total
- The extra second to do 100 batched updates is vastly outweighed by user visibility

**When Batching Isn't Needed:**
- Tasks with <10,000 iterations - update every iteration is fine
- Tasks already batched naturally (processing files, database pages, etc.)
- Tasks with expensive operations where increment cost is negligible

**Good Judgment Required:**
- **Too frequent:** Updates every 10 items in a billion-item loop = wasteful
- **Too infrequent:** Updates every billion items = no visibility
- **Just right:** Updates every 1% of progress = smooth visibility, minimal cost

This pattern is especially valuable for:
- Data processing pipelines
- Batch ETL jobs
- Large-scale computations
- Database migrations
- File processing

---

## Best Practices

### 1. Always Use Try-With-Resources

✅ **DO:**
```java
try (StatusContext context = new StatusContext("work");
     StatusScope scope = context.createScope("Tasks");
     StatusTracker<Task> tracker = scope.trackTask(task)) {
    task.execute();
}
```

❌ **DON'T:**
```java
StatusContext context = new StatusContext("work");
StatusScope scope = context.createScope("Tasks");
// Forgot to close - memory leak!
```

### 2. Use Volatile for Progress/State Fields

✅ **DO:**
```java
private volatile double progress;
private volatile RunState state;
```

❌ **DON'T:**
```java
private double progress;  // Not thread-safe!
private RunState state;   // Monitor thread may see stale values
```

### 3. Update Progress Monotonically

✅ **DO:**
```java
for (int i = 0; i < total; i++) {
    doWork(i);
    progress = (double) (i + 1) / total;  // Always increasing
}
```

❌ **DON'T:**
```java
progress = 0.5;
// ... later ...
progress = 0.3;  // Going backwards confuses users
```

### 4. Use Explicit Scopes for Organization

For simple, unrelated tasks, scopeless tracking is fine:

✅ **GOOD (Simple Cases):**
```java
// Single task or multiple independent tasks
try (StatusContext context = new StatusContext("work")) {
    context.addSink(new ConsoleLoggerSink());

    try (StatusTracker<Task1> t1 = context.track(new Task1());
         StatusTracker<Task2> t2 = context.track(new Task2())) {
        t1.getTracked().execute();
        t2.getTracked().execute();
    }
}
```

✅ **BETTER (Complex Workflows):**
```java
// Related tasks should be organized in scopes
try (StatusScope dataScope = context.createScope("DataProcessing");
     StatusScope ioScope = context.createScope("IOOperations")) {

    StatusTracker<DataTask> dt = dataScope.trackTask(new DataTask());
    StatusTracker<IOTask> io = ioScope.trackTask(new IOTask());

    // Can check scope completion
    while (!dataScope.isComplete()) { Thread.sleep(10); }
}
```

**When to use explicit scopes:**
- Multiple related tasks that should be grouped
- Need to check completion of a group of tasks
- Want hierarchical organization for better visualization
- Building complex pipelines with phases

### 5. Provide Meaningful Task Names

✅ **DO:**
```java
public class DataLoader implements StatusSource<DataLoader> {
    private final String source;

    public String getName() {
        return "Loading data from " + source;
    }
}
```

❌ **DON'T:**
```java
// toString() returns "DataLoader@1a2b3c" - not helpful
```

### 6. Handle Exceptions Properly

✅ **DO:**
```java
public void execute() {
    state = RunState.RUNNING;
    try {
        doWork();
        state = RunState.SUCCESS;
    } catch (Exception e) {
        state = RunState.FAILED;
        throw e;
    }
}
```

❌ **DON'T:**
```java
public void execute() {
    state = RunState.RUNNING;
    doWork();  // If this throws, state stays RUNNING
    state = RunState.SUCCESS;
}
```

---

## Troubleshooting

### Problem: Progress not updating in console

**Check:**
1. Is your task updating the `progress` field?
2. Is the field marked `volatile`?
3. Is the StatusContext still open (try-with-resources)?
4. Have you added a sink to the context?

### Problem: Tasks showing as incomplete

**Check:**
1. Did you close the tracker when the task finished?
2. Did you set the state to SUCCESS/FAILED?
3. Is the tracker's scope still open?

### Problem: Memory leak / resources not cleaned up

**Check:**
1. Are you using try-with-resources for all trackers and scopes?
2. Did you close the StatusContext?
3. Are background threads still running?

### Problem: Can't create child trackers

**Error:** `IllegalStateException: Cannot create children from a leaf tracker`

**Solution:** StatusTrackers are leaf nodes. Use StatusScope for hierarchy:

```java
// Wrong
StatusTracker<Task> parent = scope.trackTask(new Task());
parent.createChild(new SubTask());  // Error!

// Right
StatusScope parentScope = scope.createChildScope("Parent");
StatusScope childScope = parentScope.createChildScope("Child");
StatusTracker<Task> task1 = parentScope.trackTask(new Task1());
StatusTracker<Task> task2 = childScope.trackTask(new Task2());
```

---

## Additional Resources

- [README.md](README.md) - Architectural details and design rationale
- [StatusContext.java](StatusContext.java) - Session coordinator
- [StatusTracker.java](StatusTracker.java) - Task tracker implementation
- [StatusScope.java](StatusScope.java) - Organizational scope
- [StatusSource.java](eventing/StatusSource.java) - Interface for self-reporting tasks
- [StatusSink.java](eventing/StatusSink.java) - Interface for custom sinks
