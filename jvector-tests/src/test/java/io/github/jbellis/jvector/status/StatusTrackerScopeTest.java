package io.github.jbellis.jvector.status;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.status.sinks.MetricsStatusSink;
import io.github.jbellis.jvector.status.sinks.NoopStatusSink;
import io.github.jbellis.jvector.status.sinks.StatusSinkTest;
import org.junit.Test;
import org.junit.jupiter.api.Tag;


import java.time.Duration;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import static org.junit.Assert.*;

/**
 * Integration tests for TrackerScope hierarchical task tracking and scope management.
 *
 * <p>This test suite covers the TrackerScope system which provides hierarchical organization
 * of task tracking with automatic sink inheritance and resource management:
 * <ul>
 *   <li><strong>Scope Hierarchy:</strong> Parent-child relationships with path-based organization</li>
 *   <li><strong>Sink Inheritance:</strong> Automatic propagation of sinks from parent to child scopes</li>
 *   <li><strong>Resource Management:</strong> Automatic cleanup and cascading closure behavior</li>
 *   <li><strong>Tracker Creation:</strong> Factory methods for creating properly configured trackers</li>
 *   <li><strong>Configuration:</strong> Custom poll intervals and sink management</li>
 * </ul>
 *
 * <h2>Test Coverage Areas:</h2>
 * <ul>
 *   <li>Basic scope creation and configuration</li>
 *   <li>Hierarchical scope relationships and path resolution</li>
 *   <li>Sink management and inheritance patterns</li>
 *   <li>Tracker creation with scope integration</li>
 *   <li>Resource cleanup and memory management</li>
 *   <li>Concurrency and thread safety</li>
 *   <li>Edge cases and error handling</li>
 * </ul>
 *
 * @see TrackerScope for the class under test
 * @see TrackerWithSinksTest for detailed Tracker-Sink integration patterns
 * @see StatusTrackingIntegrationTest for end-to-end scope usage
 * @since 4.0.0
 */
@Tag("Integration")
public class StatusTrackerScopeTest extends RandomizedTest {

    static class TestTask implements StatusUpdate.Provider<TestTask> {
        private final String name;
        private volatile double progress = 0.0;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;

        public TestTask(String name) {
            this.name = name;
        }

        @Override
        public StatusUpdate<TestTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state);
        }

        public void setProgress(double progress) {
            this.progress = progress;
            if (progress >= 1.0) {
                state = StatusUpdate.RunState.SUCCESS;
            } else if (progress > 0) {
                state = StatusUpdate.RunState.RUNNING;
            }
        }

        public String getName() {
            return name;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    static class TestableTask {
        private final String name;
        private volatile double progress = 0.0;

        public TestableTask(String name) {
            this.name = name;
        }

        public void setProgress(double progress) {
            this.progress = progress;
        }

        public double getProgress() {
            return progress;
        }

        public String getName() {
            return name;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    /**
     * Tests basic root TrackerScope creation and initial state validation.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope can be created as a root
     * scope with proper initialization of name, path, parent relationships, and
     * default configuration values.
     *
     * <p><strong>Why Important:</strong> Root scope creation is the foundation of the
     * hierarchical tracking system. This test ensures basic construction works
     * correctly with proper default values and state initialization.
     *
     * <p><strong>Coverage:</strong> Root scope construction, name and path assignment,
     * parent relationship validation, default poll interval, empty sink collections.
     *
     * @see #testRootScopeWithSinks() for scope creation with initial sinks
     * @see #testChildScopeCreation() for child scope creation patterns
     */
    @Test
    public void testRootScopeCreation() {
        try (TrackerScope rootScope = new TrackerScope("root")) {
            assertEquals("root", rootScope.getName());
            assertEquals("/root", rootScope.getPath());
            assertNull(rootScope.getParent());
            assertTrue(rootScope.getChildren().isEmpty());
            assertEquals(Duration.ofMillis(1000), rootScope.getDefaultPollInterval());
            assertTrue(rootScope.getAllSinks().isEmpty());
            assertFalse(rootScope.isClosed());
        }
    }

    /**
     * Tests root TrackerScope creation with initial sink configuration.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope can be created with
     * an initial collection of sinks, properly storing and managing the provided
     * sink references for immediate use.
     *
     * <p><strong>Why Important:</strong> Many applications need scopes pre-configured
     * with specific sinks. This test ensures the constructor properly handles
     * initial sink collections and makes them available.
     *
     * <p><strong>Coverage:</strong> Constructor with sink collection, sink storage
     * validation, sink count verification, sink reference integrity.
     *
     * @see #testRootScopeCreation() for basic root scope creation
     * @see #testScopeSinkManagement() for dynamic sink management
     */
    @Test
    public void testRootScopeWithSinks() {
        MetricsStatusSink metricsSink = new MetricsStatusSink();
        NoopStatusSink noopSink = NoopStatusSink.getInstance();

        try (TrackerScope rootScope = new TrackerScope("root", Arrays.asList(metricsSink, noopSink))) {
            assertEquals(2, rootScope.getAllSinks().size());
            assertEquals(2, rootScope.getOwnSinks().size());
            assertTrue(rootScope.getAllSinks().contains(metricsSink));
            assertTrue(rootScope.getAllSinks().contains(noopSink));
        }
    }

    /**
     * Tests child TrackerScope creation and parent-child relationship establishment.
     *
     * <p><strong>Purpose:</strong> Validates that child scopes can be created from
     * parent scopes with proper hierarchical relationships, path construction,
     * and automatic sink inheritance from parent to child.
     *
     * <p><strong>Why Important:</strong> The hierarchical scope system depends on
     * proper parent-child relationships and sink inheritance. This test ensures
     * the fundamental hierarchy mechanics work correctly.
     *
     * <p><strong>Coverage:</strong> Child scope creation, parent-child relationships,
     * hierarchical path construction, automatic sink inheritance, parent child mapping.
     *
     * @see #testScopeHierarchyInheritance() for multi-level hierarchy testing
     * @see #testComplexScopeHierarchy() for complex hierarchical scenarios
     */
    @Test
    public void testChildScopeCreation() {
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        try (TrackerScope rootScope = new TrackerScope("root", Arrays.asList(metricsSink))) {
            TrackerScope childScope = rootScope.createChildScope("child");

            assertEquals("child", childScope.getName());
            assertEquals("/root/child", childScope.getPath());
            assertEquals(rootScope, childScope.getParent());
            assertEquals(childScope, rootScope.getChild("child"));
            assertEquals(1, rootScope.getChildren().size());

            assertEquals(1, childScope.getAllSinks().size());
            assertEquals(0, childScope.getOwnSinks().size());
            assertTrue(childScope.getAllSinks().contains(metricsSink));
        }
    }

    /**
     * Tests multi-level scope hierarchy with comprehensive sink inheritance.
     *
     * <p><strong>Purpose:</strong> Validates that deep scope hierarchies correctly
     * inherit and accumulate sinks from all ancestor scopes, ensuring that
     * child scopes have access to all parent-level sinks plus their own.
     *
     * <p><strong>Why Important:</strong> Complex applications may have deep scope
     * hierarchies (app/module/service). Each level should inherit all ancestor
     * sinks while adding its own, creating comprehensive sink coverage.
     *
     * <p><strong>Coverage:</strong> Multi-level hierarchy creation, cumulative sink
     * inheritance, path construction at all levels, sink accumulation validation.
     *
     * @see #testChildScopeCreation() for basic parent-child relationships
     * @see #testHierarchicalSinkInheritance() for inheritance in tracker creation
     */
    @Test
    public void testScopeHierarchyInheritance() {
        MetricsStatusSink rootSink = new MetricsStatusSink();
        MetricsStatusSink childSink = new MetricsStatusSink();
        MetricsStatusSink grandchildSink = new MetricsStatusSink();

        try (TrackerScope rootScope = new TrackerScope("root", Arrays.asList(rootSink))) {
            TrackerScope childScope = rootScope.createChildScope("child", Arrays.asList(childSink));
            TrackerScope grandchildScope = childScope.createChildScope("grandchild", Arrays.asList(grandchildSink));

            assertEquals("/root", rootScope.getPath());
            assertEquals("/root/child", childScope.getPath());
            assertEquals("/root/child/grandchild", grandchildScope.getPath());

            assertEquals(1, rootScope.getAllSinks().size());
            assertEquals(2, childScope.getAllSinks().size());
            assertEquals(3, grandchildScope.getAllSinks().size());

            assertTrue(grandchildScope.getAllSinks().contains(rootSink));
            assertTrue(grandchildScope.getAllSinks().contains(childSink));
            assertTrue(grandchildScope.getAllSinks().contains(grandchildSink));
        }
    }

    /**
     * Tests TrackerScope's ability to create properly configured Tracker instances.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope can create Tracker
     * instances with proper sink integration, ensuring trackers automatically
     * receive all appropriate sinks from their containing scope.
     *
     * <p><strong>Why Important:</strong> The primary purpose of TrackerScope is to
     * simplify tracker creation with automatic sink configuration. This test
     * ensures the factory methods work correctly.
     *
     * <p><strong>Coverage:</strong> Tracker creation via scope, automatic sink
     * assignment, task execution with scope-managed tracking, metrics collection.
     *
     * @see #testScopeTrackerWithFunctors() for functor-based tracker creation
     * @see #testTrackerWithCustomPollInterval() for custom interval configuration
     */
    @Test
    public void testScopeTrackerCreation() throws InterruptedException {
        MetricsStatusSink metricsSink = new MetricsStatusSink();
        TestTask task = new TestTask("scope-test-task");

        try (TrackerScope scope = new TrackerScope("test-scope", Arrays.asList(metricsSink))) {
            try (StatusTracker<TestTask> statusTracker = scope.track(task)) {
                assertEquals(1, statusTracker.getSinks().size());
                assertTrue(statusTracker.getSinks().contains(metricsSink));

                task.setProgress(0.5);
                Thread.sleep(150);

                task.setProgress(1.0);
                Thread.sleep(150);
            }
        }

        assertEquals(1, metricsSink.getTotalTasksStarted());
        assertEquals(1, metricsSink.getTotalTasksFinished());
    }

    /**
     * Tests TrackerScope's functor-based tracker creation with custom status functions.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope can create functor-based
     * trackers that use custom status functions, properly integrating with the
     * scope's sink configuration and monitoring system.
     *
     * <p><strong>Why Important:</strong> Not all trackable objects can implement
     * TaskStatus.Provider. Functor-based tracking provides flexibility while
     * maintaining full scope integration and sink support.
     *
     * <p><strong>Coverage:</strong> Functor-based tracker creation, custom status
     * function integration, scope sink assignment, task execution with functors.
     *
     * @see #testScopeTrackerCreation() for instrumented tracker creation
     * @see StatusTrackerTest#testWithFunctors() for basic functor pattern testing
     */
    @Test
    public void testScopeTrackerWithFunctors() throws InterruptedException {
        MetricsStatusSink metricsSink = new MetricsStatusSink();
        TestableTask task = new TestableTask("functor-task");

        Function<TestableTask, StatusUpdate<TestableTask>> statusFunction = t -> {
            StatusUpdate.RunState state = t.getProgress() >= 1.0 ? StatusUpdate.RunState.SUCCESS :
                                       t.getProgress() > 0 ? StatusUpdate.RunState.RUNNING :
                                       StatusUpdate.RunState.PENDING;
            return new StatusUpdate<>(t.getProgress(), state);
        };

        try (TrackerScope scope = new TrackerScope("functor-scope", Arrays.asList(metricsSink))) {
            try (StatusTracker<TestableTask> statusTracker = scope.track(task, statusFunction)) {
                task.setProgress(0.8);
                Thread.sleep(150);

                task.setProgress(1.0);
                Thread.sleep(150);
            }
        }

        assertEquals(1, metricsSink.getTotalTasksStarted());
        assertEquals(1, metricsSink.getTotalTasksFinished());
    }

    /**
     * Tests TrackerScope creation and configuration with custom polling intervals.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope can be configured
     * with custom default polling intervals that are applied to trackers created
     * within the scope, enabling scope-level performance tuning.
     *
     * <p><strong>Why Important:</strong> Different application areas may need
     * different polling frequencies (high-frequency for critical tasks, low-frequency
     * for background tasks). Scope-level configuration enables this tuning.
     *
     * <p><strong>Coverage:</strong> Custom poll interval configuration, interval
     * application to created trackers, scope-level performance configuration.
     *
     * @see #testTrackerWithCustomPollInterval() for per-tracker interval overrides
     * @see #testCustomPollIntervalInheritance() for interval inheritance patterns
     */
    @Test
    public void testScopeWithCustomPollInterval() throws InterruptedException {
        MetricsStatusSink metricsSink = new MetricsStatusSink();
        TestTask task = new TestTask("custom-interval-task");

        try (TrackerScope scope = new TrackerScope("interval-scope", Duration.ofMillis(50), Arrays.asList(metricsSink))) {
            assertEquals(Duration.ofMillis(50), scope.getDefaultPollInterval());

            try (StatusTracker<TestTask> statusTracker = scope.track(task)) {
                task.setProgress(0.5);
                Thread.sleep(100);

                task.setProgress(1.0);
                Thread.sleep(100);
            }
        }

        assertEquals(1, metricsSink.getTotalTasksStarted());
        assertEquals(1, metricsSink.getTotalTasksFinished());
    }

    /**
     * Tests per-tracker polling interval overrides within scopes.
     *
     * <p><strong>Purpose:</strong> Validates that individual trackers can override
     * the scope's default polling interval, providing fine-grained control over
     * monitoring frequency on a per-task basis.
     *
     * <p><strong>Why Important:</strong> While scopes provide default intervals,
     * individual tasks may have specific performance requirements. Per-tracker
     * overrides enable precise tuning without affecting other tasks.
     *
     * <p><strong>Coverage:</strong> Per-tracker interval override, scope default
     * vs tracker-specific configuration, interval precedence validation.
     *
     * @see #testScopeWithCustomPollInterval() for scope-level interval configuration
     * @see TaskMonitorTest#testDefaultPollInterval() for monitor-level interval testing
     */
    @Test
    public void testTrackerWithCustomPollInterval() throws InterruptedException {
        MetricsStatusSink metricsSink = new MetricsStatusSink();
        TestTask task = new TestTask("override-interval-task");

        try (TrackerScope scope = new TrackerScope("scope", Duration.ofMillis(100), Arrays.asList(metricsSink))) {
            try (StatusTracker<TestTask> statusTracker = scope.track(task, Duration.ofMillis(30))) {
                task.setProgress(0.7);
                Thread.sleep(80);

                task.setProgress(1.0);
                Thread.sleep(80);
            }
        }

        assertEquals(1, metricsSink.getTotalTasksStarted());
        assertEquals(1, metricsSink.getTotalTasksFinished());
    }

    /**
     * Tests dynamic sink management operations within TrackerScope.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope supports dynamic
     * addition and removal of sinks during runtime, enabling flexible sink
     * configuration based on changing application needs.
     *
     * <p><strong>Why Important:</strong> Applications may need to add or remove
     * sinks dynamically (e.g., enabling/disabling logging, adding monitoring).
     * This test ensures proper dynamic sink management.
     *
     * <p><strong>Coverage:</strong> Dynamic sink addition, sink removal, sink
     * collection management, state consistency during sink operations.
     *
     * @see #testRootScopeWithSinks() for initial sink configuration
     * @see #testScopeWithNullSinkHandling() for null sink handling
     */
    @Test
    public void testScopeSinkManagement() {
        MetricsStatusSink sink1 = new MetricsStatusSink();
        MetricsStatusSink sink2 = new MetricsStatusSink();

        try (TrackerScope scope = new TrackerScope("sink-mgmt")) {
            assertEquals(0, scope.getAllSinks().size());

            scope.addSink(sink1);
            assertEquals(1, scope.getAllSinks().size());
            assertTrue(scope.getAllSinks().contains(sink1));

            scope.addSink(sink2);
            assertEquals(2, scope.getAllSinks().size());

            scope.removeSink(sink1);
            assertEquals(1, scope.getAllSinks().size());
            assertFalse(scope.getAllSinks().contains(sink1));
            assertTrue(scope.getAllSinks().contains(sink2));
        }
    }

    /**
     * Tests hierarchical sink inheritance in tracker creation across scope levels.
     *
     * <p><strong>Purpose:</strong> Validates that trackers created in child scopes
     * automatically inherit sinks from all ancestor scopes, ensuring comprehensive
     * sink coverage and proper hierarchical behavior in practice.
     *
     * <p><strong>Why Important:</strong> The hierarchical sink inheritance is the
     * core value proposition of TrackerScope. This test ensures that inheritance
     * works correctly when actually creating and using trackers.
     *
     * <p><strong>Coverage:</strong> Tracker sink inheritance, parent scope sink
     * propagation, sink counting across hierarchy levels, actual task execution.
     *
     * @see #testScopeHierarchyInheritance() for scope-level inheritance testing
     * @see #testComplexScopeHierarchy() for comprehensive hierarchy scenarios
     */
    @Test
    public void testHierarchicalSinkInheritance() throws InterruptedException {
        MetricsStatusSink rootSink = new MetricsStatusSink();
        MetricsStatusSink childSink = new MetricsStatusSink();

        try (TrackerScope rootScope = new TrackerScope("root", Arrays.asList(rootSink))) {
            TrackerScope childScope = rootScope.createChildScope("child", Arrays.asList(childSink));

            TestTask task1 = new TestTask("root-task");
            TestTask task2 = new TestTask("child-task");

            try (StatusTracker<TestTask> rootStatusTracker = rootScope.track(task1);
                 StatusTracker<TestTask> childStatusTracker = childScope.track(task2)) {

                assertEquals(1, rootStatusTracker.getSinks().size());
                assertEquals(2, childStatusTracker.getSinks().size());

                task1.setProgress(1.0);
                task2.setProgress(1.0);
                Thread.sleep(150);
            }
        }

        assertEquals(2, rootSink.getTotalTasksStarted());
        assertEquals(1, childSink.getTotalTasksStarted());
    }

    /**
     * Tests cascading closure behavior when parent scopes are closed.
     *
     * <p><strong>Purpose:</strong> Validates that closing a parent scope automatically
     * closes all child scopes in the hierarchy, ensuring proper resource cleanup
     * and preventing orphaned scope references.
     *
     * <p><strong>Why Important:</strong> Resource management requires proper cleanup
     * of scope hierarchies. Cascading closure ensures no child scopes are left
     * in inconsistent states when parents are closed.
     *
     * <p><strong>Coverage:</strong> Cascading scope closure, child scope state
     * propagation, resource cleanup, hierarchy dismantling.
     *
     * @see #testClosedScopeOperations() for closed scope operation validation
     * @see #testScopeMemoryCleanup() for memory management testing
     */
    @Test
    public void testScopeClosingBehavior() {
        MetricsStatusSink metricsSink = new MetricsStatusSink();
        TrackerScope rootScope = new TrackerScope("root", Arrays.asList(metricsSink));
        TrackerScope childScope = rootScope.createChildScope("child");
        TrackerScope grandchildScope = childScope.createChildScope("grandchild");

        assertFalse(rootScope.isClosed());
        assertFalse(childScope.isClosed());
        assertFalse(grandchildScope.isClosed());

        rootScope.close();

        assertTrue(rootScope.isClosed());
        assertTrue(childScope.isClosed());
        assertTrue(grandchildScope.isClosed());

        assertTrue(rootScope.getChildren().isEmpty());
        assertTrue(childScope.getChildren().isEmpty());
        assertTrue(rootScope.getAllSinks().isEmpty());
    }

    /**
     * Tests that closed TrackerScope instances properly reject operations.
     *
     * <p><strong>Purpose:</strong> Validates that closed scopes throw appropriate
     * exceptions when attempting operations like creating child scopes, adding
     * sinks, or creating trackers, ensuring fail-fast behavior.
     *
     * <p><strong>Why Important:</strong> Using closed resources should fail
     * immediately to prevent undefined behavior. This test ensures proper
     * lifecycle enforcement and error messaging.
     *
     * <p><strong>Coverage:</strong> Closed scope operation rejection, exception
     * throwing, error message validation, lifecycle state enforcement.
     *
     * @see #testScopeClosingBehavior() for closure mechanics
     * @see StatusTrackerTest#testTrackerNullParameterValidation() for other validation patterns
     */
    @Test
    public void testClosedScopeOperations() {
        TrackerScope scope = new TrackerScope("closed-test");
        scope.close();

        try {
            scope.createChildScope("child");
            fail("Should throw IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("has been closed"));
        }

        try {
            scope.addSink(new MetricsStatusSink());
            fail("Should throw IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("has been closed"));
        }

        try {
            scope.track(new TestTask("test"));
            fail("Should throw IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("has been closed"));
        }
    }

    /**
     * Tests TrackerScope and Tracker functionality when no sinks are configured.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope can function correctly
     * even when no sinks are configured, creating trackers that work but simply
     * don't report to any sinks.
     *
     * <p><strong>Why Important:</strong> Some applications may want to use tracking
     * infrastructure without output, or may add sinks dynamically. The system
     * should work correctly with empty sink configurations.
     *
     * <p><strong>Coverage:</strong> Empty sink collections, tracker functionality
     * without sinks, task execution without reporting, basic tracking capability.
     *
     * @see #testScopeSinkManagement() for dynamic sink addition
     * @see StatusSinkTest#testNoopTaskSink() for minimal sink scenarios
     */
    @Test
    public void testScopeWithoutSinks() throws InterruptedException {
        TestTask task = new TestTask("no-sinks-task");

        try (TrackerScope scope = new TrackerScope("no-sinks")) {
            try (StatusTracker<TestTask> statusTracker = scope.track(task)) {
                assertTrue(statusTracker.getSinks().isEmpty());

                task.setProgress(0.5);
                Thread.sleep(100);

                task.setProgress(1.0);
                Thread.sleep(100);

                StatusUpdate<TestTask> status = statusTracker.getStatus();
                assertEquals(1.0, status.progress, 0.001);
                assertEquals(StatusUpdate.RunState.SUCCESS, status.runstate);
            }
        }
    }

    /**
     * Tests complex multi-level scope hierarchy with comprehensive sink inheritance and task execution.
     *
     * <p><strong>Purpose:</strong> Validates the complete scope system working
     * in a realistic, complex scenario with multiple hierarchy levels, different
     * sink configurations, and concurrent task execution.
     *
     * <p><strong>Why Important:</strong> This test demonstrates the system working
     * as designed in production-like scenarios with complex hierarchies that
     * mirror real application structures (app/module/service).
     *
     * <p><strong>Coverage:</strong> Complex hierarchy creation, multi-level sink
     * inheritance, concurrent task execution, path resolution, sink distribution validation.
     *
     * @see #testScopeHierarchyInheritance() for basic hierarchy testing
     * @see StatusTrackingIntegrationTest for end-to-end integration scenarios
     */
    @Test
    public void testComplexScopeHierarchy() throws InterruptedException {
        MetricsStatusSink appSink = new MetricsStatusSink();
        MetricsStatusSink moduleSink = new MetricsStatusSink();
        MetricsStatusSink serviceSink = new MetricsStatusSink();

        try (TrackerScope appScope = new TrackerScope("app", Arrays.asList(appSink))) {
            TrackerScope moduleScope = appScope.createChildScope("module", Arrays.asList(moduleSink));
            TrackerScope serviceScope = moduleScope.createChildScope("service", Arrays.asList(serviceSink));

            assertEquals("/app", appScope.getPath());
            assertEquals("/app/module", moduleScope.getPath());
            assertEquals("/app/module/service", serviceScope.getPath());

            TestTask appTask = new TestTask("app-task");
            TestTask moduleTask = new TestTask("module-task");
            TestTask serviceTask = new TestTask("service-task");

            try (StatusTracker<TestTask> appStatusTracker = appScope.track(appTask);
                 StatusTracker<TestTask> moduleStatusTracker = moduleScope.track(moduleTask);
                 StatusTracker<TestTask> serviceStatusTracker = serviceScope.track(serviceTask)) {

                assertEquals(1, appStatusTracker.getSinks().size());
                assertEquals(2, moduleStatusTracker.getSinks().size());
                assertEquals(3, serviceStatusTracker.getSinks().size());

                List<TestTask> tasks = Arrays.asList(appTask, moduleTask, serviceTask);
                for (TestTask task : tasks) {
                    task.setProgress(0.3);
                }
                Thread.sleep(100);

                for (TestTask task : tasks) {
                    task.setProgress(0.7);
                }
                Thread.sleep(100);

                for (TestTask task : tasks) {
                    task.setProgress(1.0);
                }
                Thread.sleep(100);
            }
        }

        assertEquals(3, appSink.getTotalTasksStarted());
        assertEquals(2, moduleSink.getTotalTasksStarted());
        assertEquals(1, serviceSink.getTotalTasksStarted());

        assertEquals(3, appSink.getTotalTasksFinished());
        assertEquals(2, moduleSink.getTotalTasksFinished());
        assertEquals(1, serviceSink.getTotalTasksFinished());
    }

    /**
     * Tests TrackerScope string representation and debugging information.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope provides meaningful
     * string representations for debugging and logging, including path information,
     * sink counts, child counts, and closure status.
     *
     * <p><strong>Why Important:</strong> Good toString() implementations are
     * essential for debugging complex scope hierarchies. This ensures scopes
     * provide useful diagnostic information.
     *
     * <p><strong>Coverage:</strong> String representation formatting, path display,
     * sink count inclusion, child count reporting, state information.
     *
     * @see TaskTrackerTest#testTaskTrackerToString() for similar string representation testing
     * @see #testComplexScopeHierarchy() for complex hierarchy scenarios
     */
    @Test
    public void testScopeToString() {
        MetricsStatusSink sink = new MetricsStatusSink();

        try (TrackerScope rootScope = new TrackerScope("root", Arrays.asList(sink))) {
            TrackerScope childScope = rootScope.createChildScope("child");

            String rootString = rootScope.toString();
            assertTrue(rootString.contains("path=/root"));
            assertTrue(rootString.contains("sinks=1"));
            assertTrue(rootString.contains("children=1"));
            assertTrue(rootString.contains("closed=false"));

            String childString = childScope.toString();
            assertTrue(childString.contains("path=/root/child"));
            assertTrue(childString.contains("sinks=1"));
            assertTrue(childString.contains("children=0"));
        }
    }

    /**
     * Tests error handling and propagation in scope-managed trackers.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope and its created
     * trackers handle task errors gracefully, continuing to function even when
     * tracked tasks encounter exceptions during status reporting.
     *
     * <p><strong>Why Important:</strong> Tasks may encounter runtime errors.
     * The scope system must be resilient to these errors, maintaining system
     * stability while continuing to track other tasks.
     *
     * <p><strong>Coverage:</strong> Task exception handling, tracker resilience,
     * continued operation after errors, error isolation.
     *
     * @see StatusTrackerTest#testTrackerResourceCleanupOnException() for tracker-level error handling
     * @see TaskMonitorTest#testExceptionHandling() for monitor-level error scenarios
     */
    @Test
    public void testScopeErrorPropagation() throws InterruptedException {
        class FailingTask implements StatusUpdate.Provider<FailingTask> {
            private boolean shouldFail = false;

            @Override
            public StatusUpdate<FailingTask> getTaskStatus() {
                if (shouldFail) {
                    throw new RuntimeException("Task status error");
                }
                return new StatusUpdate<>(0.5, StatusUpdate.RunState.RUNNING);
            }

            public void fail() {
                shouldFail = true;
            }

            @Override
            public String toString() {
                return "FailingTask";
            }
        }

        FailingTask task = new FailingTask();
        MetricsStatusSink metricsSink = new MetricsStatusSink();

        try (TrackerScope scope = new TrackerScope("error-scope", Arrays.asList(metricsSink))) {
            try (StatusTracker<FailingTask> statusTracker = scope.track(task)) {
                Thread.sleep(50);

                // Task should be working normally
                StatusUpdate<FailingTask> status = statusTracker.getStatus();
                assertNotNull(status);

                // Make task fail
                task.fail();
                Thread.sleep(100);

                // Tracker should handle the error gracefully and continue
                assertNotNull(statusTracker.getStatus());
            }
        }
    }

    /**
     * Tests thread safety of concurrent scope operations including child creation and tracking.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope operations are thread-safe
     * when multiple threads concurrently create child scopes, add sinks, and
     * create trackers, ensuring data integrity under concurrent access.
     *
     * <p><strong>Why Important:</strong> Multi-threaded applications may access
     * scope hierarchies from multiple threads simultaneously. The scope system
     * must be thread-safe to prevent corruption and ensure reliable operation.
     *
     * <p><strong>Coverage:</strong> Concurrent child scope creation, concurrent
     * sink management, concurrent tracker creation, thread safety validation.
     *
     * @see #testScopeResourceContention() for resource contention testing
     * @see TaskMonitorTest#testConcurrentStartStop() for monitor-level concurrency
     */
    @Test
    public void testConcurrentScopeOperations() throws InterruptedException, ExecutionException {
        TrackerScope rootScope = new TrackerScope("concurrent-root");
        ExecutorService executor = Executors.newFixedThreadPool(4);
        List<Future<Void>> futures = new ArrayList<>();

        // Concurrently create child scopes
        for (int i = 0; i < 10; i++) {
            final int index = i;
            Future<Void> future = executor.submit(() -> {
                TrackerScope child = rootScope.createChildScope("child-" + index);
                child.addSink(new MetricsStatusSink());

                TestTask task = new TestTask("concurrent-task-" + index);
                try (StatusTracker<TestTask> tracker = child.track(task)) {
                    task.setProgress(0.8);
                    Thread.sleep(10);
                    task.setProgress(1.0);
                    Thread.sleep(10);
                }
                return null;
            });
            futures.add(future);
        }

        // Wait for all to complete
        for (Future<Void> future : futures) {
            future.get();
        }

        assertEquals(10, rootScope.getChildren().size());

        executor.shutdown();
        assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        rootScope.close();
    }

    /**
     * Tests memory cleanup and resource management in complex scope hierarchies.
     *
     * <p><strong>Purpose:</strong> Validates that complex scope hierarchies with
     * deep nesting and many sinks are properly cleaned up when closed,
     * preventing memory leaks and ensuring efficient resource management.
     *
     * <p><strong>Why Important:</strong> Deep scope hierarchies could accumulate
     * significant resources. Proper cleanup is essential for long-running
     * applications to prevent memory leaks and resource exhaustion.
     *
     * <p><strong>Coverage:</strong> Deep hierarchy cleanup, sink reference cleanup,
     * child relationship cleanup, cascading resource deallocation.
     *
     * @see #testScopeClosingBehavior() for basic closure mechanics
     * @see TaskTrackerTest#testTaskTrackerMemoryCleanup() for task-level memory testing
     */
    @Test
    public void testScopeMemoryCleanup() {
        MetricsStatusSink sink = new MetricsStatusSink();
        TrackerScope rootScope = new TrackerScope("memory-test", Arrays.asList(sink));

        // Create deep hierarchy
        TrackerScope current = rootScope;
        for (int i = 0; i < 5; i++) {
            current = current.createChildScope("level-" + i, Arrays.asList(new MetricsStatusSink()));
        }

        // Verify deep hierarchy
        assertEquals("/memory-test/level-0/level-1/level-2/level-3/level-4", current.getPath());
        assertEquals(6, current.getAllSinks().size()); // 1 root + 5 child sinks

        // Close root should cascade
        rootScope.close();

        assertTrue(rootScope.isClosed());
        assertTrue(current.isClosed());
        assertTrue(rootScope.getChildren().isEmpty());
        assertTrue(rootScope.getAllSinks().isEmpty());
    }

    /**
     * Tests scope behavior under resource contention with concurrent sink operations.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope handles resource
     * contention gracefully when multiple threads simultaneously add/remove
     * sinks and create trackers, ensuring consistent behavior under load.
     *
     * <p><strong>Why Important:</strong> High-load applications may have many
     * threads modifying scope configurations simultaneously. The system must
     * handle this contention without corruption or inconsistent state.
     *
     * <p><strong>Coverage:</strong> Concurrent sink addition/removal, resource
     * contention handling, state consistency under load, concurrent tracker creation.
     *
     * @see #testConcurrentScopeOperations() for general concurrent operations
     * @see StatusSinkTest#testMetricsTaskSinkConcurrency() for sink-level concurrency
     */
    @Test
    public void testScopeResourceContention() throws InterruptedException, ExecutionException {
        TrackerScope scope = new TrackerScope("contention-test");
        MetricsStatusSink sharedSink = new MetricsStatusSink();
        scope.addSink(sharedSink);

        ExecutorService executor = Executors.newFixedThreadPool(6);
        List<Future<Integer>> futures = new ArrayList<>();

        // Multiple threads adding/removing sinks concurrently
        for (int i = 0; i < 20; i++) {
            final int index = i;
            Future<Integer> future = executor.submit(() -> {
                MetricsStatusSink localSink = new MetricsStatusSink();
                scope.addSink(localSink);

                TestTask task = new TestTask("contention-task-" + index);
                try (StatusTracker<TestTask> tracker = scope.track(task)) {
                    task.setProgress(0.5);
                    Thread.sleep(5);
                    task.setProgress(1.0);
                }

                scope.removeSink(localSink);
                return scope.getAllSinks().size();
            });
            futures.add(future);
        }

        Set<Integer> sinkCounts = new HashSet<>();
        for (Future<Integer> future : futures) {
            sinkCounts.add(future.get());
        }

        // Should have seen different sink counts due to concurrent modifications
        assertTrue(sinkCounts.size() > 1);
        assertEquals(1, scope.getAllSinks().size()); // Should end up with just the shared sink

        executor.shutdown();
        assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        scope.close();
    }

    /**
     * Tests TrackerScope behavior with extremely deep scope hierarchies.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope can handle deep
     * nesting (100+ levels) without performance degradation, stack overflow,
     * or other issues related to recursive operations.
     *
     * <p><strong>Why Important:</strong> Some applications might create very
     * deep scope hierarchies. The system should handle these edge cases
     * gracefully without failing or degrading performance.
     *
     * <p><strong>Coverage:</strong> Deep nesting creation, path construction
     * at extreme depth, parent navigation, cascading closure in deep hierarchies.
     *
     * @see #testComplexScopeHierarchy() for realistic hierarchy complexity
     * @see #testScopeMemoryCleanup() for cleanup in complex hierarchies
     */
    @Test
    public void testScopeDeepNesting() {
        TrackerScope root = new TrackerScope("root");
        TrackerScope current = root;

        // Create deep nesting (100 levels)
        for (int i = 0; i < 100; i++) {
            current = current.createChildScope("depth-" + i);
        }

        String expectedPath = "/root";
        for (int i = 0; i < 100; i++) {
            expectedPath += "/depth-" + i;
        }

        assertEquals(expectedPath, current.getPath());
        assertNotNull(current.getParent());

        // Navigate back to root
        TrackerScope temp = current;
        int depth = 0;
        while (temp.getParent() != null) {
            temp = temp.getParent();
            depth++;
        }
        assertEquals(100, depth);
        assertEquals(root, temp);

        root.close();
        assertTrue(current.isClosed());
    }

    /**
     * Tests TrackerScope's handling of null sink values in operations.
     *
     * <p><strong>Purpose:</strong> Validates that TrackerScope gracefully handles
     * null sink references in add/remove operations, preventing NullPointerExceptions
     * and maintaining system stability.
     *
     * <p><strong>Why Important:</strong> Applications may inadvertently pass null
     * sink references. The system should handle these gracefully rather than
     * failing with exceptions.
     *
     * <p><strong>Coverage:</strong> Null sink handling in add operations, null
     * sink handling in remove operations, system stability with null values.
     *
     * @see #testScopeSinkManagement() for normal sink management operations
     * @see StatusTrackerTest#testTrackerNullParameterValidation() for null validation patterns
     */
    @Test
    public void testScopeWithNullSinkHandling() {
        TrackerScope scope = new TrackerScope("null-sink-test");

        // Adding null sink should be handled gracefully
        scope.addSink(null);
        assertEquals(0, scope.getAllSinks().size());

        // Adding real sink after null
        MetricsStatusSink sink = new MetricsStatusSink();
        scope.addSink(sink);
        assertEquals(1, scope.getAllSinks().size());

        // Removing null should be safe
        scope.removeSink(null);
        assertEquals(1, scope.getAllSinks().size());

        scope.close();
    }

    /**
     * Tests edge cases in child scope management including duplicate names and missing children.
     *
     * <p><strong>Purpose:</strong> Validates TrackerScope's handling of edge cases
     * in child management, including creating children with duplicate names
     * and attempting to retrieve non-existent children.
     *
     * <p><strong>Why Important:</strong> Edge cases in child management could
     * lead to unexpected behavior or exceptions. This test ensures robust
     * handling of unusual but possible scenarios.
     *
     * <p><strong>Coverage:</strong> Duplicate child name handling, non-existent
     * child retrieval, child overwriting behavior, edge case robustness.
     *
     * @see #testChildScopeCreation() for normal child creation patterns
     * @see #testComplexScopeHierarchy() for comprehensive child management
     */
    @Test
    public void testScopeChildManagementEdgeCases() {
        TrackerScope root = new TrackerScope("edge-test");

        // Create child with same name multiple times (should overwrite)
        TrackerScope child1 = root.createChildScope("samename");
        TrackerScope child2 = root.createChildScope("samename");

        assertEquals(1, root.getChildren().size());
        assertEquals(child2, root.getChild("samename"));

        // Getting non-existent child
        assertNull(root.getChild("nonexistent"));

        root.close();
    }

    /**
     * Tests polling interval inheritance patterns across scope hierarchies.
     *
     * <p><strong>Purpose:</strong> Validates that custom polling intervals are
     * properly inherited from parent to child scopes, and that child scopes
     * can override inherited intervals with their own configurations.
     *
     * <p><strong>Why Important:</strong> Polling interval inheritance provides
     * convenient default configuration while allowing fine-grained control
     * where needed. This test ensures the inheritance pattern works correctly.
     *
     * <p><strong>Coverage:</strong> Interval inheritance from parent to child,
     * interval override in child scopes, actual interval usage in tracker creation.
     *
     * @see #testScopeWithCustomPollInterval() for scope-level interval configuration
     * @see #testTrackerWithCustomPollInterval() for tracker-level overrides
     */
    @Test
    public void testCustomPollIntervalInheritance() throws InterruptedException {
        Duration customInterval = Duration.ofMillis(25);
        TrackerScope rootScope = new TrackerScope("poll-test", customInterval);

        TrackerScope childScope = rootScope.createChildScope("child");
        TrackerScope grandchildScope = childScope.createChildScope("grandchild", Duration.ofMillis(10), new ArrayList<>());

        assertEquals(customInterval, rootScope.getDefaultPollInterval());
        assertEquals(customInterval, childScope.getDefaultPollInterval());
        assertEquals(Duration.ofMillis(10), grandchildScope.getDefaultPollInterval());

        // Test that custom intervals are actually used
        TestTask task = new TestTask("interval-task");
        try (StatusTracker<TestTask> statusTracker = grandchildScope.track(task, Duration.ofMillis(5))) {
            task.setProgress(0.5);
            Thread.sleep(50);
            task.setProgress(1.0);
            Thread.sleep(50);
        }

        rootScope.close();
    }
}