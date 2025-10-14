/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.status.sinks;

import io.github.jbellis.jvector.status.eventing.RunState;
import io.github.jbellis.jvector.status.eventing.StatusSink;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.TrackerScope;
import io.github.jbellis.jvector.status.eventing.StatusUpdate;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jline.utils.NonBlockingReader;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.terminal.Size;
import org.jline.utils.AttributedString;
import org.jline.utils.AttributedStringBuilder;
import org.jline.utils.AttributedStyle;
import org.jline.utils.Display;


import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * A sophisticated terminal-based status sink that provides a hierarchical, stateful view
 * of task progress using JLine3. This enhanced version includes a scrollable logging panel,
 * captures all console output for integrated display, and supports custom keyboard handlers
 * for application-specific interactivity.
 *
 * <p>Features:
 * <ul>
 *   <li>Hierarchical task display with parent-child relationships</li>
 *   <li>Real-time updates without terminal scrolling</li>
 *   <li>Scrollable logging panel for console output</li>
 *   <li>Full terminal control with output redirection</li>
 *   <li>Color-coded status indicators</li>
 *   <li>Progress bars with percentage display</li>
 *   <li>Task duration tracking</li>
 *   <li>Automatic cleanup of completed tasks</li>
 *   <li>Custom keyboard handlers for application-specific controls</li>
 *   <li>Interactive search and filtering capabilities</li>
 * </ul>
 *
 * <h2>Display Layout:</h2>
 * <pre>
 * ╔═══ Task Status Monitor ═══════════════════════════════════════╗
 * ║                                                               ║
 * ║ ▶ [14:32:15] RootTask [████████████░░░░░░░░]  60% (2.3s)    ║
 * ║   ├─ ● [14:32:16] SubTask1 [████████████████████] 100% ✓    ║
 * ║   └─ ▶ [14:32:17] SubTask2 [██████░░░░░░░░░░░░░░]  30%     ║
 * ║                                                               ║
 * ║ Active: 2 | Completed: 1 | Failed: 0                        ║
 * ╠═══ Console Output ════════════════════════════════════════════╣
 * ║ [INFO ] Starting data processing...                          ║
 * ║ [DEBUG] Loading configuration from file                      ║
 * ║ [WARN ] Cache miss for key: user_123                        ║
 * ║ [INFO ] Processing batch 1 of 10                            ║
 * ║ ▼ (↑/↓ to scroll, 4 more lines)                             ║
 * ╚═══════════════════════════════════════════════════════════════╝
 * </pre>
 *
 * <h2>Built-in Keyboard Controls:</h2>
 * <ul>
 *   <li><strong>↑ / ↓:</strong> Scroll through console log output</li>
 *   <li><strong>[ / ]:</strong> Adjust split between task and log panels</li>
 *   <li><strong>PgUp / PgDn:</strong> Quick split adjustment</li>
 *   <li><strong>Home:</strong> Reset scroll positions and split to defaults</li>
 *   <li><strong>End:</strong> Jump to end of logs and tasks</li>
 *   <li><strong>s:</strong> Save current display to file</li>
 *   <li><strong>q:</strong> Quit and shutdown (when auto-exit enabled)</li>
 *   <li><strong>?:</strong> Show help panel</li>
 *   <li><strong>/:</strong> Enter search mode</li>
 * </ul>
 *
 * <h2>Custom Keyboard Handlers:</h2>
 * <p>Applications can register custom keyboard handlers using
 * {@link Builder#withKeyHandler(String, Runnable)} to extend the interactive
 * capabilities. Currently supported custom key combinations include:</p>
 * <ul>
 *   <li><strong>shift-left:</strong> Shift + Left Arrow</li>
 *   <li><strong>shift-right:</strong> Shift + Right Arrow</li>
 * </ul>
 *
 * <p>Example of registering a custom handler:</p>
 * <pre>{@code
 * ConsolePanelSink sink = ConsolePanelSink.builder()
 *     .withKeyHandler("shift-right", () -> {
 *         clock.speedUp();
 *         sink.addLogMessage("Time speed increased");
 *     })
 *     .withKeyHandler("shift-left", () -> {
 *         clock.slowDown();
 *         sink.addLogMessage("Time speed decreased");
 *     })
 *     .build();
 * }</pre>
 *
 * <h2>Thread Safety:</h2>
 * <p>This class is thread-safe. Status updates can be received from multiple threads
 * concurrently. A dedicated render thread handles all terminal I/O and keyboard input
 * processing. Custom keyboard handlers are executed synchronously on the render thread,
 * so they should complete quickly to avoid blocking the UI.</p>
 *
 * @see StatusSink
 * @see StatusTracker
 * @see Builder
 * @since 4.0.0
 */
public class ConsolePanelSink implements StatusSink, AutoCloseable {

    private static final Logger logger = LogManager.getLogger(ConsolePanelSink.class);

    private Terminal terminal;
    private Display display;
    private final Thread renderThread;
    private final Map<StatusTracker<?>, TaskNode> taskNodes;
    private final Map<TrackerScope, ScopeNode> scopeNodes;
    private final DisplayNode rootNode;
    private final DateTimeFormatter timeFormatter;
    private final long refreshRateMs;
    private final long completedRetentionMs;
    private final boolean useColors;
    private final boolean autoExit;
    private final AtomicBoolean closed;
    private final AtomicBoolean shouldRender;
    private final AtomicBoolean introComplete; // Flag to prevent rendering during intro

    // Custom keyboard handlers
    private final Map<String, Runnable> customKeyHandlers;

    // Logging panel components
    private final LinkedList<String> logBuffer;  // Simple linked list for efficient head/tail operations
    private final int maxLogLines;
    private volatile int logScrollOffset;
    private volatile int taskScrollOffset;
    private volatile int splitOffset;  // Controls split between task panel and log panel
    private volatile boolean isUserScrollingLogs = false;  // Track if user is manually scrolling
    private volatile long lastLogDisplayTime = 0;
    private final ReentrantReadWriteLock logLock;
    private final PrintStream originalOut;
    private final PrintStream originalErr;
    private final LogCapturePrintStream capturedOut;
    private final LogCapturePrintStream capturedErr;
    private volatile int lastTaskContentHeight = 10;
    private volatile int lastLogContentHeight = 5;
    private volatile List<String> lastRenderSnapshot = Collections.emptyList();

    // Double-tap 'q' to exit tracking
    private volatile long lastQPressTime = 0;
    private static final long DOUBLE_TAP_WINDOW_MS = 300; // 300ms window for double-tap
    private volatile boolean autoExitEnabled = false; // Track current auto-exit state

    // Help panel state
    private volatile boolean showingHelp = false;

    // Search state
    private enum SearchMode { NONE, EDITING, NAVIGATING }
    private volatile SearchMode searchMode = SearchMode.NONE;
    private volatile String searchPattern = "";
    private volatile List<Integer> searchMatches = new ArrayList<>();
    private volatile int currentSearchIndex = 0;
    private volatile String searchError = null;

    // Log level filtering
    private static final org.apache.logging.log4j.Level[] LOG_LEVELS = {
        org.apache.logging.log4j.Level.ALL,
        org.apache.logging.log4j.Level.TRACE,
        org.apache.logging.log4j.Level.DEBUG,
        org.apache.logging.log4j.Level.INFO,
        org.apache.logging.log4j.Level.WARN,
        org.apache.logging.log4j.Level.ERROR,
        org.apache.logging.log4j.Level.FATAL,
        org.apache.logging.log4j.Level.OFF
    };


    // ANSI color codes for different states
    private static final AttributedStyle STYLE_PENDING = AttributedStyle.DEFAULT.foreground(AttributedStyle.WHITE);
    private static final AttributedStyle STYLE_RUNNING = AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN);
    private static final AttributedStyle STYLE_SUCCESS = AttributedStyle.DEFAULT.foreground(AttributedStyle.GREEN);
    private static final AttributedStyle STYLE_FAILED = AttributedStyle.DEFAULT.foreground(AttributedStyle.RED);
    private static final AttributedStyle STYLE_HEADER = AttributedStyle.DEFAULT.bold();
    private static final AttributedStyle STYLE_LOG_INFO = AttributedStyle.DEFAULT.foreground(AttributedStyle.WHITE);
    private static final AttributedStyle STYLE_LOG_WARN = AttributedStyle.DEFAULT.foreground(AttributedStyle.YELLOW);
    private static final AttributedStyle STYLE_LOG_ERROR = AttributedStyle.DEFAULT.foreground(AttributedStyle.RED);
    private static final AttributedStyle STYLE_LOG_DEBUG = AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT | AttributedStyle.BLACK);
    private static final AttributedStyle STYLE_SECONDARY = AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT | AttributedStyle.CYAN);
    private static final AttributedStyle STYLE_BORDER = AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.BRIGHT | AttributedStyle.CYAN);
    private static final AttributedStyle STYLE_BORDER_TITLE = AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.YELLOW);
    private static final DateTimeFormatter SCREENSHOT_FORMAT = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    private ConsolePanelSink(Builder builder) {
        try {
            this.terminal = TerminalBuilder.builder()
                    .system(true)
                    .jansi(true)
                    .jna(true)  // Enable JNA for better terminal support
                    .color(builder.useColors)  // Explicitly set color support
                    .build();

            // Enter raw mode to capture single keystrokes without waiting for Enter
            terminal.enterRawMode();

            // Create display with fullscreen mode enabled for proper rendering
            this.display = new Display(terminal, true);

            // Resize display to current terminal size
            Size initialSize = terminal.getSize();
            if (initialSize == null || initialSize.getRows() <= 0 || initialSize.getColumns() <= 0) {
                initialSize = new Size(100, 40);
            }
            display.resize(initialSize.getRows(), initialSize.getColumns());

            // Initialize the display by clearing and setting up the screen
            try {
                terminal.puts(org.jline.utils.InfoCmp.Capability.clear_screen);
                terminal.flush();
            } catch (Exception e) {
                logger.warn("Could not clear screen: {}", e.getMessage());
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize JLine terminal: " + e.getMessage(), e);
        }

        this.refreshRateMs = builder.refreshRateMs;
        this.completedRetentionMs = builder.completedRetentionMs;
        this.useColors = builder.useColors;
        this.autoExit = builder.autoExit;
        this.autoExitEnabled = builder.autoExit; // Initialize from builder
        this.maxLogLines = builder.maxLogLines;
        this.customKeyHandlers = new ConcurrentHashMap<>(builder.customKeyHandlers);
        this.taskNodes = new ConcurrentHashMap<>();
        this.scopeNodes = new ConcurrentHashMap<>();
        this.rootNode = new RootNode();
        this.timeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        this.closed = new AtomicBoolean(false);
        this.shouldRender = new AtomicBoolean(true);
        this.introComplete = new AtomicBoolean(true); // Default to true, set to false by showIntroScreen()

        // Initialize logging components
        this.logBuffer = new LinkedList<>();
        this.logScrollOffset = 0;
        this.splitOffset = 0;  // Start with default split
        this.isUserScrollingLogs = false;
        this.lastLogDisplayTime = 0;
        this.logLock = new ReentrantReadWriteLock();

        // Capture System.out and System.err
        this.originalOut = System.out;
        this.originalErr = System.err;
        this.capturedOut = new LogCapturePrintStream("OUT");
        this.capturedErr = new LogCapturePrintStream("ERR");

        // Redirect console output only if requested
        if (builder.captureSystemStreams) {
            System.setOut(capturedOut);
            System.setErr(capturedErr);
        }

        // Create and start the dedicated render thread
        this.renderThread = new Thread(this::renderLoop, "ConsolePanelSink-Renderer");
        this.renderThread.setDaemon(true);
        this.renderThread.start();

        // Add shutdown hook to properly clean up terminal on exit
        Runtime.getRuntime().addShutdownHook(new Thread(new ShutdownCleanupRunnable(this), "ConsolePanelSink-Shutdown"));

        // Register with LogBuffer to receive log messages
        LogBuffer.setActiveSink(this);

        // Force an immediate full frame render to initialize the layout
        try {
            Thread.sleep(50); // Brief pause to let thread start
            // Do a direct refresh call to trigger immediate render
            refresh();
            Thread.sleep(50); // Give time for the initial render to complete
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void renderLoop() {
        // Render loop with non-blocking input handling
        NonBlockingReader reader = null;
        try {
            // Set up non-blocking reader for keyboard input
            reader = terminal.reader();

            long lastRenderTime = System.currentTimeMillis();
            long lastCleanupTime = System.currentTimeMillis();

            // Log that render loop has started
            logger.debug("Render loop started with non-blocking input");

            while (!closed.get()) {
                long now = System.currentTimeMillis();

                // Check for keyboard input (non-blocking)
                try {
                    int c = reader.read(1); // Non-blocking read with 1ms timeout
                    if (c != -2 && c != -1) { // -2 means no input available, -1 means EOF
                        handleInput(reader, c);
                    }
                } catch (IOException e) {
                    // Ignore read errors to prevent interrupting the render loop
                }

                // Clean up completed tasks periodically
                if (now - lastCleanupTime >= 1000) { // Check every second
                    cleanupCompletedTasks(now);
                    lastCleanupTime = now;

                    // Check for auto-exit condition
                    if (autoExitEnabled && isEverythingComplete()) {
                        addLogMessage("Auto-exit: All work complete - shutting down...");
                        performExit();
                        break;
                    }
                }

                // Render at specified refresh rate
                if (now - lastRenderTime >= refreshRateMs) {
                    refresh();
                    lastRenderTime = now;
                }

                // Small sleep to prevent CPU spinning
                if (!closed.get()) {
                    Thread.sleep(10);
                }
            }
        } catch (Exception e) {
            if (!closed.get()) {
                logger.error("Render loop error", e);
            }
        } finally {
            // Clean up reader
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    // Ignore
                }
            }
            logger.debug("Render loop exited");
        }
    }

    private boolean isEverythingComplete() {
        // Check if root node is complete (which recursively checks all scopes and tasks)
        return rootNode.isComplete();
    }

    private void cleanupCompletedTasks(long now) {
        // Remove completed tasks and scopes based on parent completion status
        List<Map.Entry<StatusTracker<?>, TaskNode>> tasksToRemove = new ArrayList<>();
        List<Map.Entry<TrackerScope, ScopeNode>> scopesToRemove = new ArrayList<>();

        // Cleanup tasks
        for (Map.Entry<StatusTracker<?>, TaskNode> entry : taskNodes.entrySet()) {
            TaskNode node = entry.getValue();
            if (node.finishTime > 0) {
                // Check if this completed task should be removed
                if (node.parent == null || node.parent instanceof RootNode) {
                    // Root-level task - use standard retention time
                    if ((now - node.finishTime) > completedRetentionMs) {
                        tasksToRemove.add(entry);
                    }
                } else if (node.parent.isComplete()) {
                    // Parent (scope or task) is also completed - remove child after brief delay
                    // This ensures the final state is visible before cleanup
                    if ((now - node.finishTime) > 1000) {  // 1 second minimum visibility
                        tasksToRemove.add(entry);
                    }
                }
                // If parent is still running, keep this completed child visible
            }
        }

        // Cleanup scopes
        for (Map.Entry<TrackerScope, ScopeNode> entry : scopeNodes.entrySet()) {
            ScopeNode node = entry.getValue();
            if (node.finishTime > 0) {
                // Check if this completed scope should be removed
                if (node.parent == null || node.parent instanceof RootNode) {
                    // Root-level scope - use standard retention time
                    if ((now - node.finishTime) > completedRetentionMs) {
                        scopesToRemove.add(entry);
                    }
                } else if (node.parent.isComplete()) {
                    // Parent scope is also completed - remove child after brief delay
                    if ((now - node.finishTime) > 1000) {  // 1 second minimum visibility
                        scopesToRemove.add(entry);
                    }
                }
            }
        }

        // Execute removals
        for (Map.Entry<StatusTracker<?>, TaskNode> entry : tasksToRemove) {
            TaskNode node = entry.getValue();
            taskNodes.remove(entry.getKey());
            if (node.parent != null) {
                node.parent.children.remove(node);
            }
        }

        for (Map.Entry<TrackerScope, ScopeNode> entry : scopesToRemove) {
            ScopeNode node = entry.getValue();
            scopeNodes.remove(entry.getKey());
            if (node.parent != null) {
                node.parent.children.remove(node);
            }
        }
    }


    /**
     * Add a log message to the display buffer.
     * This is called by LogBuffer to add logging framework messages.
     * Only sink methods should mutate the logBuffer.
     */
    /**
     * Adds a message to the log buffer. The message will be timestamped and displayed
     * in the console output panel.
     *
     * @param message the message to add
     */
    public void addLogMessage(String message) {
        if (message == null || message.trim().isEmpty()) {
            return;
        }

        logLock.writeLock().lock();
        try {
            // Add timestamp if not present
            if (!message.matches("^\\[\\d{2}:\\d{2}:\\d{2}\\].*")) {
                message = "[" + LocalDateTime.now().format(timeFormatter) + "] " + message;
            }

            logBuffer.addLast(message);

            // Limit buffer size to maxLogLines (default 1000)
            while (logBuffer.size() > maxLogLines) {
                logBuffer.removeFirst();
                // When manually scrolling, we need to adjust offset to maintain the same view
                // Even if offset is 0, removing from front means we need to "scroll up" to stay in place
                // However, we can't have negative offset, so content will shift if at the very top
                if (isUserScrollingLogs && logScrollOffset > 0) {
                    logScrollOffset--;
                } else if (!isUserScrollingLogs) {
                    // When auto-scrolling, decrement to maintain bottom view
                    if (logScrollOffset > 0) {
                        logScrollOffset--;
                    }
                }
            }

            // Auto-scroll to latest if not manually scrolling
            if (!isUserScrollingLogs) {
                int maxScroll = Math.max(0, logBuffer.size() - getLogPanelHeight());
                logScrollOffset = maxScroll;
            }
        } finally {
            logLock.writeLock().unlock();
        }
    }
    
    private void handleSearchInput(NonBlockingReader reader, int c) {
        if (searchMode == SearchMode.EDITING) {
            // Enter - switch to navigation mode
            if (c == '\n' || c == '\r') {
                if (searchPattern.length() >= 2 && !searchMatches.isEmpty()) {
                    searchMode = SearchMode.NAVIGATING;
                    currentSearchIndex = 0;
                    jumpToMatch(0);
                } else if (searchPattern.length() < 2) {
                    searchError = "Pattern too short (min 2 chars)";
                } else {
                    searchError = "No matches found";
                }
                return;
            }

            // Handle [ and ] for log level during search (same as normal mode)
            if (c == '[') {
                cycleLogLevel(false);
                updateSearchPreview();
                return;
            }

            if (c == ']') {
                cycleLogLevel(true);
                updateSearchPreview();
                return;
            }

            // ESC - cancel search
            if (c == 27) {
                searchMode = SearchMode.NONE;
                searchPattern = "";
                searchMatches.clear();
                searchError = null;
                return;
            }

            // Backspace
            if (c == 127 || c == '\b') {
                if (searchPattern.length() > 0) {
                    searchPattern = searchPattern.substring(0, searchPattern.length() - 1);
                    updateSearchPreview();
                }
                return;
            }

            // Add character to search pattern (all printable ASCII allowed)
            if (c >= 32 && c < 127) { // Printable ASCII
                searchPattern += (char) c;
                updateSearchPreview();
            }
        } else if (searchMode == SearchMode.NAVIGATING) {
            // ESC - exit navigation mode
            if (c == 27) {
                searchMode = SearchMode.NONE;
                searchPattern = "";
                searchMatches.clear();
                searchError = null;
                return;
            }

            // n - next match
            if (c == 'n') {
                currentSearchIndex = (currentSearchIndex + 1) % searchMatches.size();
                jumpToMatch(currentSearchIndex);
                return;
            }

            // p or N - previous match
            if (c == 'p' || c == 'N') {
                currentSearchIndex = (currentSearchIndex - 1 + searchMatches.size()) % searchMatches.size();
                jumpToMatch(currentSearchIndex);
                return;
            }
        }
    }

    private void updateSearchPreview() {
        searchError = null;
        searchMatches.clear();

        if (searchPattern.length() < 2) {
            return;
        }

        try {
            java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(searchPattern);

            // Search through log buffer
            for (int i = 0; i < logBuffer.size(); i++) {
                String line = logBuffer.get(i);
                if (pattern.matcher(line).find()) {
                    searchMatches.add(i);
                }
            }
        } catch (java.util.regex.PatternSyntaxException e) {
            searchError = e.getMessage();
            searchMatches.clear();
        }
    }

    private void jumpToMatch(int matchIndex) {
        if (matchIndex >= 0 && matchIndex < searchMatches.size()) {
            int matchLine = searchMatches.get(matchIndex);
            logScrollOffset = Math.max(0, matchLine - getLogPanelHeight() / 2);
            isUserScrollingLogs = true;
        }
    }

    private void performExit() {
        // Signal to close first
        if (!closed.compareAndSet(false, true)) {
            return; // Already closing
        }

        // Clear the display and show exit message
        try {
            List<AttributedString> exitMessage = new ArrayList<>();
            exitMessage.add(new AttributedStringBuilder()
                .style(AttributedStyle.DEFAULT.foreground(AttributedStyle.YELLOW))
                .append("Console panel shutting down...")
                .toAttributedString());
            exitMessage.add(new AttributedStringBuilder()
                .append("Restoring normal terminal...")
                .toAttributedString());
            display.update(exitMessage, exitMessage.size());
            terminal.flush();
        } catch (Exception e) {
            // Ignore display errors during shutdown
        }

        // Perform actual cleanup
        performCleanup();
    }

    private void handleInput(NonBlockingReader reader, int c) {
        logLock.writeLock().lock();
        try {
            // Handle [ and ] for log level
            // [ = decrease verbosity (less verbose, higher level)
            // ] = increase verbosity (more verbose, lower level)
            if (c == '[') {
                cycleLogLevel(false);
                return;
            }

            if (c == ']') {
                cycleLogLevel(true);
                return;
            }

            // If showing help, any key closes it
            if (showingHelp) {
                showingHelp = false;
                return;
            }

            // Handle search mode
            if (searchMode != SearchMode.NONE) {
                handleSearchInput(reader, c);
                return;
            }

            // Handle help command
            if (c == '?') {
                showingHelp = true;
                return;
            }

            // Handle search command
            if (c == '/') {
                searchMode = SearchMode.EDITING;
                searchPattern = "";
                searchMatches.clear();
                currentSearchIndex = 0;
                searchError = null;
                return;
            }

            // Handle quit command
            if (c == 'q' || c == 'Q') {
                handleQuitCommand();
                return;
            }


            if (c == 's' || c == 'S') {
                saveScreenshot();
                return;
            }

            // Handle arrow keys and special sequences (including numpad +/-)
            if (c == 27) { // ESC sequence
                try {
                    int next = reader.read(10);

                    // If no follow-up character, treat ESC as standalone quit command (like 'q')
                    if (next == -2) { // timeout, standalone ESC
                        handleQuitCommand();
                        return;
                    }

                    if (next == 'O') {
                        // ESC O sequences - just handle special keys, removed numpad +/- confusion
                        int key = reader.read(10);
                        // Fall through to regular handling
                    }

                    // For ESC [ sequences
                    if (next == '[') {
                        int key = reader.read(10);

                        switch (key) {
                            case 'A': // Up arrow - scroll logs up (show older content)
                                // Enable manual scrolling FIRST to prevent auto-scroll from interfering
                                isUserScrollingLogs = true;
                                int maxScrollRangeUp = Math.max(0, logBuffer.size() - getLogPanelHeight());
                                // Clamp current offset to valid range
                                logScrollOffset = Math.max(0, Math.min(logScrollOffset, maxScrollRangeUp));
                                // Then scroll up if possible
                                if (logScrollOffset > 0) {
                                    logScrollOffset--;
                                }
                                break;
                            case 'B': // Down arrow - scroll logs down (show newer content)
                                int maxLogScroll = Math.max(0, logBuffer.size() - getLogPanelHeight());
                                // First clamp current offset to valid range
                                logScrollOffset = Math.max(0, Math.min(logScrollOffset, maxLogScroll));
                                // Then scroll down if possible
                                if (logScrollOffset < maxLogScroll) {
                                    logScrollOffset++;
                                    isUserScrollingLogs = true;
                                } else {
                                    isUserScrollingLogs = false; // At bottom, resume auto-follow
                                }
                                break;
                            case '1': // ESC[1;2A or ESC[1;2B - Shift+arrows
                                int next2 = reader.read(10);
                                if (next2 == ';') {
                                    int modifier = reader.read(10);
                                    if (modifier == '2') { // Shift modifier
                                        int direction = reader.read(10);
                                        if (direction == 'A' || direction == 'B') {
                                            // Shift+Up/Down - page up/down logs
                                            int pageSize = getLogPanelHeight() - 1;
                                            int maxScroll = Math.max(0, logBuffer.size() - getLogPanelHeight());
                                            // First clamp current offset
                                            logScrollOffset = Math.max(0, Math.min(logScrollOffset, maxScroll));
                                            if (direction == 'A') { // Shift+Up - page up
                                                if (logScrollOffset > 0) {
                                                    logScrollOffset = Math.max(0, logScrollOffset - pageSize);
                                                    isUserScrollingLogs = true;
                                                }
                                            } else if (direction == 'B') { // Shift+Down - page down
                                                if (logScrollOffset < maxScroll) {
                                                    logScrollOffset = Math.min(logScrollOffset + pageSize, maxScroll);
                                                    isUserScrollingLogs = logScrollOffset < maxScroll;
                                                }
                                            }
                                        } else if (direction == 'C') { // Shift+Right
                                            Runnable handler = customKeyHandlers.get("shift-right");
                                            if (handler != null) {
                                                handler.run();
                                            }
                                        } else if (direction == 'D') { // Shift+Left
                                            Runnable handler = customKeyHandlers.get("shift-left");
                                            if (handler != null) {
                                                handler.run();
                                            }
                                        }
                                    }
                                }
                                break;
                            case '5': // Page Up (ESC[5~) - Scroll logs up by page
                                if (reader.read(10) == '~') { // consume ~
                                    int pageSize = getLogPanelHeight() - 1;
                                    int maxScroll = Math.max(0, logBuffer.size() - getLogPanelHeight());
                                    logScrollOffset = Math.max(0, Math.min(logScrollOffset, maxScroll));
                                    if (logScrollOffset > 0) {
                                        logScrollOffset = Math.max(0, logScrollOffset - pageSize);
                                        isUserScrollingLogs = true;
                                    }
                                }
                                break;
                            case '6': // Page Down (ESC[6~) - Scroll logs down by page
                                if (reader.read(10) == '~') { // consume ~
                                    int pageSize = getLogPanelHeight() - 1;
                                    int maxScroll = Math.max(0, logBuffer.size() - getLogPanelHeight());
                                    logScrollOffset = Math.max(0, Math.min(logScrollOffset, maxScroll));
                                    if (logScrollOffset < maxScroll) {
                                        logScrollOffset = Math.min(logScrollOffset + pageSize, maxScroll);
                                        isUserScrollingLogs = logScrollOffset < maxScroll;
                                    }
                                }
                                break;
                            case 'H': // Home
                                logScrollOffset = 0;
                                taskScrollOffset = 0;
                                splitOffset = 0; // Reset split to default
                                break;
                            case 'F': // End
                                logScrollOffset = Math.max(0, logBuffer.size() - getLogPanelHeight());
                                taskScrollOffset = Math.max(0, taskNodes.size() - 10);
                                break;
                        }
                    }
                } catch (IOException e) {
                    // Ignore
                }
            }
        } finally {
            logLock.writeLock().unlock();
        }
    }

    /**
     * Called when a scope is created. Adds it to the display hierarchy.
     */
    @Override
    public void scopeStarted(TrackerScope scope) {
        if (closed.get()) return;

        // Find parent scope or use root
        TrackerScope parentScope = scope.getParent();
        DisplayNode parent = parentScope != null ? scopeNodes.get(parentScope) : rootNode;

        ScopeNode node = new ScopeNode(scope, parent);
        scopeNodes.put(scope, node);

        if (parent != null) {
            parent.children.add(node);
        }
    }

    /**
     * Called when a scope is closed. Marks it as finished but keeps it visible for retention period.
     */
    @Override
    public void scopeFinished(TrackerScope scope) {
        if (closed.get()) return;

        ScopeNode node = scopeNodes.get(scope);
        if (node != null) {
            node.finishTime = System.currentTimeMillis();
            // Don't remove immediately - let cleanup handle it based on retention time
        }
    }

    @Override
    public void taskStarted(StatusTracker<?> task) {
        if (closed.get()) return;

        // Find parent: either a scope or root
        TrackerScope parentScope = task.getParentScope();
        DisplayNode parent = parentScope != null ? scopeNodes.get(parentScope) : rootNode;

        TaskNode node = new TaskNode(task, parent);
        taskNodes.put(task, node);

        if (parent != null) {
            parent.children.add(node);
        }
        // Note: Render thread will pick up this change at next refresh cycle
    }

    /**
     * Handles quit command logic - supports both 'q' and ESC keys.
     * Implements double-tap detection and auto-exit toggle behavior.
     */
    private void handleQuitCommand() {
        long now = System.currentTimeMillis();
        boolean isDoubleTap = (now - lastQPressTime) < DOUBLE_TAP_WINDOW_MS;
        lastQPressTime = now;

        // Check if all work is complete (scopes and tasks)
        boolean allTasksFinished = isEverythingComplete();

        // Double-tap always exits
        if (isDoubleTap) {
            addLogMessage("Double-tap detected - shutting down console panel...");
            performExit();
            return;
        }

        // If auto-exit is enabled and all tasks are finished, exit immediately
        if (autoExitEnabled && allTasksFinished) {
            addLogMessage("All tasks finished - shutting down console panel...");
            performExit();
            return;
        }

        // If auto-exit is enabled but tasks are still running, just disable it
        if (autoExitEnabled) {
            autoExitEnabled = false;
            addLogMessage("Auto-exit disabled. Press 'q'/ESC again within 300ms to exit.");
            return;
        }

        // If auto-exit is disabled, enable it
        if (!autoExitEnabled) {
            autoExitEnabled = true;
            if (allTasksFinished) {
                addLogMessage("Auto-exit enabled - all work complete, exiting now...");
                performExit();
            } else {
                addLogMessage("Auto-exit enabled. Will exit when all work completes.");
            }
        }
    }

    @Override
    public void taskUpdate(StatusTracker<?> task, StatusUpdate<?> status) {
        if (closed.get()) return;

        // Buffer the update - no rendering here, just update data structures
        TaskNode node = taskNodes.get(task);
        if (node != null) {
            node.lastStatus = status;
            node.lastUpdateTime = System.currentTimeMillis();
        }
        // Note: Render thread will pick up this change at next refresh cycle
    }

    @Override
    public void taskFinished(StatusTracker<?> task) {
        if (closed.get()) return;

        // Buffer the update - no rendering here, just update data structures
        TaskNode node = taskNodes.get(task);
        if (node != null) {
            node.finishTime = System.currentTimeMillis();
            // Get the final status to ensure we show 100% for SUCCESS tasks
            StatusUpdate<?> finalStatus = task.getStatus();
            if (finalStatus != null) {
                // Ensure completed tasks show 100% progress if they succeeded
                if (finalStatus.runstate == RunState.SUCCESS) {
                    node.lastStatus = new StatusUpdate<>(1.0, RunState.SUCCESS, finalStatus.tracked);
                } else {
                    node.lastStatus = finalStatus;
                }
                node.lastUpdateTime = System.currentTimeMillis();
            }
            // Note: Completed tasks will be cleaned up by the render thread
            // based on completedRetentionMs
        }
    }


    private static volatile int refreshCount = 0;
    private Size lastKnownSize = null;

    private void refresh() {
        if (closed.get() || !introComplete.get()) return;

        try {
            refreshCount++;

//            // Debug: log refresh attempts
//            if (refreshCount == 1 || refreshCount % 50 == 0) {
//                System.err.println("[ConsolePanelSink] Refresh #" + refreshCount + " starting");
//            }

            // Get terminal size
            Size size = terminal.getSize();

            // If terminal size is invalid, use a reasonable default
            if (size == null || size.getRows() <= 0 || size.getColumns() <= 0) {
                if (refreshCount == 1) {  // Only log once on first refresh
                    if (size != null) {
                        logger.info("Terminal size detection failed (got {}x{}). Using default size 100x40. " +
                            "This may happen when running in certain IDEs or piped environments.",
                            size.getColumns(), size.getRows());
                    } else {
                        logger.info("Terminal size is null. Using default size 100x40.");
                    }
                }
                size = new Size(100, 40);  // More reasonable default for modern terminals
            }

            // Detect terminal resize
            boolean terminalResized = false;
            if (lastKnownSize != null &&
                (lastKnownSize.getRows() != size.getRows() || lastKnownSize.getColumns() != size.getColumns())) {
                terminalResized = true;

                // Force a complete redraw on resize
                try {
                    // Clear the entire screen
                    terminal.puts(org.jline.utils.InfoCmp.Capability.clear_screen);
                    terminal.puts(org.jline.utils.InfoCmp.Capability.cursor_home);
                    terminal.flush();

                    // Reset the display to force full redraw
                    display.clear();
                    display.resize(size.getRows(), size.getColumns());

                    // Reset display state to force complete refresh
                    display.update(Collections.emptyList(), 0);

                } catch (Exception e) {
                    // If clear fails, try alternative approach
                    try {
                        terminal.writer().print("\033[2J\033[H"); // ANSI clear screen and home
                        terminal.flush();
                    } catch (Exception ignored) {}
                }
            }
            lastKnownSize = size;

            // Only enable debug logging if explicitly requested
            boolean debugThisRefresh = false; // (refreshCount % 100 == 1);  // Uncomment for debugging

            List<AttributedString> lines = new ArrayList<>();

            // Calculate layout dimensions - minimize border overhead
            // Base: 1 top border, 1 middle divider, 1 bottom border, 1 status bar = 4 total overhead lines
            // Add 2 more lines if search panel is active (border + content, reuses middle divider)
            int totalOverhead = 4;
            int searchPanelHeight = 0;
            if (searchMode != SearchMode.NONE) {
                searchPanelHeight = 2; // Search panel: border + content line
                totalOverhead += searchPanelHeight;
            }
            int availableContent = Math.max(2, size.getRows() - totalOverhead);

            // Split available content between task and log panels (2/3 for tasks, 1/3 for logs)
            int baseTaskContent = Math.max(1, (availableContent * 2) / 3);
            int taskContentHeight = Math.max(1, Math.min(availableContent - 1, baseTaskContent + splitOffset));
            int logContentHeight = Math.max(1, availableContent - taskContentHeight);

            lastTaskContentHeight = taskContentHeight;
            lastLogContentHeight = logContentHeight;

            // Build the display with minimal borders
            if (showingHelp) {
                renderHelpPanel(lines, size.getColumns(), size.getRows());
            } else {
                renderCompactHeader(lines, size.getColumns());
                renderTaskContent(lines, size.getColumns(), taskContentHeight);

                // Insert search panel if active
                if (searchMode != SearchMode.NONE) {
                    renderSearchPanel(lines, size.getColumns());
                }

                renderMiddleDivider(lines, size.getColumns());
                renderLogContent(lines, size.getColumns(), logContentHeight);
                renderBottomBar(lines, size.getColumns());
            }

            // Ensure we don't exceed terminal height
            while (lines.size() > size.getRows()) {
                lines.remove(lines.size() - 1);
            }

            // Pad to fill screen
            while (lines.size() < size.getRows()) {
                lines.add(new AttributedString(""));
            }

            // Debug: Check what we're about to render
            if (debugThisRefresh) {
                logger.debug("About to update display with {} lines", lines.size());
                if (!lines.isEmpty()) {
                    String firstLine = lines.get(0).toAnsi(terminal);
                    logger.debug("First line content (len={}): {}", firstLine.length(),
                        firstLine.length() > 50 ? firstLine.substring(0, 50) + "..." : firstLine);
                }
            }

            // Update display - Display class handles differential updates
            // Position cursor at bottom right to hide it
            display.update(lines, size.cursorPos(size.getRows() - 1, size.getColumns() - 1));
            terminal.flush();

            List<String> snapshot = new ArrayList<>(lines.size());
            for (AttributedString line : lines) {
                snapshot.add(line.toString());
            }
            lastRenderSnapshot = snapshot;

            if (debugThisRefresh) {
                logger.debug("Display update completed");
            }

        } catch (Exception e) {
            // Log error but continue
            logger.error("Display update error: {}", e.getMessage(), e);
        }
    }


    // New compact rendering methods
    private void renderCompactHeader(List<AttributedString> lines, int width) {
        // Single top border line with title
        lines.add(buildSectionBorder('╔', '╗', "Active Tasks", width));
    }

    private void renderSearchPanel(List<AttributedString> lines, int width) {
        // Top border
        lines.add(buildSectionBorder('╠', '╣', "Search", width));

        // Search content line
        AttributedStringBuilder content = new AttributedStringBuilder();

        if (searchMode == SearchMode.EDITING) {
            // Editing mode - show pattern input
            content.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN))
                   .append("Pattern: ");
            content.style(AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.YELLOW))
                   .append(searchPattern.isEmpty() ? "_" : searchPattern);

            // Show match count or error
            if (searchError != null) {
                content.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.RED))
                       .append(" [ERROR: " + searchError + "]");
            } else if (searchPattern.length() >= 2) {
                content.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.GREEN))
                       .append(" [" + searchMatches.size() + " matches]");
            } else {
                content.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT | AttributedStyle.BLACK))
                       .append(" [min 2 chars]");
            }

            content.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN))
                   .append(" | Enter: navigate | ESC: cancel");
        } else if (searchMode == SearchMode.NAVIGATING) {
            // Navigation mode - show current match
            content.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN))
                   .append("Searching: ");
            content.style(AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.YELLOW))
                   .append(searchPattern);
            content.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.GREEN))
                   .append(String.format(" | Match %d/%d", currentSearchIndex + 1, searchMatches.size()));
            content.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN))
                   .append(" | n: next | p: prev | ESC: exit");
        }

        lines.add(wrapWithSideBorders(content.toAttributedString(), width));
        // Note: No bottom border - renderMiddleDivider will provide the divider to Console Output
    }

    private void renderMiddleDivider(List<AttributedString> lines, int width) {
        // Single divider line between panels with title
        lines.add(buildSectionBorder('╠', '╣', "Console Output", width));
    }

    private void renderBottomBar(List<AttributedString> lines, int width) {
        // Bottom border
        lines.add(buildSectionBorder('╚', '╝', null, width));

        // Status bar (kept separate for information display)
        renderStatusLine(lines, width);
    }

    private void renderTaskContent(List<AttributedString> lines, int width, int contentHeight) {
        // Render task lines without additional borders
        List<AttributedString> taskLines = collectTaskContentLines(width - 4, contentHeight);
        for (AttributedString line : taskLines) {
            lines.add(wrapWithSideBorders(line, width));
        }
    }

    private void renderLogContent(List<AttributedString> lines, int width, int contentHeight) {
        // Render log lines without additional borders
        List<AttributedString> logLines = collectLogContentLines(width - 4, contentHeight);
        for (AttributedString line : logLines) {
            lines.add(wrapWithSideBorders(line, width));
        }
    }

    private AttributedString wrapWithSideBorders(AttributedString content, int width) {
        AttributedStringBuilder builder = new AttributedStringBuilder();
        builder.style(STYLE_BORDER).append("║ ");
        builder.append(content);

        // Calculate padding
        int contentLength = content.columnLength();
        int paddingNeeded = Math.max(0, width - contentLength - 4); // 4 for "║ " and " ║"
        builder.append(" ".repeat(paddingNeeded));

        builder.style(STYLE_BORDER).append(" ║");
        return builder.toAttributedString();
    }

    private List<AttributedString> collectTaskContentLines(int innerWidth, int contentHeight) {
        List<AttributedString> result = new ArrayList<>();

        // Collect task entries
        List<SectionLine> taskEntries = new ArrayList<>();
        collectTaskLines(rootNode, taskEntries, "", true, innerWidth);

        if (taskEntries.isEmpty()) {
            result.add(new AttributedString(center("No active tasks", innerWidth)));
        } else {
            // Add visible task lines
            int startIdx = Math.min(taskScrollOffset, Math.max(0, taskEntries.size() - contentHeight));
            int endIdx = Math.min(taskEntries.size(), startIdx + contentHeight);

            for (int i = startIdx; i < endIdx; i++) {
                SectionLine line = taskEntries.get(i);
                if (line.style != null) {
                    result.add(new AttributedStringBuilder().style(line.style).append(line.text).toAttributedString());
                } else {
                    result.add(new AttributedString(line.text));
                }
            }
        }

        // Pad to fill height
        while (result.size() < contentHeight) {
            result.add(new AttributedString(""));
        }

        return result;
    }

    private List<AttributedString> collectLogContentLines(int innerWidth, int contentHeight) {
        List<AttributedString> result = new ArrayList<>();

        logLock.readLock().lock();
        try {
            int totalLogs = logBuffer.size();

            if (totalLogs > 0) {
                // Calculate starting position for display
                // When not scrolling, show the bottom (most recent) contentHeight lines
                // When scrolling, logScrollOffset is the start index (0 = top of buffer)
                int startIdx;
                if (isUserScrollingLogs) {
                    // logScrollOffset is the start index - clamp to valid range
                    int maxStartIdx = Math.max(0, totalLogs - contentHeight);
                    startIdx = Math.max(0, Math.min(logScrollOffset, maxStartIdx));
                } else {
                    // Show the bottom (most recent) contentHeight lines
                    startIdx = Math.max(0, totalLogs - contentHeight);
                }

                // Prepare search highlighting pattern
                java.util.regex.Pattern highlightPattern = null;
                if (!searchMatches.isEmpty() && !searchPattern.isEmpty()) {
                    try {
                        highlightPattern = java.util.regex.Pattern.compile(searchPattern);
                    } catch (Exception e) {
                        // Ignore pattern errors during rendering
                    }
                }

                // Most common case: showing recent logs (at or near the end)
                // Use descending iterator and collect the needed lines
                if (startIdx >= totalLogs - contentHeight * 2) {
                    // We're close to the end, use descending iterator
                    Iterator<String> descIter = logBuffer.descendingIterator();
                    List<String> tempLines = new ArrayList<>();
                    List<Integer> tempIndices = new ArrayList<>();

                    // Skip the newest lines we don't need
                    int toSkip = totalLogs - startIdx - contentHeight;
                    for (int i = 0; i < toSkip && descIter.hasNext(); i++) {
                        descIter.next();
                    }

                    // Collect the lines we need (in reverse order)
                    int currentIdx = totalLogs - toSkip - 1;
                    for (int i = 0; i < contentHeight && descIter.hasNext(); i++) {
                        tempLines.add(descIter.next());
                        tempIndices.add(currentIdx--);
                    }

                    // Reverse to get correct order and process
                    Collections.reverse(tempLines);
                    Collections.reverse(tempIndices);
                    for (int i = 0; i < tempLines.size(); i++) {
                        String line = tempLines.get(i);
                        int lineIdx = tempIndices.get(i);
                        result.add(formatLogLine(line, innerWidth, lineIdx, highlightPattern));
                    }
                } else {
                    // We're closer to the start, use forward iterator
                    Iterator<String> iter = logBuffer.iterator();

                    // Skip to start position
                    for (int i = 0; i < startIdx && iter.hasNext(); i++) {
                        iter.next();
                    }

                    // Collect the lines we need
                    for (int i = 0; i < contentHeight && iter.hasNext(); i++) {
                        String line = iter.next();
                        int lineIdx = startIdx + i;
                        result.add(formatLogLine(line, innerWidth, lineIdx, highlightPattern));
                    }
                }
            }
        } finally {
            logLock.readLock().unlock();
        }

        // Pad to fill height
        while (result.size() < contentHeight) {
            result.add(new AttributedString(""));
        }

        return result;
    }

    private AttributedString formatLogLine(String line, int maxWidth, int lineIndex, java.util.regex.Pattern highlightPattern) {
        String logLine = fitLine(line, maxWidth);
        AttributedStyle baseStyle = getLogStyle(logLine);

        // Check if this line is a search match
        boolean isMatch = searchMatches.contains(lineIndex);
        boolean isCurrentMatch = (searchMode == SearchMode.NAVIGATING &&
                                  !searchMatches.isEmpty() &&
                                  searchMatches.get(currentSearchIndex) == lineIndex);

        if (highlightPattern != null && isMatch) {
            // Highlight matching portions
            return highlightMatches(logLine, highlightPattern, baseStyle, isCurrentMatch);
        } else if (isMatch) {
            // Just show with base style if it's a match but no pattern
            return new AttributedStringBuilder().style(baseStyle).append(logLine).toAttributedString();
        } else {
            return new AttributedStringBuilder().style(baseStyle).append(logLine).toAttributedString();
        }
    }

    private AttributedString highlightMatches(String line, java.util.regex.Pattern pattern, AttributedStyle baseStyle, boolean isCurrentMatch) {
        AttributedStringBuilder builder = new AttributedStringBuilder();
        java.util.regex.Matcher matcher = pattern.matcher(line);

        int lastEnd = 0;
        while (matcher.find()) {
            // Add text before match with base style
            if (matcher.start() > lastEnd) {
                builder.style(baseStyle).append(line.substring(lastEnd, matcher.start()));
            }

            // Add matched text with highlight
            // Current match gets green background, other matches get yellow background
            if (isCurrentMatch) {
                builder.style(AttributedStyle.DEFAULT.bold().background(AttributedStyle.GREEN).foreground(AttributedStyle.BLACK))
                       .append(line.substring(matcher.start(), matcher.end()));
            } else {
                builder.style(AttributedStyle.DEFAULT.bold().background(AttributedStyle.YELLOW).foreground(AttributedStyle.BLACK))
                       .append(line.substring(matcher.start(), matcher.end()));
            }
            lastEnd = matcher.end();
        }

        // Add remaining text
        if (lastEnd < line.length()) {
            builder.style(baseStyle).append(line.substring(lastEnd));
        }

        return builder.toAttributedString();
    }

    private void renderStatusLine(List<AttributedString> lines, int width) {
        AttributedStringBuilder statusBar = new AttributedStringBuilder();

        // Get current log level and total buffered logs
        String logLevel = getCurrentLogLevel();
        int totalBuffered = LogBuffer.getAllLogEntries().size();
        int filtered = logBuffer.size();

        // Build compact status line
        statusBar.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN))
                .append(" Logs: " + filtered + "/" + totalBuffered)
                .append(" | ")
                .append("Level: ")
                .style(AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.YELLOW))
                .append(logLevel)
                .style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN))
                .append(" (+/-)")
                .append(" | ")
                .append("Scroll: ")
                .style(!isUserScrollingLogs ?
                       AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.GREEN) :
                       AttributedStyle.DEFAULT.foreground(AttributedStyle.YELLOW))
                .append(!isUserScrollingLogs ? "AUTO" : "MANUAL")
                .style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN))
                .append(" | ");

        statusBar.append("AutoExit: ")
                .style(autoExitEnabled ?
                       AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.GREEN) :
                       AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT | AttributedStyle.BLACK))
                .append(autoExitEnabled ? "ON" : "OFF")
                .style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN))
                .append(" | ")
                .append("?: help");

        // Pad to width
        int currentLen = statusBar.toAttributedString().columnLength();
        if (currentLen < width) {
            statusBar.append(" ".repeat(width - currentLen));
        }

        lines.add(statusBar.toAttributedString());
    }

    private void renderHelpPanel(List<AttributedString> lines, int width, int height) {
        AttributedStringBuilder builder = new AttributedStringBuilder();

        // Title
        lines.add(new AttributedString(""));
        builder.style(AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.CYAN));
        builder.append(center("KEYBOARD SHORTCUTS", width));
        lines.add(builder.toAttributedString());
        lines.add(new AttributedString(""));

        // Help content
        addHelpLine(lines, "Navigation", "", width);
        addHelpLine(lines, "  ↑ / ↓", "Scroll log panel up/down", width);
        addHelpLine(lines, "  PgUp / PgDn", "Page up/down in logs", width);
        addHelpLine(lines, "  Shift+↑ / Shift+↓", "Page up/down in logs (alternate)", width);
        addHelpLine(lines, "  Home", "Jump to top of logs, reset split", width);
        addHelpLine(lines, "  End", "Jump to bottom of logs", width);
        lines.add(new AttributedString(""));

        addHelpLine(lines, "Panel Controls", "", width);
        addHelpLine(lines, "  [ / ]", "Adjust split ratio between tasks and logs", width);
        lines.add(new AttributedString(""));

        addHelpLine(lines, "Log Filtering", "", width);
        addHelpLine(lines, "  ] (not in search)", "Increase verbosity (show more logs)", width);
        addHelpLine(lines, "  [ (not in search)", "Decrease verbosity (show fewer logs)", width);
        lines.add(new AttributedString(""));

        addHelpLine(lines, "Search", "", width);
        addHelpLine(lines, "  /", "Search logs with regex pattern", width);
        addHelpLine(lines, "  n / p", "Next/previous search match", width);
        addHelpLine(lines, "  [ / ]", "Change log level (works during search)", width);
        addHelpLine(lines, "  ESC", "Cancel search", width);
        lines.add(new AttributedString(""));

        addHelpLine(lines, "Exit Controls", "", width);
        addHelpLine(lines, "  q or ESC (single)", "Toggle auto-exit on/off", width);
        addHelpLine(lines, "  q/ESC (auto-exit ON", "Exits when all tasks finish", width);
        addHelpLine(lines, "       + tasks done)", "", width);
        addHelpLine(lines, "  q+q or ESC+ESC", "Double-tap (<300ms) to force exit anytime", width);
        lines.add(new AttributedString(""));

        builder = new AttributedStringBuilder();
        builder.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN));
        builder.append("  ");
        builder.style(AttributedStyle.DEFAULT.italic());
        builder.append("Note: Auto-exit defaults to OFF. Panel stays open");
        lines.add(builder.toAttributedString());
        builder = new AttributedStringBuilder();
        builder.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN));
        builder.append("  ");
        builder.style(AttributedStyle.DEFAULT.italic());
        builder.append("after tasks finish so you can review the results.");
        lines.add(builder.toAttributedString());
        lines.add(new AttributedString(""));

        addHelpLine(lines, "Other", "", width);
        addHelpLine(lines, "  s", "Save screenshot to file", width);
        addHelpLine(lines, "  ?", "Show/hide this help", width);
        lines.add(new AttributedString(""));

        // Current status display
        builder = new AttributedStringBuilder();
        builder.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN));
        builder.append("  Current status: ");
        builder.style(AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.YELLOW));
        builder.append("Scroll=");
        builder.append(!isUserScrollingLogs ? "AUTO" : "MANUAL");
        builder.append(", AutoExit=");
        builder.style(autoExitEnabled ?
                     AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.GREEN) :
                     AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.YELLOW));
        builder.append(autoExitEnabled ? "ON" : "OFF");

        lines.add(builder.toAttributedString());
        lines.add(new AttributedString(""));

        // Bottom message
        builder = new AttributedStringBuilder();
        builder.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.YELLOW));
        builder.append(center("Press any key to close help", width));
        lines.add(builder.toAttributedString());

        // Pad to fill screen
        while (lines.size() < height) {
            lines.add(new AttributedString(""));
        }
    }

    private void addHelpLine(List<AttributedString> lines, String key, String description, int width) {
        AttributedStringBuilder builder = new AttributedStringBuilder();
        if (description.isEmpty()) {
            // Section header
            builder.style(AttributedStyle.DEFAULT.bold().foreground(AttributedStyle.YELLOW));
            builder.append("  " + key);
        } else {
            // Regular help line
            builder.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.GREEN));
            builder.append(String.format("  %-20s", key));
            builder.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.WHITE));
            builder.append(" " + description);
        }
        lines.add(builder.toAttributedString());
    }

    private String getCurrentLogLevel() {
        // Return the display level from LogBuffer
        org.apache.logging.log4j.Level displayLevel = LogBuffer.getDisplayLevel();
        return displayLevel != null ? displayLevel.name() : "INFO";
    }

    /**
     * Cycles the display log level up or down through the standard Log4j2 levels.
     * This changes the display filtering, not the appender level.
     * @param increaseVerbosity true to show more logs (lower level), false to show fewer logs (higher level)
     */
    private void cycleLogLevel(boolean increaseVerbosity) {
        try {
            org.apache.logging.log4j.Level currentLevel = LogBuffer.getDisplayLevel();

            // Find current level index
            int currentIndex = 3; // Default to INFO
            for (int i = 0; i < LOG_LEVELS.length; i++) {
                if (LOG_LEVELS[i].equals(currentLevel)) {
                    currentIndex = i;
                    break;
                }
            }

            // Calculate new index
            int newIndex;
            if (increaseVerbosity) {
                newIndex = Math.max(0, currentIndex - 1); // Move towards ALL/TRACE
            } else {
                newIndex = Math.min(LOG_LEVELS.length - 1, currentIndex + 1); // Move towards OFF
            }

            // Set new display level (this will trigger refresh)
            org.apache.logging.log4j.Level newLevel = LOG_LEVELS[newIndex];

            // Add a pre-change message
            addLogMessage("[SYSTEM] Changing display level from " + currentLevel.name() + " to " + newLevel.name() + "...");

            // This will trigger refreshDisplayBuffer() via LogBuffer.setDisplayLevel()
            LogBuffer.setDisplayLevel(newLevel);

            // The refresh adds its own stats message, so we don't need to add another one here
        } catch (Exception e) {
            addLogMessage("[ERROR] Failed to change display level: " + e.getMessage());
        }
    }

    /**
     * Refreshes the display buffer by refiltering all log entries from LogBuffer
     * based on the current display level. Rebuilds from the tail backwards to keep
     * the most recent matching entries.
     *
     * @return statistics message about the refresh for logging
     */
    public String refreshDisplayBuffer() {
        logLock.writeLock().lock();
        try {
            // Save current buffer size for debugging
            int beforeSize = logBuffer.size();

            // Clear current display buffer
            logBuffer.clear();

            // Collect all matching entries first
            java.util.List<String> matchingEntries = new java.util.ArrayList<>();
            org.apache.logging.log4j.Level displayLevel = LogBuffer.getDisplayLevel();

            // Get the backing buffer
            java.util.Queue<LogBuffer.LogEntry> backingBuffer = LogBuffer.getAllLogEntries();

            // CRITICAL DEBUG: Check if backing buffer is empty
            if (backingBuffer == null) {
                logBuffer.addLast("[ERROR] LogBuffer.getAllLogEntries() returned NULL!");
                return "ERROR: Backing buffer is null";
            }

            if (backingBuffer.isEmpty()) {
                logBuffer.addLast("[WARNING] LogBuffer backing buffer is EMPTY - no events have been captured yet");
                return "WARNING: Backing buffer is empty (0 entries)";
            }

            // Count entries by level for debugging
            int totalEntries = 0;
            java.util.Map<String, Integer> countByLevel = new java.util.LinkedHashMap<>();

            for (LogBuffer.LogEntry entry : backingBuffer) {
                totalEntries++;
                String levelName = entry.level.name();
                countByLevel.put(levelName, countByLevel.getOrDefault(levelName, 0) + 1);

                // Debug first 3 entries to see what we're comparing
                if (totalEntries <= 3) {
                    String debugMsg = String.format("[DEBUG] Entry %d: level=%s (%d) vs display=%s (%d), match=%b",
                                                   totalEntries, entry.level.name(), entry.level.intLevel(),
                                                   displayLevel.name(), displayLevel.intLevel(),
                                                   entry.level.intLevel() <= displayLevel.intLevel());
                    // Store for later display
                    matchingEntries.add(debugMsg);
                }

                // Add entry if its level is at or above display level
                // Note: Lower intLevel = more severe (ERROR=200, WARN=300, INFO=400, DEBUG=500, TRACE=600)
                // So to show INFO and above, we want intLevel <= INFO.intLevel()
                if (entry.level.intLevel() <= displayLevel.intLevel()) {
                    matchingEntries.add(entry.formattedMessage);
                }
            }

            // Keep only the tail (most recent entries) if we exceed maxLogLines
            int startIndex = Math.max(0, matchingEntries.size() - maxLogLines);
            for (int i = startIndex; i < matchingEntries.size(); i++) {
                logBuffer.addLast(matchingEntries.get(i));
            }

            // Reset scroll if we were at the bottom
            if (!isUserScrollingLogs) {
                logScrollOffset = Math.max(0, logBuffer.size() - getLogPanelHeight());
            }

            // Build detailed stats message
            StringBuilder stats = new StringBuilder();
            stats.append(String.format("Buffer refresh: %d→%d entries | Backing: %d total (",
                                      beforeSize, logBuffer.size(), totalEntries));
            boolean first = true;
            for (java.util.Map.Entry<String, Integer> e : countByLevel.entrySet()) {
                if (!first) stats.append(", ");
                stats.append(e.getKey()).append("=").append(e.getValue());
                first = false;
            }
            stats.append(String.format(") | Level=%s | Matched=%d | Showing=%d",
                                      displayLevel.name(), matchingEntries.size(), logBuffer.size()));

            return stats.toString();
        } finally {
            logLock.writeLock().unlock();
        }
    }

    private void renderTaskPanel(List<AttributedString> lines, int width, int contentHeight) {
        int innerWidth = Math.max(10, width - 4);

        List<SectionLine> taskEntries = new ArrayList<>();
        collectTaskLines(rootNode, taskEntries, "", true, innerWidth);

        if (taskEntries.isEmpty()) {
            taskEntries.add(new SectionLine(center("No active tasks", innerWidth), STYLE_SECONDARY));
        }

        long active = 0;
        long completed = 0;
        long failed = 0;
        for (TaskNode node : taskNodes.values()) {
            if (node.finishTime > 0) {
                if (node.lastStatus != null && node.lastStatus.runstate == RunState.FAILED) {
                    failed++;
                } else {
                    completed++;
                }
            } else {
                active++;
            }
        }

        SectionLine summaryLine = new SectionLine(
                String.format("Active: %d  Completed: %d  Failed: %d", active, completed, failed),
                STYLE_SECONDARY);

        int bodyLines = Math.max(0, contentHeight - 1);
        int totalEntries = taskEntries.size();
        int maxScrollStart = Math.max(0, totalEntries - bodyLines);
        int startIdx = Math.max(0, Math.min(taskScrollOffset, maxScrollStart));
        int endIdx = Math.min(totalEntries, startIdx + bodyLines);

        List<SectionLine> visibleBody = new ArrayList<>();
        if (bodyLines > 0) {
            visibleBody.addAll(taskEntries.subList(startIdx, endIdx));

            if (totalEntries > bodyLines && !visibleBody.isEmpty()) {
                String indicatorText = String.format("Tasks %d-%d of %d (PgUp/PgDn)",
                        startIdx + 1, endIdx, totalEntries);
                SectionLine indicator = new SectionLine(indicatorText, STYLE_SECONDARY);
                visibleBody.set(visibleBody.size() - 1, indicator);
            }
        }

        renderBoxedSection(lines, "Active Tasks", visibleBody, summaryLine, width, contentHeight);
    }

    private void collectTaskLines(DisplayNode node, List<SectionLine> lines, String prefix, boolean isLast, int innerWidth) {
        // Format this node's line (skip for root)
        if (!(node instanceof RootNode)) {
            String nodeLine = formatNodeLine(node, prefix, isLast, innerWidth);
            nodeLine = fitLine(nodeLine, innerWidth);
            lines.add(new SectionLine(nodeLine, AttributedStyle.DEFAULT));
        }

        // Recursively add children
        List<DisplayNode> children = new ArrayList<>(node.children);
        for (int i = 0; i < children.size(); i++) {
            DisplayNode child = children.get(i);
            boolean childIsLast = (i == children.size() - 1);

            String childPrefix = prefix;
            if (!(node instanceof RootNode)) {
                childPrefix += isLast ? "  " : "│ ";
            }

            collectTaskLines(child, lines, childPrefix, childIsLast, innerWidth);
        }
    }

    private String formatNodeLine(DisplayNode node, String prefix, boolean isLast, int availableWidth) {
        if (node instanceof TaskNode) {
            return formatTaskLine((TaskNode) node, prefix, isLast, availableWidth);
        } else if (node instanceof ScopeNode) {
            return formatScopeLine((ScopeNode) node, prefix, isLast, availableWidth);
        }
        return "";
    }

    private String formatScopeLine(ScopeNode node, String prefix, boolean isLast, int availableWidth) {
        StringBuilder line = new StringBuilder();

        // Tree connector and base prefix
        if (!prefix.isEmpty()) {
            line.append(prefix);
            line.append(isLast ? "└─ " : "├─ ");
        }

        // Scope icon
        line.append(node.getSymbol()).append(" ");
        line.append(node.getName());

        // Add completion indicator for closed scopes
        if (node.isComplete()) {
            line.append(" ✓");
        }

        return line.toString();
    }

    private String formatTaskLine(TaskNode node, String prefix, boolean isLast, int availableWidth) {
        StringBuilder rightPortion = new StringBuilder();

        // Build the right-aligned portion (duration, then progress bar with percentage)
        
        // Duration first - use elapsed running time if task is/was running
        long duration;
        if (node.tracker != null && node.tracker.getRunningStartTime() != null) {
            // Task has started running - use actual running time
            if (node.finishTime > 0) {
                duration = node.finishTime - node.tracker.getRunningStartTime();
            } else {
                duration = node.tracker.getElapsedRunningTime();
            }
        } else if (node.lastStatus != null && node.lastStatus.runstate == RunState.PENDING) {
            // Task hasn't started yet
            duration = 0;
        } else {
            // Fallback to old calculation
            duration = (node.finishTime > 0 ? node.finishTime : System.currentTimeMillis()) - node.startTime;
        }
        long seconds = Math.max(0, Math.round(duration / 1000.0));
        rightPortion.append(String.format(" (%ds) ", seconds));

        // Progress bar with percentage centered in it (fixed 22 characters total)
        if (node.lastStatus != null) {
            String progressBarWithPercent = createProgressBarWithCenteredPercent(node.lastStatus.progress);
            rightPortion.append(progressBarWithPercent);
        } else {
            rightPortion.append("[        0%          ]");
        }

        // Completion marker
        if (node.finishTime > 0 && node.lastStatus != null) {
            if (node.lastStatus.runstate == RunState.SUCCESS) {
                rightPortion.append(" ✓");
            } else if (node.lastStatus.runstate == RunState.FAILED) {
                rightPortion.append(" ✗");
            }
        }

        StringBuilder line = new StringBuilder();

        // Tree connector and base prefix
        if (!prefix.isEmpty()) {
            line.append(prefix);
            line.append(isLast ? "└─ " : "├─ ");
        }

        // Status icon
        line.append(node.getSymbol()).append(" ");

        // Determine maximum available width for the task name before adding context/spaces
        int maxNameWidth = Math.max(0, availableWidth - line.length() - rightPortion.length());
        String taskName = node.getName();
        if (taskName.length() > maxNameWidth) {
            taskName = fitTaskName(taskName, maxNameWidth);
        }
        line.append(taskName);

        // Calculate space for contextual details
        int leftLength = line.length();
        int rightLength = rightPortion.length();
        int totalUsed = leftLength + rightLength;
        int spacesNeeded = Math.max(1, availableWidth - totalUsed);

        // Add contextual details in the middle if space allows
        if (spacesNeeded > 5) {
            StringBuilder context = new StringBuilder();

            // Add task state if not running
            if (node.lastStatus != null) {
                if (node.lastStatus.runstate == RunState.PENDING) {
                    context.append(" [pending]");
                } else if (node.lastStatus.runstate == RunState.RUNNING) {
                    // Add any additional context from the task if available
                    Object tracked = node.lastStatus.tracked;
                    if (tracked != null && tracked.toString().contains(":")) {
                        // Extract detail after colon if present
                        String detail = tracked.toString();
                        int colonIdx = detail.indexOf(":");
                        if (colonIdx >= 0 && colonIdx < detail.length() - 1) {
                            context.append(" -").append(detail.substring(colonIdx + 1).trim());
                        }
                    }
                }
            }

            line.append(context);
            spacesNeeded = Math.max(1, availableWidth - leftLength - context.length() - rightLength);
        }

        // Fill with spaces
        for (int i = 0; i < spacesNeeded; i++) {
            line.append(" ");
        }

        // Add right-aligned portion
        line.append(rightPortion);

        return line.toString();
    }

    private void renderLogPanel(List<AttributedString> lines, Size size) {
        renderLogPanel(lines, size.getColumns(), 10);
    }

    private void renderLogPanel(List<AttributedString> lines, int width, int contentHeight) {
        int innerWidth = Math.max(10, width - 4);
        List<SectionLine> logLines = new ArrayList<>();
        SectionLine footerLine;

        logLock.readLock().lock();
        try {
            int bodyLines = Math.max(0, contentHeight - 1);
            int totalLogs = logBuffer.size();

            int startIdx;
            if (isUserScrollingLogs) {
                int maxScrollStart = Math.max(0, totalLogs - bodyLines);
                startIdx = Math.max(0, Math.min(logScrollOffset, maxScrollStart));
                logScrollOffset = startIdx;
            } else {
                startIdx = Math.max(0, totalLogs - bodyLines);
                logScrollOffset = startIdx;
            }
            int endIdx = Math.min(totalLogs, startIdx + bodyLines);

            for (int i = startIdx; i < endIdx; i++) {
                String logLine = logBuffer.get(i);
                logLines.add(new SectionLine(fitLine(logLine, innerWidth), getLogStyle(logLine)));
            }

            String footerText;
            if (totalLogs == 0) {
                footerText = "Waiting for log output…";
            } else if (bodyLines == 0) {
                footerText = String.format("%d log lines (expand panel to view)", totalLogs);
            } else if (totalLogs > bodyLines) {
                if (isUserScrollingLogs) {
                    footerText = String.format("Logs %d-%d of %d (↑/↓ to scroll)",
                            startIdx + 1, endIdx, totalLogs);
                } else {
                    footerText = String.format("LIVE showing last %d of %d lines (↑ to scroll)",
                            Math.max(0, endIdx - startIdx), totalLogs);
                }
            } else {
                footerText = String.format("Showing all %d log lines", totalLogs);
            }

            footerLine = new SectionLine(fitLine(footerText, innerWidth), STYLE_SECONDARY);
            lastLogDisplayTime = System.currentTimeMillis();
        } finally {
            logLock.readLock().unlock();
        }

        renderBoxedSection(lines, "Console Output", logLines, footerLine, width, contentHeight);
    }

    private void renderBoxedSection(List<AttributedString> target, String title, List<SectionLine> body,
                                    SectionLine footer, int width, int contentHeight) {
        int adjustedHeight = Math.max(1, contentHeight);
        int innerWidth = Math.max(10, width - 4);
        target.add(buildSectionBorder('╔', '╗', title, width));

        int rowsRendered = 0;
        int bodyLines = adjustedHeight - (footer != null ? 1 : 0);
        if (bodyLines < 0) {
            bodyLines = 0;
        }

        for (int i = 0; i < bodyLines; i++) {
            SectionLine line = (i < body.size()) ? body.get(i) : null;
            target.add(renderBoxLine(line, innerWidth));
            rowsRendered++;
        }

        if (footer != null) {
            target.add(renderBoxLine(footer, innerWidth));
            rowsRendered++;
        }

        while (rowsRendered < adjustedHeight) {
            target.add(renderBoxLine(null, innerWidth));
            rowsRendered++;
        }

        target.add(buildSectionBorder('╚', '╝', null, width));
    }

    private AttributedString renderBoxLine(SectionLine line, int innerWidth) {
        String text = line != null ? line.text : "";
        String fitted = fitLine(text, innerWidth);
        int padding = Math.max(0, innerWidth - fitted.length());

        AttributedStringBuilder builder = new AttributedStringBuilder();
        builder.style(STYLE_BORDER).append("║ ");
        if (line != null && line.style != null) {
            builder.style(line.style);
        } else {
            builder.style(AttributedStyle.DEFAULT);
        }
        builder.append(fitted);
        builder.style(STYLE_BORDER).append(" ".repeat(padding)).append(" ║");
        return builder.toAttributedString();
    }

    private AttributedString buildSectionBorder(char left, char right, String title, int width) {
        int innerWidth = Math.max(0, width - 2);
        AttributedStringBuilder builder = new AttributedStringBuilder();

        // Apply bright cyan bold style for borders
        builder.style(STYLE_BORDER);
        builder.append(String.valueOf(left));

        if (title != null && !title.isEmpty()) {
            String trimmed = title.trim();
            if (trimmed.length() > innerWidth - 4) {
                trimmed = fitLine(trimmed, innerWidth - 4);
            }

            int titleLen = trimmed.length() + 4; // Account for spaces and equals
            int remaining = Math.max(0, innerWidth - titleLen);
            int leftPad = remaining / 2;
            int rightPad = remaining - leftPad;

            // Left padding with double lines
            for (int i = 0; i < leftPad; i++) {
                builder.append("═");
            }

            // Title with yellow highlight
            builder.append("═");
            builder.style(STYLE_BORDER_TITLE).append(" " + trimmed + " ");
            builder.style(STYLE_BORDER).append("═");

            // Right padding with double lines
            for (int i = 0; i < rightPad; i++) {
                builder.append("═");
            }
        } else {
            // Fill with double lines
            for (int i = 0; i < innerWidth; i++) {
                builder.append("═");
            }
        }

        builder.append(String.valueOf(right));
        return builder.toAttributedString();
    }

    private String fitLine(String text, int maxWidth) {
        if (text == null) {
            return "";
        }
        if (text.length() <= maxWidth) {
            return text;
        }
        if (maxWidth <= 1) {
            return text.substring(0, Math.max(0, maxWidth));
        }
        return text.substring(0, Math.max(0, maxWidth - 1)) + "…";
    }

    private int getLogPanelHeight() {
        Size size = terminal.getSize();
        int taskLines = countTaskLines(rootNode);
        int headerFooterLines = 6; // Headers and footers
        int remaining = size.getRows() - taskLines - headerFooterLines;
        return Math.max(5, Math.min(remaining, 10));
    }

    private static final class SectionLine {
        final String text;
        final AttributedStyle style;

        SectionLine(String text, AttributedStyle style) {
            this.text = text == null ? "" : text;
            this.style = style;
        }
    }

    private int countTaskLines(DisplayNode node) {
        int count = (node instanceof RootNode) ? 0 : 1;
        for (DisplayNode child : node.children) {
            count += countTaskLines(child);
        }
        return count;
    }

    private AttributedStyle getLogStyle(String logLine) {
        String upper = logLine.toUpperCase();
        if (upper.contains("[ERROR]") || upper.contains("ERROR") || upper.contains("SEVERE")) {
            return STYLE_LOG_ERROR;
        } else if (upper.contains("[WARN]") || upper.contains("WARNING")) {
            return STYLE_LOG_WARN;
        } else if (upper.contains("[DEBUG]") || upper.contains("TRACE")) {
            return STYLE_LOG_DEBUG;
        } else {
            return STYLE_LOG_INFO;
        }
    }

    private String center(String text, int width) {
        int padding = (width - text.length()) / 2;
        return " ".repeat(Math.max(0, padding)) + text + " ".repeat(Math.max(0, width - text.length() - padding));
    }

    private void renderStatusBar(List<AttributedString> lines, int width) {
        // Add bottom border first
        lines.add(buildSectionBorder('╚', '╝', null, width));

        // Create bottom status bar
        AttributedStringBuilder statusBar = new AttributedStringBuilder();

        // Get current time
        String timeStr = LocalDateTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss"));

        // Count active tasks
        long activeTasks = taskNodes.values().stream()
                .filter(n -> n.lastStatus != null && n.lastStatus.runstate == RunState.RUNNING)
                .count();
        long completedTasks = taskNodes.values().stream()
                .filter(n -> n.lastStatus != null && n.lastStatus.runstate == RunState.SUCCESS)
                .count();

        // Build status line
        statusBar.style(AttributedStyle.DEFAULT.background(AttributedStyle.BLUE).foreground(AttributedStyle.WHITE))
                .append(" ")
                .append(timeStr)
                .append(" │ ");

        statusBar.style(AttributedStyle.DEFAULT.background(AttributedStyle.BLUE).foreground(AttributedStyle.YELLOW))
                .append("Active: ").append(String.valueOf(activeTasks))
                .append(" ");

        statusBar.style(AttributedStyle.DEFAULT.background(AttributedStyle.BLUE).foreground(AttributedStyle.GREEN))
                .append("Complete: ").append(String.valueOf(completedTasks))
                .append(" ");

        statusBar.style(AttributedStyle.DEFAULT.background(AttributedStyle.BLUE).foreground(AttributedStyle.WHITE))
                .append("│ ")
                .append(isUserScrollingLogs ? "↑↓: Scroll Logs │" : "↑: Scroll Logs │");
        statusBar.append(" PgUp/PgDn: Adjust Split │ q: Quit");

        // Pad to full width
        int currentLen = statusBar.toAttributedString().columnLength();
        if (currentLen < width) {
            statusBar.append(" ".repeat(width - currentLen));
        }

        lines.add(statusBar.toAttributedString());
    }


    private String createProgressBar(double progress) {
        int barLength = 20;

        // Braille patterns for 1/8 increments
        char[] brailleProgress = {
            ' ',     // 0/8 - empty
            '⡀',     // 1/8
            '⡄',     // 2/8
            '⡆',     // 3/8
            '⡇',     // 4/8
            '⣇',     // 5/8
            '⣧',     // 6/8
            '⣷',     // 7/8
        };
        char fullBlock = '⣿';  // 8/8 - full

        // Calculate progress in terms of 1/8 increments
        double totalEighths = barLength * 8.0 * progress;
        int fullChars = (int) (totalEighths / 8);
        int remainder = (int) (totalEighths % 8);

        StringBuilder bar = new StringBuilder("[");

        for (int i = 0; i < barLength; i++) {
            if (i < fullChars) {
                bar.append(fullBlock);
            } else if (i == fullChars && remainder > 0) {
                bar.append(brailleProgress[remainder]);
            } else {
                bar.append(' ');
            }
        }

        bar.append("]");
        return bar.toString();
    }

    private String createProgressBarWithCenteredPercent(double progress) {
        int barLength = 20;  // Total bar length
        String percentStr = String.format("%3.0f%%", progress * 100);
        int percentLen = percentStr.length();

        // Braille patterns for 1/8 increments (0/8 to 7/8 filled)
        // Using vertical Braille patterns that fill from left to right
        char[] brailleProgress = {
            ' ',     // 0/8 - empty
            '⡀',     // 1/8
            '⡄',     // 2/8
            '⡆',     // 3/8
            '⡇',     // 4/8
            '⣇',     // 5/8
            '⣧',     // 6/8
            '⣷',     // 7/8
        };
        char fullBlock = '⣿';  // 8/8 - full

        // Calculate progress in terms of 1/8 increments
        double totalEighths = barLength * 8.0 * progress;
        int fullChars = (int) (totalEighths / 8);
        int remainder = (int) (totalEighths % 8);

        // Calculate where to place the percentage (centered)
        int percentStart = (barLength - percentLen) / 2;

        StringBuilder bar = new StringBuilder("[");

        for (int i = 0; i < barLength; i++) {
            // Check if we should insert percentage text here
            if (i >= percentStart && i < percentStart + percentLen) {
                bar.append(percentStr.charAt(i - percentStart));
            } else if (i < fullChars) {
                bar.append(fullBlock);
            } else if (i == fullChars && remainder > 0) {
                bar.append(brailleProgress[remainder]);
            } else {
                bar.append(' ');
            }
        }

        bar.append("]");
        return bar.toString();
    }

    private String getTaskName(StatusTracker<?> tracker, int maxWidth) {
        String fullName = StatusTracker.extractTaskName(tracker);
        return fitTaskName(fullName, maxWidth);
    }

    private String fitTaskName(String name, int maxWidth) {
        if (maxWidth <= 0) {
            return "";
        }

        if (name.length() <= maxWidth) {
            return name;
        }

        if (maxWidth <= 3) {
            return name.substring(0, maxWidth);
        }

        return name.substring(0, maxWidth - 3) + "...";
    }

    public boolean isClosed() {
        return closed.get();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            performCleanup();
        }
    }

    private void performCleanup() {
        // Clear the active sink
        LogBuffer.clearActiveSink();

        // Capture current log buffer state before closing
        List<String> logsSnapshot;
        logLock.readLock().lock();
        try {
            logsSnapshot = new ArrayList<>(logBuffer);
        } finally {
            logLock.readLock().unlock();
        }

        // Stop the render thread first
        try {
            renderThread.join(2000); // Wait up to 2 seconds for clean shutdown
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        try {
            // Clear the display completely and reset terminal
            display.update(Collections.emptyList(), 0);

            // Clear screen and reset cursor
            terminal.puts(org.jline.utils.InfoCmp.Capability.clear_screen);
            terminal.puts(org.jline.utils.InfoCmp.Capability.cursor_home);

            // Reset all terminal attributes
            terminal.writer().print("\033[0m");      // Reset colors
            terminal.writer().print("\033[?25h");    // Show cursor
            terminal.writer().print("\033[?1049l");  // Exit alternate screen (if used)
            terminal.writer().flush();

            // Exit raw mode to restore normal terminal behavior
            org.jline.terminal.Attributes attrs = terminal.getAttributes();
            attrs.setLocalFlag(org.jline.terminal.Attributes.LocalFlag.ICANON, true);
            attrs.setLocalFlag(org.jline.terminal.Attributes.LocalFlag.ECHO, true);
            terminal.setAttributes(attrs);

            terminal.flush();
            terminal.close();
        } catch (Exception e) {
            logger.error("Error during terminal cleanup", e);
            // Force terminal restoration even if normal cleanup failed
            try {
                System.out.print("\033[0m");      // Reset colors
                System.out.print("\033[?25h");    // Show cursor
                System.out.flush();
            } catch (Exception ignored) {}
        } finally {
            // Always restore original streams, even if terminal cleanup fails
            System.setOut(originalOut);
            System.setErr(originalErr);

            // Force terminal reset using direct ANSI codes to stdout
            try {
                originalOut.print("\033[0m");       // Reset all attributes
                originalOut.print("\033[?25h");     // Show cursor
                originalOut.print("\033[?1049l");   // Exit alternate screen
                originalOut.print("\033c");         // Reset terminal (RIS)
                originalOut.flush();
            } catch (Exception ignored) {}

            // Print a newline to ensure clean prompt
            originalOut.println();

            // Dump the log buffer to stdout for user context
            if (!logsSnapshot.isEmpty()) {
                originalOut.println("=== Console Log History ===");
                for (String log : logsSnapshot) {
                    originalOut.println(log);
                }
                originalOut.println("=== End Console Log History ===");
            }

            originalOut.flush();
        }
    }


    /**
     * Custom PrintStream that captures output and adds it to the log buffer
     */
    private class LogCapturePrintStream extends PrintStream {
        private final String prefix;
        private final ByteArrayOutputStream pendingBytes;

        LogCapturePrintStream(String prefix) {
            super(new ByteArrayOutputStream());
            this.prefix = prefix;
            this.pendingBytes = new ByteArrayOutputStream();
        }

        @Override
        public synchronized void println(String x) {
            writeByteArray((x == null ? "null" : x).getBytes(StandardCharsets.UTF_8));
            write('\n');
        }

        @Override
        public synchronized void println() {
            write('\n');
        }

        @Override
        public synchronized void print(String s) {
            writeByteArray((s == null ? "null" : s).getBytes(StandardCharsets.UTF_8));
        }

        @Override
        public synchronized void print(char[] s) {
            if (s == null) {
                writeByteArray("null".getBytes(StandardCharsets.UTF_8));
            } else {
                writeByteArray(new String(s).getBytes(StandardCharsets.UTF_8));
            }
        }

        @Override
        public synchronized void println(char[] s) {
            print(s);
            write('\n');
        }

        @Override
        public synchronized void write(byte[] buf, int off, int len) {
            if (buf == null || len <= 0) {
                return;
            }

            int end = off + len;
            for (int i = off; i < end; i++) {
                byte b = buf[i];
                if (b == '\n' || b == '\r') {
                    flushPendingBytes();
                } else {
                    pendingBytes.write(b);
                }
            }
        }

        @Override
        public synchronized void write(int b) {
            byte value = (byte) b;
            if (value == '\n' || value == '\r') {
                flushPendingBytes();
            } else {
                pendingBytes.write(value);
            }
        }

        @Override
        public synchronized void flush() {
            flushPendingBytes();
        }

        private void writeByteArray(byte[] data) {
            if (data == null || data.length == 0) {
                return;
            }
            write(data, 0, data.length);
        }

        private void flushPendingBytes() {
            if (pendingBytes.size() == 0) {
                return;
            }

            String line = new String(pendingBytes.toByteArray(), StandardCharsets.UTF_8);
            pendingBytes.reset();
            emitLine(line);
        }

        private void emitLine(String line) {
            if (line == null || line.trim().isEmpty()) {
                return;
            }

            logLock.writeLock().lock();
            try {
                String decorated = line;
                if (!decorated.matches("^\\[\\d{2}:\\d{2}:\\d{2}\\].*")) {
                    decorated = "[" + LocalDateTime.now().format(timeFormatter) + "] " + decorated;
                }

                if (prefix != null && !prefix.isEmpty()) {
                    if (decorated.startsWith("[") && decorated.indexOf(']') != -1) {
                        int closing = decorated.indexOf(']');
                        decorated = decorated.substring(0, closing + 1) + " [" + prefix + "]" + decorated.substring(closing + 1);
                    } else {
                        decorated = "[" + prefix + "] " + decorated;
                    }
                }

                logBuffer.addLast(decorated);

                while (logBuffer.size() > maxLogLines) {
                    logBuffer.removeFirst();
                    // When manually scrolling, we need to adjust offset to maintain the same view
                    // Even if offset is 0, removing from front means we need to "scroll up" to stay in place
                    // However, we can't have negative offset, so content will shift if at the very top
                    if (isUserScrollingLogs && logScrollOffset > 0) {
                        logScrollOffset--;
                    } else if (!isUserScrollingLogs) {
                        // When auto-scrolling, decrement to maintain bottom view
                        if (logScrollOffset > 0) {
                            logScrollOffset--;
                        }
                    }
                }

                if (!isUserScrollingLogs) {
                    int maxScroll = Math.max(0, logBuffer.size() - getLogPanelHeight());
                    logScrollOffset = maxScroll;
                }
            } finally {
                logLock.writeLock().unlock();
            }
        }
    }

    private void saveScreenshot() {
        List<String> snapshot = lastRenderSnapshot;
        if (snapshot == null || snapshot.isEmpty()) {
            addLogMessage("No screen content available to save.");
            return;
        }

        String timestamp = LocalDateTime.now().format(SCREENSHOT_FORMAT);
        Path path = Paths.get(String.format("console-panel-%s.txt", timestamp));
        try {
            Files.write(path, snapshot, StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
            addLogMessage("Saved console snapshot to " + path.toAbsolutePath());
        } catch (IOException e) {
            addLogMessage("Failed to save console snapshot: " + e.getMessage());
        }
    }

    /**
     * Base class for display hierarchy nodes (scopes and tasks)
     */
    private static abstract class DisplayNode {
        final DisplayNode parent;
        final List<DisplayNode> children;
        final long startTime;

        DisplayNode(DisplayNode parent) {
            this.parent = parent;
            this.children = Collections.synchronizedList(new ArrayList<>());
            this.startTime = System.currentTimeMillis();
        }

        abstract String getName();
        abstract String getSymbol();
        abstract boolean isComplete();
    }

    /**
     * Root node containing all top-level scopes and trackers
     */
    private static class RootNode extends DisplayNode {
        RootNode() {
            super(null);
        }

        @Override
        String getName() {
            return "Root";
        }

        @Override
        String getSymbol() {
            return "";
        }

        @Override
        boolean isComplete() {
            return children.stream().allMatch(DisplayNode::isComplete);
        }
    }

    /**
     * Node representing an organizational scope
     */
    private static class ScopeNode extends DisplayNode {
        final TrackerScope scope;
        long finishTime;

        ScopeNode(TrackerScope scope, DisplayNode parent) {
            super(parent);
            this.scope = scope;
            this.finishTime = 0;
        }

        @Override
        String getName() {
            return scope.toString(); // Uses scope's toString with task counts
        }

        @Override
        String getSymbol() {
            return "📁"; // Folder icon for scopes
        }

        @Override
        boolean isComplete() {
            return finishTime > 0 || scope.isClosed() || scope.isComplete();
        }
    }

    /**
     * Node representing an actual task being tracked
     */
    private static class TaskNode extends DisplayNode {
        final StatusTracker<?> tracker;
        StatusUpdate<?> lastStatus;
        long lastUpdateTime;
        long finishTime;

        TaskNode(StatusTracker<?> tracker, DisplayNode parent) {
            super(parent);
            this.tracker = tracker;
            this.lastUpdateTime = startTime;
            this.finishTime = 0;
        }

        @Override
        String getName() {
            return StatusTracker.extractTaskName(tracker);
        }

        @Override
        String getSymbol() {
            if (lastStatus == null) {
                return "○"; // Pending
            }
            switch (lastStatus.runstate) {
                case PENDING: return "○";
                case RUNNING: return "▶";
                case SUCCESS: return "●";
                case FAILED: return "✗";
                case CANCELLED: return "◼";
                default: return "?";
            }
        }

        @Override
        boolean isComplete() {
            return finishTime > 0;
        }
    }

    /**
     * Runnable implementation for JVM shutdown hook that ensures proper terminal cleanup.
     * This named type improves stack trace clarity during shutdown sequences and makes
     * debugging shutdown-related issues easier to diagnose.
     */
    private static final class ShutdownCleanupRunnable implements Runnable {
        private final ConsolePanelSink sink;

        ShutdownCleanupRunnable(ConsolePanelSink sink) {
            this.sink = sink;
        }

        @Override
        public void run() {
            if (!sink.closed.get()) {
                sink.close();
            }
        }
    }

    /**
     * Create a new builder for configuring ConsolePanelSink.
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring {@link ConsolePanelSink} instances with fluent method chaining.
     * Provides control over refresh rates, task retention, logging behavior, keyboard handlers,
     * and visual presentation.
     *
     * <p><strong>Default Configuration:</strong></p>
     * <ul>
     *   <li>Refresh rate: 250ms</li>
     *   <li>Completed task retention: 5000ms (5 seconds)</li>
     *   <li>Color output: enabled</li>
     *   <li>Max log lines: 1000</li>
     *   <li>System stream capture: disabled</li>
     *   <li>Auto-exit on 'q': disabled</li>
     *   <li>Custom keyboard handlers: none</li>
     * </ul>
     *
     * <p><strong>Example:</strong></p>
     * <pre>{@code
     * ConsolePanelSink sink = ConsolePanelSink.builder()
     *     .withRefreshRateMs(100)
     *     .withCompletedTaskRetention(3, TimeUnit.SECONDS)
     *     .withColorOutput(true)
     *     .withMaxLogLines(500)
     *     .withCaptureSystemStreams(true)
     *     .withAutoExit(true)
     *     .withKeyHandler("shift-right", () -> handleSpeedUp())
     *     .build();
     * }</pre>
     *
     * @see ConsolePanelSink
     */
    public static class Builder {
        private long refreshRateMs = 250;
        private long completedRetentionMs = 5000;
        private boolean useColors = true;
        private int maxLogLines = 1000;
        private boolean captureSystemStreams = false;
        private boolean autoExit = false; // Default to false - user must enable
        private Map<String, Runnable> customKeyHandlers = new HashMap<>();

        public Builder withRefreshRate(long duration, TimeUnit unit) {
            this.refreshRateMs = unit.toMillis(duration);
            return this;
        }

        public Builder withCompletedTaskRetention(long duration, TimeUnit unit) {
            this.completedRetentionMs = unit.toMillis(duration);
            return this;
        }

        public Builder withColorOutput(boolean useColors) {
            this.useColors = useColors;
            return this;
        }

        public Builder withMaxLogLines(int maxLogLines) {
            this.maxLogLines = maxLogLines;
            return this;
        }

        public Builder withCaptureSystemStreams(boolean capture) {
            this.captureSystemStreams = capture;
            return this;
        }

        public Builder withRefreshRateMs(long refreshRateMs) {
            this.refreshRateMs = refreshRateMs;
            return this;
        }

        /**
         * Configures whether the panel should exit automatically when 'q' is pressed.
         * When false, the panel will remain open even when 'q' is pressed, requiring
         * external shutdown (e.g., via close() or application exit).
         *
         * @param autoExit true (default) to allow 'q' to exit, false to disable auto-exit
         * @return this builder
         */
        public Builder withAutoExit(boolean autoExit) {
            this.autoExit = autoExit;
            return this;
        }

        /**
         * Registers a custom keyboard handler for the specified key combination.
         * This allows applications to extend the interactive capabilities of the
         * console panel with application-specific controls.
         *
         * <p>The handler is invoked synchronously on the render thread when the
         * corresponding key combination is detected. Handlers should execute quickly
         * to avoid blocking the UI. For longer operations, consider spawning a
         * separate thread from within the handler.
         *
         * <p><strong>Supported Key Combinations:</strong></p>
         * <ul>
         *   <li><code>"shift-left"</code> - Shift + Left Arrow key</li>
         *   <li><code>"shift-right"</code> - Shift + Right Arrow key</li>
         * </ul>
         *
         * <p><strong>Example Usage:</strong></p>
         * <pre>{@code
         * SimulatedClock clock = new SimulatedClock();
         * ConsolePanelSink sink = ConsolePanelSink.builder()
         *     .withKeyHandler("shift-right", () -> {
         *         clock.speedUp();
         *         sink.addLogMessage("Simulation speed: " + clock.getSpeedDescription());
         *     })
         *     .withKeyHandler("shift-left", () -> {
         *         clock.slowDown();
         *         sink.addLogMessage("Simulation speed: " + clock.getSpeedDescription());
         *     })
         *     .build();
         * }</pre>
         *
         * <p><strong>Thread Safety:</strong> Handlers are executed on the render thread,
         * so any shared state accessed by the handler should be properly synchronized.</p>
         *
         * @param key the key combination identifier (case-sensitive)
         * @param handler the runnable to execute when the key is pressed; must not be null
         * @return this builder for method chaining
         * @throws NullPointerException if handler is null
         */
        public Builder withKeyHandler(String key, Runnable handler) {
            this.customKeyHandlers.put(key, handler);
            return this;
        }

        public ConsolePanelSink build() {
            return new ConsolePanelSink(this);
        }
    }
}
