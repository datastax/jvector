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

import io.github.jbellis.jvector.status.StatusSink;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.StatusUpdate;
import org.jline.utils.NonBlockingReader;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.terminal.Size;
import org.jline.utils.AttributedString;
import org.jline.utils.AttributedStringBuilder;
import org.jline.utils.AttributedStyle;
import org.jline.utils.Display;


import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.nio.charset.StandardCharsets;

/**
 * A sophisticated terminal-based status sink that provides a hierarchical, stateful view
 * of task progress using JLine3. This enhanced version includes a scrollable logging panel
 * and captures all console output for integrated display.
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
 * @see StatusSink
 * @see StatusTracker
 * @since 4.0.0
 */
public class ConsolePanelSink implements StatusSink, AutoCloseable {

    private Terminal terminal;
    private Display display;
    private final Thread renderThread;
    private final Map<StatusTracker<?>, TaskNode> taskNodes;
    private final TaskNode rootNode;
    private final DateTimeFormatter timeFormatter;
    private final long refreshRateMs;
    private final long completedRetentionMs;
    private final boolean useColors;
    private final AtomicBoolean closed;
    private final AtomicBoolean shouldRender;

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
                System.err.println("[ConsolePanelSink] Could not clear screen: " + e.getMessage());
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize JLine terminal: " + e.getMessage(), e);
        }

        this.refreshRateMs = builder.refreshRateMs;
        this.completedRetentionMs = builder.completedRetentionMs;
        this.useColors = builder.useColors;
        this.maxLogLines = builder.maxLogLines;
        this.taskNodes = new ConcurrentHashMap<>();
        this.rootNode = new TaskNode(null, null);
        this.timeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        this.closed = new AtomicBoolean(false);
        this.shouldRender = new AtomicBoolean(true);

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
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (!closed.get()) {
                close();
            }
        }, "ConsolePanelSink-Shutdown"));

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
            System.err.println("[ConsolePanelSink] Render loop started with non-blocking input");

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
                }

                // Render at specified refresh rate
                if (now - lastRenderTime >= refreshRateMs) {
                    refresh();
                    lastRenderTime = now;
                }

                // Small sleep to prevent CPU spinning
                Thread.sleep(10);
            }
        } catch (Exception e) {
            if (!closed.get()) {
                System.err.println("[ConsolePanelSink] Render loop error: " + e.getMessage());
                e.printStackTrace();
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
            System.err.println("[ConsolePanelSink] Render loop exited");
        }
    }

    private void cleanupCompletedTasks(long now) {
        // Remove completed tasks based on parent completion status
        // A completed task is removed only when:
        // 1. It has no parent (root task) and exceeded retention time, OR
        // 2. Its parent task is also completed
        List<Map.Entry<StatusTracker<?>, TaskNode>> toRemove = new ArrayList<>();

        for (Map.Entry<StatusTracker<?>, TaskNode> entry : taskNodes.entrySet()) {
            TaskNode node = entry.getValue();
            if (node.finishTime > 0) {
                // Check if this completed task should be removed
                if (node.parent == null) {
                    // Root task - use standard retention time
                    if ((now - node.finishTime) > completedRetentionMs) {
                        toRemove.add(entry);
                    }
                } else if (node.parent.finishTime > 0) {
                    // Parent is also completed - remove child after brief delay
                    // This ensures the final state is visible before cleanup
                    if ((now - node.finishTime) > 1000) {  // 1 second minimum visibility
                        toRemove.add(entry);
                    }
                }
                // If parent is still running, keep this completed child visible
            }
        }

        for (Map.Entry<StatusTracker<?>, TaskNode> entry : toRemove) {
            StatusTracker<?> tracker = entry.getKey();
            TaskNode node = entry.getValue();

            taskNodes.remove(tracker);
            if (node.parent != null) {
                node.parent.children.remove(node);
            } else {
                rootNode.children.remove(node);
            }
        }
    }


    /**
     * Add a log message to the display buffer.
     * This is called by LogBuffer to add logging framework messages.
     * Only sink methods should mutate the logBuffer.
     */
    void addLogMessage(String message) {
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
                if (logScrollOffset > 0) {
                    logScrollOffset--;
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

    private void handleInput(NonBlockingReader reader, int c) {
        logLock.writeLock().lock();
        try {
            // Handle quit command
            if (c == 'q' || c == 'Q') {
                // Signal to close - don't call close() directly to avoid deadlock
                closed.set(true);
                return;
            }

            // Handle [ and ] for adjusting split position
            if (c == '[') {
                // Expand task panel, shrink log panel
                splitOffset = Math.max(-10, splitOffset - 2);
                return;
            }
            if (c == ']') {
                // Shrink task panel, expand log panel
                splitOffset = Math.min(10, splitOffset + 2);
                return;
            }

            // Handle arrow keys and special sequences
            if (c == 27) { // ESC sequence
                try {
                    int next = reader.read(10);
                    if (next == '[') {
                        int key = reader.read(10);
                        switch (key) {
                            case 'A': // Up arrow - scroll logs up
                                if (logScrollOffset > 0) {
                                    logScrollOffset--;
                                    isUserScrollingLogs = true; // User is manually scrolling
                                }
                                break;
                            case 'B': // Down arrow - scroll logs down
                                int maxLogScroll = Math.max(0, logBuffer.size() - getLogPanelHeight());
                                if (logScrollOffset < maxLogScroll) {
                                    logScrollOffset++;
                                    isUserScrollingLogs = true; // User is manually scrolling
                                } else if (logScrollOffset == maxLogScroll) {
                                    isUserScrollingLogs = false; // At bottom, resume auto-follow
                                }
                                break;
                            case '1': // ESC[1;2A or ESC[1;2B - Shift+arrows
                                int next2 = reader.read(10);
                                if (next2 == ';') {
                                    int modifier = reader.read(10);
                                    if (modifier == '2') { // Shift modifier
                                        int direction = reader.read(10);
                                        int pageSize = getLogPanelHeight() - 1;
                                        if (direction == 'A') { // Shift+Up - page up
                                            logScrollOffset = Math.max(0, logScrollOffset - pageSize);
                                            isUserScrollingLogs = true;
                                        } else if (direction == 'B') { // Shift+Down - page down
                                            int maxScroll = Math.max(0, logBuffer.size() - getLogPanelHeight());
                                            logScrollOffset = Math.min(maxScroll, logScrollOffset + pageSize);
                                            isUserScrollingLogs = logScrollOffset < maxScroll;
                                        }
                                    }
                                }
                                break;
                            case '5': // Page Up (ESC[5~) - Expand task panel, shrink log panel
                                if (reader.read(10) == '~') { // consume ~
                                    splitOffset = Math.max(-10, splitOffset - 2);
                                }
                                break;
                            case '6': // Page Down (ESC[6~) - Shrink task panel, expand log panel
                                if (reader.read(10) == '~') { // consume ~
                                    splitOffset = Math.min(10, splitOffset + 2);
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

    @Override
    public void taskStarted(StatusTracker<?> task) {
        if (closed.get()) return;

        // Buffer the update - no rendering here, just update data structures
        TaskNode node = new TaskNode(task, findParent(task));
        taskNodes.put(task, node);

        TaskNode parent = node.parent != null ? node.parent : rootNode;
        parent.children.add(node);
        // Note: Render thread will pick up this change at next refresh cycle
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
                if (finalStatus.runstate == StatusUpdate.RunState.SUCCESS) {
                    node.lastStatus = new StatusUpdate<>(1.0, StatusUpdate.RunState.SUCCESS, finalStatus.tracked);
                } else {
                    node.lastStatus = finalStatus;
                }
                node.lastUpdateTime = System.currentTimeMillis();
            }
            // Note: Completed tasks will be cleaned up by the render thread
            // based on completedRetentionMs
        }
    }

    private TaskNode findParent(StatusTracker<?> task) {
        // Use the parent relationship from StatusTracker
        StatusTracker<?> parentTracker = task.getParent();
        if (parentTracker != null) {
            return taskNodes.get(parentTracker);
        }
        return null;
    }

    private static volatile int refreshCount = 0;
    private Size lastKnownSize = null;

    private void refresh() {
        if (closed.get()) return;

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
                        System.err.println("[ConsolePanelSink] Terminal size detection failed (got " +
                            size.getColumns() + "x" + size.getRows() +
                            "). Using default size 100x40. " +
                            "This may happen when running in certain IDEs or piped environments.");
                    } else {
                        System.err.println("[ConsolePanelSink] Terminal size is null. Using default size 100x40.");
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
            // We need: 1 top border, 1 middle divider, 1 bottom border, 1 status bar = 4 total overhead lines
            int totalOverhead = 4;
            int availableContent = Math.max(2, size.getRows() - totalOverhead);

            // Split available content between task and log panels (2/3 for tasks, 1/3 for logs)
            int baseTaskContent = Math.max(1, (availableContent * 2) / 3);
            int taskContentHeight = Math.max(1, Math.min(availableContent - 1, baseTaskContent + splitOffset));
            int logContentHeight = Math.max(1, availableContent - taskContentHeight);

            lastTaskContentHeight = taskContentHeight;
            lastLogContentHeight = logContentHeight;

            // Build the display with minimal borders
            renderCompactHeader(lines, size.getColumns());
            renderTaskContent(lines, size.getColumns(), taskContentHeight);
            renderMiddleDivider(lines, size.getColumns());
            renderLogContent(lines, size.getColumns(), logContentHeight);
            renderBottomBar(lines, size.getColumns());

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
                System.err.println("[ConsolePanelSink] About to update display with " + lines.size() + " lines");
                if (!lines.isEmpty()) {
                    String firstLine = lines.get(0).toAnsi(terminal);
                    System.err.println("[ConsolePanelSink] First line content (len=" + firstLine.length() + "): " +
                        (firstLine.length() > 50 ? firstLine.substring(0, 50) + "..." : firstLine));
                }
            }

            // Update display - Display class handles differential updates
            // Position cursor at bottom right to hide it
            display.update(lines, size.cursorPos(size.getRows() - 1, size.getColumns() - 1));
            terminal.flush();

            if (debugThisRefresh) {
                System.err.println("[ConsolePanelSink] Display update completed");
            }

        } catch (Exception e) {
            // Log error but continue
            System.err.println("[ConsolePanelSink] Display update error: " + e.getMessage());
        }
    }


    // New compact rendering methods
    private void renderCompactHeader(List<AttributedString> lines, int width) {
        // Single top border line with title
        lines.add(buildSectionBorder('╔', '╗', "Active Tasks", width));
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
                int startIdx = Math.max(0, totalLogs - contentHeight);
                if (isUserScrollingLogs) {
                    startIdx = Math.min(logScrollOffset, Math.max(0, totalLogs - contentHeight));
                }

                // Most common case: showing recent logs (at or near the end)
                // Use descending iterator and collect the needed lines
                if (startIdx >= totalLogs - contentHeight * 2) {
                    // We're close to the end, use descending iterator
                    Iterator<String> descIter = logBuffer.descendingIterator();
                    List<String> tempLines = new ArrayList<>();

                    // Skip the newest lines we don't need
                    int toSkip = totalLogs - startIdx - contentHeight;
                    for (int i = 0; i < toSkip && descIter.hasNext(); i++) {
                        descIter.next();
                    }

                    // Collect the lines we need (in reverse order)
                    for (int i = 0; i < contentHeight && descIter.hasNext(); i++) {
                        tempLines.add(descIter.next());
                    }

                    // Reverse to get correct order and process
                    Collections.reverse(tempLines);
                    for (String line : tempLines) {
                        String logLine = fitLine(line, innerWidth);
                        AttributedStyle style = getLogStyle(logLine);
                        result.add(new AttributedStringBuilder().style(style).append(logLine).toAttributedString());
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
                        String logLine = fitLine(line, innerWidth);
                        AttributedStyle style = getLogStyle(logLine);
                        result.add(new AttributedStringBuilder().style(style).append(logLine).toAttributedString());
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

    private void renderStatusLine(List<AttributedString> lines, int width) {
        AttributedStringBuilder statusBar = new AttributedStringBuilder();

        // Count active tasks
        long activeTasks = taskNodes.values().stream()
                .filter(n -> n.finishTime == 0)
                .count();
        long completedTasks = taskNodes.values().stream()
                .filter(n -> n.finishTime > 0)
                .count();

        // Build compact status line
        statusBar.style(AttributedStyle.DEFAULT.foreground(AttributedStyle.CYAN))
                .append(" Active: " + activeTasks)
                .append(" | ")
                .append("Done: " + completedTasks)
                .append(" | ")
                .append("Logs: " + logBuffer.size())
                .append(" | ")
                .append("↑↓: scroll, q: quit");

        // Pad to width
        int currentLen = statusBar.toAttributedString().columnLength();
        if (currentLen < width) {
            statusBar.append(" ".repeat(width - currentLen));
        }

        lines.add(statusBar.toAttributedString());
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
                if (node.lastStatus != null && node.lastStatus.runstate == StatusUpdate.RunState.FAILED) {
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

    private void collectTaskLines(TaskNode node, List<SectionLine> lines, String prefix, boolean isLast, int innerWidth) {
        if (node.tracker != null) {
            String taskLine = formatTaskLine(node, prefix, isLast, innerWidth);
            taskLine = fitLine(taskLine, innerWidth);
            lines.add(new SectionLine(taskLine, AttributedStyle.DEFAULT));
        }

        List<TaskNode> children = new ArrayList<>(node.children);
        for (int i = 0; i < children.size(); i++) {
            TaskNode child = children.get(i);
            boolean childIsLast = (i == children.size() - 1);

            String childPrefix = prefix;
            if (node.tracker != null) {
                childPrefix += isLast ? "  " : "│ ";
            }

            collectTaskLines(child, lines, childPrefix, childIsLast, innerWidth);
        }
    }

    private String formatTaskLine(TaskNode node, String prefix, boolean isLast, int availableWidth) {
        StringBuilder line = new StringBuilder();

        // Tree connector
        if (!prefix.isEmpty()) {
            line.append(prefix);
            line.append(isLast ? "└─ " : "├─ ");
        }

        // Status icon
        String statusIcon = getStatusIcon(node);
        line.append(statusIcon).append(" ");

        // Task name
        String taskName = getTaskName(node.tracker);
        line.append(taskName);

        // Build the right-aligned portion (duration, then progress bar with percentage)
        StringBuilder rightPortion = new StringBuilder();

        // Duration first - use elapsed running time if task is/was running
        long duration;
        if (node.tracker != null && node.tracker.getRunningStartTime() != null) {
            // Task has started running - use actual running time
            if (node.finishTime > 0) {
                duration = node.finishTime - node.tracker.getRunningStartTime();
            } else {
                duration = node.tracker.getElapsedRunningTime();
            }
        } else if (node.lastStatus != null && node.lastStatus.runstate == StatusUpdate.RunState.PENDING) {
            // Task hasn't started yet
            duration = 0;
        } else {
            // Fallback to old calculation
            duration = (node.finishTime > 0 ? node.finishTime : System.currentTimeMillis()) - node.startTime;
        }
        rightPortion.append(String.format(" (%.1fs) ", duration / 1000.0));

        // Progress bar with percentage centered in it (fixed 22 characters total)
        if (node.lastStatus != null) {
            String progressBarWithPercent = createProgressBarWithCenteredPercent(node.lastStatus.progress);
            rightPortion.append(progressBarWithPercent);
        } else {
            rightPortion.append("[        0%          ]");
        }

        // Completion marker
        if (node.finishTime > 0 && node.lastStatus != null) {
            if (node.lastStatus.runstate == StatusUpdate.RunState.SUCCESS) {
                rightPortion.append(" ✓");
            } else if (node.lastStatus.runstate == StatusUpdate.RunState.FAILED) {
                rightPortion.append(" ✗");
            }
        }

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
                if (node.lastStatus.runstate == StatusUpdate.RunState.PENDING) {
                    context.append(" [pending]");
                } else if (node.lastStatus.runstate == StatusUpdate.RunState.RUNNING) {
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

    private int countTaskLines(TaskNode node) {
        int count = node.tracker != null ? 1 : 0;
        for (TaskNode child : node.children) {
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
                .filter(n -> n.lastStatus != null && n.lastStatus.runstate == StatusUpdate.RunState.RUNNING)
                .count();
        long completedTasks = taskNodes.values().stream()
                .filter(n -> n.lastStatus != null && n.lastStatus.runstate == StatusUpdate.RunState.SUCCESS)
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

    private String getStatusIcon(TaskNode node) {
        if (node.lastStatus == null) {
            return "○"; // Pending
        }

        switch (node.lastStatus.runstate) {
            case PENDING:
                return "○";
            case RUNNING:
                return "▶";
            case SUCCESS:
                return "●";
            case FAILED:
                return "✗";
            case CANCELLED:
                return "◼";
            default:
                return "?";
        }
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

    private String getTaskName(StatusTracker<?> tracker) {
        Object tracked = tracker.getTracked();
        try {
            java.lang.reflect.Method getNameMethod = tracked.getClass().getMethod("getName");
            Object name = getNameMethod.invoke(tracked);
            if (name instanceof String) {
                return (String) name;
            }
        } catch (Exception ignored) {
        }

        String fullName = tracked.toString();
        // Truncate if too long
        if (fullName.length() > 20) {
            return fullName.substring(0, 17) + "...";
        }
        return fullName;
    }

    public boolean isClosed() {
        return closed.get();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            // Clear the active sink
            LogBuffer.clearActiveSink();

            // Restore original streams
            System.setOut(originalOut);
            System.setErr(originalErr);

            // Stop the render thread
            try {
                renderThread.join(2000); // Wait up to 2 seconds for clean shutdown
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            try {
                // Clear the display and restore terminal
                display.update(Collections.emptyList(), 0);

                // Exit raw mode to restore normal terminal behavior
                terminal.getAttributes().setLocalFlag(org.jline.terminal.Attributes.LocalFlag.ICANON, true);
                terminal.getAttributes().setLocalFlag(org.jline.terminal.Attributes.LocalFlag.ECHO, true);
                terminal.setAttributes(terminal.getAttributes());

                terminal.flush();
                terminal.close();
            } catch (Exception e) {
                // Ignore close errors
            }
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
                    if (logScrollOffset > 0) {
                        logScrollOffset--;
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

    /**
     * Internal class representing a task node in the hierarchy
     */
    private static class TaskNode {
        final StatusTracker<?> tracker;
        final TaskNode parent;
        final List<TaskNode> children;
        final long startTime;
        StatusUpdate<?> lastStatus;
        long lastUpdateTime;
        long finishTime;

        TaskNode(StatusTracker<?> tracker, TaskNode parent) {
            this.tracker = tracker;
            this.parent = parent;
            this.children = Collections.synchronizedList(new ArrayList<>());
            this.startTime = System.currentTimeMillis();
            this.lastUpdateTime = startTime;
            this.finishTime = 0;
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
     * Builder for configuring ConsolePanelSink
     */
    public static class Builder {
        private long refreshRateMs = 100;
        private long completedRetentionMs = 5000;
        private boolean useColors = true;
        private int maxLogLines = 1000;
        private boolean captureSystemStreams = false;

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


        public ConsolePanelSink build() {
            return new ConsolePanelSink(this);
        }
    }
}
