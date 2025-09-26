package io.github.jbellis.jvector.status.asyncdag;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class TaskLayer implements Callable<String> {
    private final String taskName;
    private final int layer;
    private final List<TaskLayer> subTasks;
    private final ExecutorService executor;
    private final long workDurationMillis;
    private final String workResult;

    public TaskLayer(String taskName, int layer, ExecutorService executor) {
        this.taskName = taskName;
        this.layer = layer;
        this.executor = executor;
        this.subTasks = new ArrayList<>();
        this.workDurationMillis = 0;
        this.workResult = null;
    }

    public TaskLayer(String taskName, long workDurationMillis, String workResult) {
        this.taskName = taskName;
        this.layer = 1;
        this.executor = null;
        this.subTasks = new ArrayList<>();
        this.workDurationMillis = workDurationMillis;
        this.workResult = workResult;
    }

    public void addSubTask(TaskLayer task) {
        subTasks.add(task);
    }

    @Override
    public String call() throws Exception {
        System.out.println("TaskLayer" + layer + "[" + taskName + "] starting");

        if (subTasks.isEmpty()) {
            long startTime = System.currentTimeMillis();
            long endTime = startTime + workDurationMillis;
            while (System.currentTimeMillis() < endTime) {
                Thread.sleep(Math.min(100, endTime - System.currentTimeMillis()));
            }
            System.out.println("TaskLayer" + layer + "[" + taskName + "] completed work, returning: " + workResult);
            return workResult;
        } else {
            System.out.println("TaskLayer" + layer + "[" + taskName + "] delegating to " + subTasks.size() + " subtasks");

            List<Future<String>> futures = new ArrayList<>();
            for (TaskLayer task : subTasks) {
                futures.add(executor.submit(task));
            }

            StringBuilder results = new StringBuilder();
            results.append("TaskLayer").append(layer).append("[").append(taskName).append("] results: ");

            for (Future<String> future : futures) {
                String result = future.get();
                results.append(result).append("; ");
            }

            String finalResult = results.toString();
            System.out.println("TaskLayer" + layer + "[" + taskName + "] completed delegation");
            return finalResult;
        }
    }
}