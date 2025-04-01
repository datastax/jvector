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
package io.github.jbellis.jvector.bench.output;

import java.util.ArrayList;
import java.util.List;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;

public class PersistentTextTable implements TableRepresentation {
    private final List<String> resultsTable = new ArrayList<>();
    private static final String DB_URL = "jdbc:sqlite:benchmark_results.db";
    private static final String CREATE_TABLE_SQL = "CREATE TABLE IF NOT EXISTS benchmark_results (" +
            "elapsed_seconds INTEGER," +
            "qps INTEGER," +
            "mean_latency_us REAL," +
            "p999_latency_us REAL," +
            "mean_visited REAL," +
            "recall REAL);";
    private static final String INSERT_SQL = "INSERT INTO benchmark_results " +
        "(elapsed_seconds, qps, mean_latency_us, p999_latency_us, mean_visited, recall) " +
        "VALUES (?, ?, ?, ?, ?, ?);";

    private final Connection connection;
    private final PreparedStatement insertStatement;

    public PersistentTextTable() {
        resultsTable.add("\n");
        resultsTable.add(String.format(
                "%-12s | %-12s | %-18s | %-18s | %-15s | %-12s",
                "Elapsed(s)", "QPS", "Mean Latency (µs)", "P99.9 Latency (µs)", "Mean Visited", "Recall (%)"
        ));
        resultsTable.add("-----------------------------------------------------------------------------------------------");

        try {
            connection = DriverManager.getConnection(DB_URL);
            connection.createStatement().execute(CREATE_TABLE_SQL);
            insertStatement = connection.prepareStatement(INSERT_SQL);
        } catch (SQLException e) {
            throw new RuntimeException("Error initializing SQLite database", e);
        }
    }

    @Override
    public void addEntry(long elapsedSeconds, long qps, double meanLatency, double p999Latency, double meanVisited, double recallPercentage) {
        resultsTable.add(String.format(
                "%-12d | %-12d | %-18.3f | %-18.3f | %-15.3f | %10.2f%%",
                elapsedSeconds, qps, meanLatency, p999Latency, meanVisited, recallPercentage*100
        ));

        try {
            insertStatement.setLong(1, elapsedSeconds);
            insertStatement.setLong(2, qps);
            insertStatement.setDouble(3, meanLatency);
            insertStatement.setDouble(4, p999Latency);
            insertStatement.setDouble(5, meanVisited);
            insertStatement.setDouble(6, recallPercentage*100);
            insertStatement.executeUpdate();
        } catch (SQLException e) {
            throw new RuntimeException("Error inserting benchmark result into SQLite", e);
        }
    }

    @Override
    public void print() {
        for (String line : resultsTable) {
            System.out.println(line);
        }
    }

    @Override
    public void tearDown() {
        try {
            if (insertStatement != null) insertStatement.close();
            if (connection != null) connection.close();
        } catch (SQLException e) {
            throw new RuntimeException("Error closing SQLite database connection", e);
        }
    }
}
