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

package io.github.jbellis.jvector.example.util;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Creates a simple csv file logger based on Apache Commons CSV.
 */
public class CsvLogger implements AutoCloseable {
    private static final String DEFAULT_PREFIX = "bench_log_";
    private static final DateTimeFormatter TS_FMT =
            DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    private final Path outFile;
    private final List<Map.Entry<String,Object>> rowTemplate;
    private CSVPrinter printer;
    private boolean headerWritten = false;

    /**
     * @param filePrefix   prefix for the filename (null/blank will result in default)
     * @param columnNames  list of column header names
     */
    public CsvLogger(String filePrefix, List<String> columnNames) {
        String prefix = (filePrefix == null || filePrefix.isBlank())
                ? DEFAULT_PREFIX
                : filePrefix;
        String ts = LocalDateTime.now().format(TS_FMT);
        this.outFile = Paths.get(prefix + ts + ".csv");

        // build a template with null values
        this.rowTemplate = new ArrayList<>();
        for (String name : columnNames) {
            rowTemplate.add(new AbstractMap.SimpleEntry<>(name, null));
        }
        System.out.println("Creating log file: " + outFile);
    }

    /**
     * Update a column's value by its header name
     * @param key   column header
     * @param value column value
     */
    public void setValue(String key, Object value) {
        for (Map.Entry<String,Object> entry : rowTemplate) {
            if (entry.getKey().equals(key)) {
                entry.setValue(value);
                return;
            }
        }
        throw new IllegalArgumentException("No column named '" + key + "'");
    }

    /**
     * Write a row based on the current template.  Emits header on first call.
     */
    public void log() throws IOException {
        if (!headerWritten) {
            boolean exists = Files.exists(outFile);
            Writer writer = Files.newBufferedWriter(
                    outFile,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND
            );
            List<String> headers = new ArrayList<>();
            for (Map.Entry<String,Object> e : rowTemplate) {
                headers.add(e.getKey());
            }
            CSVFormat fmt = CSVFormat.DEFAULT
                    .withHeader(headers.toArray(new String[0]))
                    .withSkipHeaderRecord(exists);
            printer = new CSVPrinter(writer, fmt);
            headerWritten = true;
        }
        // collect and print current values
        List<Object> values = new ArrayList<>();
        for (Map.Entry<String,Object> e : rowTemplate) {
            values.add(e.getValue());
        }
        printer.printRecord(values);
        printer.flush();
    }

    @Override
    public void close() throws IOException {
        if (printer != null) {
            printer.close();
        }
    }
}