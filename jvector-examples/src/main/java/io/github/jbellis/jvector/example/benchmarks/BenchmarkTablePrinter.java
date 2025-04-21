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

package io.github.jbellis.jvector.example.benchmarks;

import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Prints a two‐dimensional table:
 *  - First column is the Overquery value (double)
 *  - Subsequent columns are defined by Metric tuples
 */
public class BenchmarkTablePrinter {
    private static final int MIN_COLUMN_WIDTH     = 11;
    private static final int MIN_HEADER_PADDING   = 3;

    private String headerFmt;
    private String rowFmt;

    public BenchmarkTablePrinter() {
        headerFmt = null;
        rowFmt = null;
    }


    private void initializeHeader(List<Metric> cols) {
        if (headerFmt != null) {
            return;
        }

        // Build the format strings for header & rows
        StringBuilder hsb = new StringBuilder();
        StringBuilder rsb = new StringBuilder();

        // 1) Overquery column width
        hsb.append("%-12s");
        rsb.append("%-12.2f");

        // 2) One column per Metric
        for (Metric m : cols) {
            String hdr = m.getHeader();
            String spec = m.getFmtSpec();
            int width = Math.max(MIN_COLUMN_WIDTH, hdr.length() + MIN_HEADER_PADDING);

            // Header: Always a string
            hsb.append(" %-").append(width).append("s");
            // Row: Use the Metric’s fmtSpec (e.g. ".2f", ".3f")
            rsb.append(" %-").append(width).append(spec);
        }

        this.headerFmt = hsb.toString();
        this.rowFmt = rsb.append("%n").toString();

        System.out.println();
        printHeader(cols);
    }

    /**
     * Call this once to print all the global parameters before the table.
     *
     * @param params a map from parameter name (e.g. "mGrid") to its List value
     */
    public void printConfig(Map<String, ?> params) {
        System.out.println();
        System.out.println("Configuration:");
        params.forEach((name, values) ->
                System.out.printf(Locale.US, "  %-22s: %s%n", name, values)
        );
    }

    private void printHeader(List<Metric> cols) {
        // Prepare array: First "Overquery", then each Metric header
        Object[] hdrs = new Object[cols.size() + 1];
        hdrs[0] = "Overquery";
        for (int i = 0; i < cols.size(); i++) {
            hdrs[i + 1] = cols.get(i).getHeader();
        }

        // Print header line
        String line = String.format(Locale.US, headerFmt, hdrs);
        System.out.println(line);
        // Underline of same length
        System.out.println(String.join("",
                Collections.nCopies(line.length(), "-")
        ));
    }

    /**
     * Print a row of data.
     *
     * @param overquery the first‐column value
     * @param cols list of metrics to print
     */
    public void printRow(double overquery,
                         List<Metric> cols) {
        initializeHeader(cols);

        // Build argument array: First overquery, then each Metric.extract(...)
        Object[] vals = new Object[cols.size() + 1];
        vals[0] = overquery;
        for (int i = 0; i < cols.size(); i++) {
            vals[i + 1] = cols.get(i).getValue();
        }

        // Print the formatted row
        System.out.printf(Locale.US, rowFmt, vals);
    }

    /**
     * Prints a blank line after the table ends.
     * Must be called manually.
     */
    public void printFooter() {
        System.out.println();
    }
}


