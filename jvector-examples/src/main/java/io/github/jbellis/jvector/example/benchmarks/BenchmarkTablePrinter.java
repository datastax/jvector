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
    private static final int MAX_COLUMN_WIDTH = 15; // tune as desired

    private String headerFmt;
    private String rowFmt;
    private int[] colWidths;

    public BenchmarkTablePrinter() {
        headerFmt = null;
        rowFmt = null;
    }

    /**
     * Clears header/row formats so the next printed row will emit a fresh header.
     * Call this when the set of columns may change (e.g., when topK changes).
     */
    public void resetTable() {
        headerFmt = null;
        rowFmt = null;
        colWidths = null;
    }

    private void initializeHeader(List<Metric> cols) {
        if (headerFmt != null) {
            return;
        }
        this.colWidths = new int[cols.size() + 1];

        // Build the format strings for header & rows
        StringBuilder hsb = new StringBuilder();
        StringBuilder rsb = new StringBuilder();

        // 1) Overquery column width
        // Overquery column width
        hsb.append("%-12s");
        rsb.append("%-12.2f");
        colWidths[0] = 12;

        // 2) One column per Metric
        int i = 0;
        for (Metric m : cols) {
            String hdr = m.getHeader();
            String spec = m.getFmtSpec();
            int width = Math.max(MIN_COLUMN_WIDTH, hdr.length() + MIN_HEADER_PADDING);
            width = Math.min(width, MAX_COLUMN_WIDTH);

            colWidths[i + 1] = width;

            // Header: Always a string
            hsb.append(" %-").append(width).append("s");
            // Row: Use the Metric’s fmtSpec (e.g. ".2f", ".3f")
            rsb.append(" %-").append(width).append(spec);

            i++;
        }

        this.headerFmt = hsb.toString();
        this.rowFmt = rsb.append("%n").toString();

        System.out.println();
        printHeader(cols);
    }

    /**
     * Prints the run-wide configuration header (index settings followed by query settings)
     * once per run, before any results table output.
     *
     * Iteration order is preserved (use an insertion-ordered map such as {@link java.util.LinkedHashMap}).
     *
     * @param indexParams ordered map of index-construction parameters to print
     * @param queryParams ordered map of query/search parameters to print
     */
    public void printConfig(Map<String, ?> indexParams, Map<String, ?> queryParams) {
        printSection("\nIndex configuration", indexParams);
        printSection("\nQuery configuration", queryParams);
    }

    private void printSection(String title, Map<String, ?> params) {
        System.out.println(title + ":");
        for (var e : params.entrySet()) {
            System.out.printf("  %-20s %s%n", e.getKey(), String.valueOf(e.getValue()));
        }
    }

    private void printHeader(List<Metric> cols) {
        // Two header lines: split long headers onto a second line when possible
        Object[] hdrs1 = new Object[cols.size() + 1];
        Object[] hdrs2 = new Object[cols.size() + 1];

        hdrs1[0] = "Overquery";
        hdrs2[0] = "";

        boolean anySecondLine = false;

        for (int i = 0; i < cols.size(); i++) {
            String hdr = cols.get(i).getHeader();
            int width = colWidths[i + 1];

            String[] parts = splitHeader2(hdr, width);
            hdrs1[i + 1] = parts[0];
            hdrs2[i + 1] = parts[1];
            if (!parts[1].isEmpty()) anySecondLine = true;
        }

        String line1 = String.format(Locale.US, headerFmt, hdrs1);
        System.out.println(line1);

        if (anySecondLine) {
            String line2 = String.format(Locale.US, headerFmt, hdrs2);
            System.out.println(line2);
        }

        System.out.println(String.join("", Collections.nCopies(line1.length(), "-")));
    }

    /**
     * Print a row of data.
     *
     * @param overquery the first‐column value
     * @param cols list of metrics to print
     */
    public void printRow(double overquery,
                         List<Metric> cols) {
        initializeHeader(cols); // lazy: prints header on the first row after resetTable()

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

    // Helper for splitting the header into two rows
    private static String[] splitHeader2(String hdr, int colWidth) {
        if (hdr == null) return new String[] { "", "" };

        // Manual break: "Line1\nLine2"
        int nl = hdr.indexOf('\n');
        if (nl >= 0) {
            String a = hdr.substring(0, nl).trim();
            String b = hdr.substring(nl + 1).trim();
            return new String[] { a, b };
        }

        // Leave a little slack for padding/alignment
        int max = Math.max(1, colWidth - MIN_HEADER_PADDING);

        String s = hdr.trim();
        if (s.length() <= max) return new String[] { s, "" };

        // Find last space before max; if none, hard-split
        int cut = s.lastIndexOf(' ', max);
        if (cut <= 0) cut = max;

        String a = s.substring(0, cut).trim();
        String b = s.substring(cut).trim();
        return new String[] { a, b };
    }
}
