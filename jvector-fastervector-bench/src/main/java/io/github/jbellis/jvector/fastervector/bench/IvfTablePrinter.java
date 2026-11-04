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

package io.github.jbellis.jvector.fastervector.bench;

import io.github.jbellis.jvector.example.benchmarks.Metric;

import java.io.PrintStream;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Console table for IVF results, laid out like jvector-examples' {@code BenchmarkTablePrinter},
 * whose first column is fixed to the graph's overquery. Here the leading columns are
 * {@code nprobe} and {@code Threads}, then one column per metric. A new header is printed
 * after {@link #resetTable()}, for example when topK, and so the metric columns, change.
 */
final class IvfTablePrinter {
    private static final int MIN_COLUMN_WIDTH = 11;
    private static final int MIN_HEADER_PADDING = 3;
    private static final int MAX_COLUMN_WIDTH = 15;
    private static final int KEY_COLUMN_WIDTH = 8;

    private final PrintStream out;
    private String headerFmt;
    private String rowFmt;

    IvfTablePrinter(PrintStream out) {
        this.out = out;
    }

    void resetTable() {
        headerFmt = null;
        rowFmt = null;
    }

    void printRow(int nprobe, int threads, List<Metric> metrics) {
        if (headerFmt == null) {
            printHeader(metrics);
        }
        Object[] values = new Object[metrics.size() + 2];
        values[0] = nprobe;
        values[1] = threads;
        for (int i = 0; i < metrics.size(); i++) {
            values[i + 2] = metrics.get(i).getValue();
        }
        out.printf(Locale.US, rowFmt, values);
    }

    void printFooter() {
        out.println();
    }

    private void printHeader(List<Metric> metrics) {
        StringBuilder header = new StringBuilder("%-" + KEY_COLUMN_WIDTH + "s %-" + KEY_COLUMN_WIDTH + "s");
        StringBuilder row = new StringBuilder("%-" + KEY_COLUMN_WIDTH + "d %-" + KEY_COLUMN_WIDTH + "d");
        Object[] line1 = new Object[metrics.size() + 2];
        Object[] line2 = new Object[metrics.size() + 2];
        line1[0] = "nprobe";
        line1[1] = "Threads";
        line2[0] = "";
        line2[1] = "";
        boolean secondLine = false;
        for (int i = 0; i < metrics.size(); i++) {
            Metric m = metrics.get(i);
            int width = Math.min(Math.max(MIN_COLUMN_WIDTH, m.getHeader().length() + MIN_HEADER_PADDING), MAX_COLUMN_WIDTH);
            header.append(" %-").append(width).append('s');
            row.append(" %-").append(width).append(m.getFmtSpec());
            String[] parts = splitHeader(m.getHeader(), width);
            line1[i + 2] = parts[0];
            line2[i + 2] = parts[1];
            secondLine |= !parts[1].isEmpty();
        }
        headerFmt = header.toString();
        rowFmt = row.append("%n").toString();

        out.println();
        String first = String.format(Locale.US, headerFmt, line1);
        out.println(first);
        if (secondLine) {
            out.println(String.format(Locale.US, headerFmt, line2));
        }
        out.println(String.join("", Collections.nCopies(first.length(), "-")));
    }

    /** Splits at an explicit newline, else at the last space that fits, as BenchmarkTablePrinter does. */
    static String[] splitHeader(String header, int width) {
        int nl = header.indexOf('\n');
        if (nl >= 0) {
            return new String[]{header.substring(0, nl).trim(), header.substring(nl + 1).trim()};
        }
        int max = Math.max(1, width - MIN_HEADER_PADDING);
        String s = header.trim();
        if (s.length() <= max) {
            return new String[]{s, ""};
        }
        int cut = s.lastIndexOf(' ', max);
        if (cut <= 0) {
            cut = max;
        }
        return new String[]{s.substring(0, cut).trim(), s.substring(cut).trim()};
    }
}
