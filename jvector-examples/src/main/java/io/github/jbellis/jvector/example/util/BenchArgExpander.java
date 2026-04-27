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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/// Expands shorthand numeric range patterns in benchArgs tokens.
///
/// A token may embed one or more `[start..end]` ranges, e.g.
/// `sift1m:label_[00..11]` expands to `sift1m:label_00`, `sift1m:label_01`, ... `sift1m:label_11`.
/// The zero-padding width is the max of the two endpoint token lengths, so `[00..11]` pads to 2
/// and `[1..9]` pads to 1. Ranges expand combinatorially when more than one appears in a token,
/// and reverse ranges (`[11..00]`) count down.
public final class BenchArgExpander {

    private static final Pattern RANGE = Pattern.compile("\\[(\\d+)\\.\\.(\\d+)\\]");

    private BenchArgExpander() {}

    /// Expands a single token, returning one or more resulting tokens.
    public static List<String> expand(String token) {
        Matcher m = RANGE.matcher(token);
        if (!m.find()) {
            return List.of(token);
        }
        String startStr = m.group(1);
        String endStr = m.group(2);
        int start = Integer.parseInt(startStr);
        int end = Integer.parseInt(endStr);
        int width = Math.max(startStr.length(), endStr.length());
        String prefix = token.substring(0, m.start());
        String suffix = token.substring(m.end());
        int step = start <= end ? 1 : -1;
        List<String> out = new ArrayList<>();
        for (int i = start; step > 0 ? i <= end : i >= end; i += step) {
            String padded = String.format("%0" + width + "d", i);
            out.addAll(expand(prefix + padded + suffix));
        }
        return out;
    }

    /// Splits each arg on whitespace, drops empties, expands range patterns, and returns the flat list.
    public static String[] expandAll(String[] args) {
        if (args == null) {
            return new String[0];
        }
        List<String> out = new ArrayList<>();
        for (String arg : args) {
            if (arg == null) continue;
            for (String token : arg.split("\\s+")) {
                if (token.isEmpty()) continue;
                out.addAll(expand(token));
            }
        }
        return out.toArray(new String[0]);
    }

    /// Convenience: expand, then filter to tokens for which {@code keep} holds.
    public static String[] expandAll(String[] args, java.util.function.Predicate<String> keep) {
        return Arrays.stream(expandAll(args)).filter(Objects::nonNull).filter(keep).toArray(String[]::new);
    }
}
