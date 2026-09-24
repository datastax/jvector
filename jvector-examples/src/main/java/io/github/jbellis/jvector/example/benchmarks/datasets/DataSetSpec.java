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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/// Identifies a dataset to load: its catalog name, an optional loader profile, and an optional
/// list of symbolic {@link DataSetWrapper}s, each with optional options, to apply after loading.
///
/// ### Sugared form
/// A single string, parsed by {@link #parse(String)}:
/// ```
/// cohere                          name only; profile "default", no wrappers
/// cohere:fast                     name and profile
/// cohere(mmap)                    name and wrappers
/// cohere:default(mmap)            all three (":default" is equivalent to omitting the profile)
/// cohere(mmap, lru[grain=4096,capacityMb=512])
///                                 several wrappers, applied left to right; options in brackets
/// ```
/// The profile always precedes the parenthesised wrapper list, so `name:profile` never collides
/// with the wrapper syntax. Option keys and values may not contain `,`, `=`, `[` or `]`.
///
/// ### Structured form
/// A YAML map, accepted by {@link #from(Object)} wherever a dataset list entry may appear. Each
/// wrapper is a plain name, a map keyed by the wrapper name whose value holds its options, or a
/// map with a `name` key and the options alongside:
/// ```yaml
/// - name: cohere
///   profile: default            # optional
///   wrappers:                   # optional
///     - mmap
///     - lru: { grain: 4096, capacityMb: 512 }
///     - name: lru
///       grain: 4096
/// ```
///
/// A missing profile is always reported as {@value #DEFAULT_PROFILE}. Loaders that do not
/// understand profiles accept the default profile and reject any other; see
/// {@link DataSetLoader#loadDataSet(DataSetSpec)}. {@link #toString()} renders the canonical
/// sugared form, which round-trips through {@link #parse(String)} including wrapper options.
public final class DataSetSpec {
    /// The profile assumed when none is given.
    public static final String DEFAULT_PROFILE = "default";

    private static final Pattern SUGAR = Pattern.compile(
            "^\\s*([^:()\\s\\[\\]][^:()\\[\\]]*?)\\s*(?::\\s*([^:()\\[\\]]+?)\\s*)?(?:\\(([^()]*)\\)\\s*)?$");
    private static final Pattern WRAPPER_TOKEN = Pattern.compile(
            "^\\s*([^\\[\\],=\\s]+)\\s*(?:\\[([^\\[\\]]*)\\]\\s*)?$");

    /// One wrapper to apply: its registered name and its options as written.
    public static final class WrapperSpec {
        private final String name;
        private final Map<String, String> options;

        /// @param name    the registered wrapper name; must not be blank
        /// @param options option key/value pairs, or null for none; values are kept as strings
        public WrapperSpec(String name, Map<String, ?> options) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("Wrapper name must not be blank");
            }
            this.name = name.trim();
            Map<String, String> copy = new LinkedHashMap<>();
            if (options != null) {
                for (var e : options.entrySet()) {
                    String key = e.getKey() == null ? "" : e.getKey().trim();
                    if (key.isEmpty()) {
                        throw new IllegalArgumentException("Wrapper '" + this.name + "' has an option with an empty key");
                    }
                    String value = e.getValue() == null ? "" : String.valueOf(e.getValue()).trim();
                    for (String forbidden : new String[] {",", "=", "[", "]"}) {
                        if (key.contains(forbidden) || value.contains(forbidden)) {
                            throw new IllegalArgumentException("Wrapper '" + this.name + "' option '" + key + "=" + value
                                    + "' may not contain '" + forbidden + "'");
                        }
                    }
                    copy.put(key, value);
                }
            }
            this.options = Collections.unmodifiableMap(copy);
        }

        /// Parses a token such as `lru` or `lru[grain=4096,capacityMb=512]`.
        static WrapperSpec parse(String token) {
            Matcher m = WRAPPER_TOKEN.matcher(token);
            if (!m.matches()) {
                throw new IllegalArgumentException("Malformed wrapper '" + token.trim() + "'; expected name or name[key=value,...]");
            }
            Map<String, String> options = new LinkedHashMap<>();
            if (m.group(2) != null && !m.group(2).isBlank()) {
                for (String pair : m.group(2).split(",")) {
                    int eq = pair.indexOf('=');
                    if (eq < 0) {
                        throw new IllegalArgumentException("Malformed wrapper option '" + pair.trim() + "' in '" + token.trim() + "'; expected key=value");
                    }
                    options.put(pair.substring(0, eq).trim(), pair.substring(eq + 1).trim());
                }
            }
            return new WrapperSpec(m.group(1), options);
        }

        /// @return the registered wrapper name
        public String getName() {
            return name;
        }

        /// @return the options as written, in order; empty when none were given
        public Map<String, String> getOptions() {
            return options;
        }

        /// @return `name` or `name[key=value,...]`
        @Override
        public String toString() {
            if (options.isEmpty()) {
                return name;
            }
            StringBuilder sb = new StringBuilder(name).append('[');
            boolean first = true;
            for (var e : options.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                sb.append(e.getKey()).append('=').append(e.getValue());
            }
            return sb.append(']').toString();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof WrapperSpec)) return false;
            WrapperSpec that = (WrapperSpec) o;
            return name.equals(that.name) && options.equals(that.options);
        }

        @Override
        public int hashCode() {
            return Objects.hash(name, options);
        }
    }

    private final String name;
    private final String profile;
    private final List<WrapperSpec> wrappers;

    /// Creates a spec. Blank profile means {@value #DEFAULT_PROFILE}.
    ///
    /// @param name     the dataset name; must not be blank
    /// @param profile  the loader profile, or null for the default
    /// @param wrappers wrappers in application order, or null for none
    public DataSetSpec(String name, String profile, List<WrapperSpec> wrappers) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Dataset name must not be blank");
        }
        this.name = name.trim();
        this.profile = (profile == null || profile.isBlank()) ? DEFAULT_PROFILE : profile.trim();
        this.wrappers = wrappers == null ? List.of() : List.copyOf(wrappers);
    }

    /// Parses the sugared string form described in the class documentation.
    ///
    /// @param sugared e.g. `cohere`, `cohere:fast`, `cohere(mmap)`, or `cohere:fast(mmap, lru[grain=4096])`
    /// @return the parsed spec
    /// @throws IllegalArgumentException if the string is not in the sugared form
    public static DataSetSpec parse(String sugared) {
        if (sugared == null) {
            throw new IllegalArgumentException("Dataset spec must not be null");
        }
        Matcher m = SUGAR.matcher(sugared);
        if (!m.matches()) {
            throw new IllegalArgumentException("Malformed dataset spec '" + sugared
                    + "'; expected name, name:profile, name(wrapper,...), or name:profile(wrapper,...)");
        }
        return new DataSetSpec(m.group(1), m.group(2), parseWrapperList(m.group(3)));
    }

    /// Splits a wrapper list on the commas that are not inside `[...]` and parses each token.
    static List<WrapperSpec> parseWrapperList(String list) {
        List<WrapperSpec> wrappers = new ArrayList<>();
        if (list == null) {
            return wrappers;
        }
        int depth = 0;
        int start = 0;
        for (int i = 0; i <= list.length(); i++) {
            char c = i < list.length() ? list.charAt(i) : ',';
            if (c == '[') {
                depth++;
            } else if (c == ']') {
                depth--;
                if (depth < 0) {
                    throw new IllegalArgumentException("Unbalanced ']' in wrapper list '" + list + "'");
                }
            } else if (c == ',' && depth == 0) {
                String token = list.substring(start, i);
                if (!token.isBlank()) {
                    wrappers.add(WrapperSpec.parse(token));
                }
                start = i + 1;
            }
        }
        if (depth != 0) {
            throw new IllegalArgumentException("Unbalanced '[' in wrapper list '" + list + "'");
        }
        return wrappers;
    }

    /// Converts a dataset list entry from YAML: a string in sugared form, or a map with `name`
    /// and optional `profile` and `wrappers` keys. `wrappers` may be a list or a single string;
    /// each list element is a wrapper name (optionally with bracketed options), a map keyed by the
    /// wrapper name whose value is its option map, or a map with a `name` key and options alongside.
    ///
    /// @param item the YAML value
    /// @return the spec
    /// @throws IllegalArgumentException if the value is neither form, or the map lacks a name
    @SuppressWarnings("unchecked")
    public static DataSetSpec from(Object item) {
        if (item instanceof DataSetSpec) {
            return (DataSetSpec) item;
        }
        if (item instanceof String) {
            return parse((String) item);
        }
        if (item instanceof Map) {
            Map<String, Object> map = (Map<String, Object>) item;
            Object name = map.get("name");
            if (name == null) {
                throw new IllegalArgumentException("Structured dataset entry is missing 'name': " + map);
            }
            for (String key : map.keySet()) {
                if (!key.equals("name") && !key.equals("profile") && !key.equals("wrappers")) {
                    throw new IllegalArgumentException("Unknown key '" + key + "' in dataset entry " + map
                            + "; known keys: name, profile, wrappers");
                }
            }
            Object profile = map.get("profile");
            Object wrappers = map.get("wrappers");
            List<WrapperSpec> wrapperSpecs;
            if (wrappers == null) {
                wrapperSpecs = List.of();
            } else if (wrappers instanceof String) {
                wrapperSpecs = parseWrapperList((String) wrappers);
            } else if (wrappers instanceof List) {
                wrapperSpecs = new ArrayList<>();
                for (Object w : (List<Object>) wrappers) {
                    wrapperSpecs.add(wrapperFrom(w));
                }
            } else {
                throw new IllegalArgumentException("'wrappers' must be a list or string in dataset entry " + map);
            }
            return new DataSetSpec(name.toString(), profile == null ? null : profile.toString(), wrapperSpecs);
        }
        throw new IllegalArgumentException("Dataset entry must be a string or a map, got: " + item);
    }

    @SuppressWarnings("unchecked")
    private static WrapperSpec wrapperFrom(Object item) {
        if (item instanceof String) {
            return WrapperSpec.parse((String) item);
        }
        if (item instanceof Map) {
            Map<String, Object> map = (Map<String, Object>) item;
            Object name = map.get("name");
            if (name != null) {
                Map<String, Object> options = new LinkedHashMap<>(map);
                options.remove("name");
                return new WrapperSpec(name.toString(), options);
            }
            if (map.size() == 1) {
                var entry = map.entrySet().iterator().next();
                Object value = entry.getValue();
                if (value == null) {
                    return new WrapperSpec(entry.getKey(), null);
                }
                if (value instanceof Map) {
                    return new WrapperSpec(entry.getKey(), (Map<String, Object>) value);
                }
                throw new IllegalArgumentException("Options for wrapper '" + entry.getKey() + "' must be a map, got: " + value);
            }
            throw new IllegalArgumentException("Wrapper entry must be a name, {name: options-map}, or {name: ..., option: value}; got: " + map);
        }
        throw new IllegalArgumentException("Wrapper entry must be a string or a map, got: " + item);
    }

    /// @return the dataset name as known to the loaders
    public String getName() {
        return name;
    }

    /// @return the loader profile; {@value #DEFAULT_PROFILE} when none was given
    public String getProfile() {
        return profile;
    }

    /// @return true iff the profile is {@value #DEFAULT_PROFILE}
    public boolean isDefaultProfile() {
        return DEFAULT_PROFILE.equals(profile);
    }

    /// @return wrappers in application order, with their options; empty when none were given
    public List<WrapperSpec> getWrappers() {
        return wrappers;
    }

    /// @return true iff at least one wrapper was given
    public boolean hasWrappers() {
        return !wrappers.isEmpty();
    }

    /// @return the canonical sugared form: the name, `:profile` unless default, and `(w1,w2[k=v])` if any wrappers
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(name);
        if (!isDefaultProfile()) {
            sb.append(':').append(profile);
        }
        if (hasWrappers()) {
            sb.append('(');
            for (int i = 0; i < wrappers.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(wrappers.get(i));
            }
            sb.append(')');
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DataSetSpec)) return false;
        DataSetSpec that = (DataSetSpec) o;
        return name.equals(that.name) && profile.equals(that.profile) && wrappers.equals(that.wrappers);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, profile, wrappers);
    }
}
