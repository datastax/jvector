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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/// Facade for locating datasets across multiple {@link DataSetLoader} implementations and
/// layering {@link DataSetWrapper}s over what they load.
///
/// Returns a {@link DataSetInfo} handle whose vector data is loaded lazily on the first
/// call to {@link DataSetInfo#getDataSet()}, allowing callers to inspect dataset metadata
/// (name, similarity function) without incurring the cost of reading vectors into memory.
///
/// ### Wrappers
/// Every dataset name accepted here may be a {@link DataSetSpec} in sugared form, e.g.
/// `cohere-english-v3-100k(mmap)` or `cohere-english-v3-100k(mmap,lru[grain=4096,capacityMb=512])`.
/// Symbolic wrappers are resolved through {@link #wrapperProviders}, which hands each wrapper its
/// bracketed (or, in YAML, mapped) options. When a spec names no wrappers,
/// {@link #defaultWrappers} apply, which cache base vectors in heap memory so that datasets
/// behave as they did when loaders read everything into lists. Datasets larger than memory name
/// `mmap` (serve straight from the mapped file) or `lru` (a bounded grain cache in front of it)
/// instead, which replaces that default. Passing wrapper providers
/// explicitly replaces both; a spec that also names wrappers is rejected in that case.
///
/// @see DataSetInfo
/// @see DataSetLoader
/// @see DataSetSpec
public class DataSets {
    private static final Logger logger = LoggerFactory.getLogger(DataSets.class);

    public static final List<DataSetLoader> defaultLoaders = new ArrayList<>() {{

        /// Scans the jvector-examples/yaml-configs/dataset-catalogs/ directory for .yaml/.yml files.
        ///
        /// To add your own datasets:
        /// 1. Add a .yaml file with your dataset mappings (see local-catalog.yaml for examples)
        /// 2. For private remote datasets, use baseurl with ${SECRET_HASH} style env vars
        ///
        add(new DataSetLoaderSimpleMFD("jvector-examples/yaml-configs/dataset-catalogs"));

    }};

    /// Symbolic wrapper names, as used in `name(wrapper,...)` and structured `wrappers:` lists,
    /// mapped to the factory that builds each from its options. Additional wrappers may be
    /// registered here; a wrapper without options registers via {@link DataSetWrapper.Factory#optionless}.
    public static final Map<String, DataSetWrapper.Factory> wrapperProviders = new LinkedHashMap<>() {{
        put(InMemoryCachedDataSet.WRAPPER_NAME, InMemoryCachedDataSet.FACTORY);
        put(MMapCachedDataSet.WRAPPER_NAME, MMapCachedDataSet.FACTORY);
        put(LruCachedDataSet.WRAPPER_NAME, LruCachedDataSet.FACTORY);
    }};

    /// Wrappers applied when a dataset spec names none: the base vectors are cached in heap memory.
    public static final List<DataSetWrapper.Provider> defaultWrappers = new ArrayList<>(List.of(InMemoryCachedDataSet.PROVIDER));

    /// Loads a dataset by name or sugared spec using the {@link #defaultLoaders} and either the
    /// spec's wrappers or the {@link #defaultWrappers}.
    ///
    /// @param dataSetName the logical dataset name (e.g. {@code "ada002-100k"}), optionally with profile and wrappers
    /// @return a lazy {@link DataSetInfo} handle, or empty if no loader recognises the name
    public static Optional<DataSetInfo> loadDataSet(String dataSetName) {
        return loadDataSet(DataSetSpec.parse(dataSetName));
    }

    /// Loads a dataset by spec using the {@link #defaultLoaders} and either the spec's wrappers or
    /// the {@link #defaultWrappers}.
    ///
    /// @param spec the dataset name, profile, and wrapper names
    /// @return a lazy {@link DataSetInfo} handle, or empty if no loader recognises the name
    public static Optional<DataSetInfo> loadDataSet(DataSetSpec spec) {
        return loadDataSet(spec, defaultLoaders);
    }

    /// Loads a dataset by name or sugared spec, trying each loader in order until one matches, and
    /// applying either the spec's wrappers or the {@link #defaultWrappers}.
    ///
    /// @param dataSetName the logical dataset name (e.g. {@code "ada002-100k"}), optionally with profile and wrappers
    /// @param loaders     the loaders to try, in priority order
    /// @return a lazy {@link DataSetInfo} handle, or empty if no loader recognises the name
    public static Optional<DataSetInfo> loadDataSet(String dataSetName, Collection<DataSetLoader> loaders) {
        return loadDataSet(DataSetSpec.parse(dataSetName), loaders);
    }

    /// Loads a dataset by spec, trying each loader in order until one matches, and applying either
    /// the spec's wrappers (resolved through {@link #wrapperProviders}) or the {@link #defaultWrappers}.
    ///
    /// @param spec    the dataset name, profile, and wrapper names
    /// @param loaders the loaders to try, in priority order
    /// @return a lazy {@link DataSetInfo} handle, or empty if no loader recognises the name
    /// @throws IllegalArgumentException if the spec names a wrapper that is not registered
    public static Optional<DataSetInfo> loadDataSet(DataSetSpec spec, Collection<DataSetLoader> loaders) {
        List<DataSetWrapper.Provider> wrappers = spec.hasWrappers() ? resolveWrappers(spec.getWrappers()) : defaultWrappers;
        return loadDataSet(new DataSetSpec(spec.getName(), spec.getProfile(), null), loaders, wrappers);
    }

    /// Loads a dataset by name, trying each loader in order until one matches, then applies exactly
    /// the given wrapper providers, in order, when the dataset is first materialised.
    ///
    /// @param dataSetName the logical dataset name, optionally with a profile; it must not name wrappers
    /// @param loaders     the loaders to try, in priority order
    /// @param wrappers    the wrappers to layer over the loaded dataset, outermost last; may be empty
    /// @return a lazy {@link DataSetInfo} handle, or empty if no loader recognises the name
    public static Optional<DataSetInfo> loadDataSet(String dataSetName,
                                                    Collection<DataSetLoader> loaders,
                                                    Collection<DataSetWrapper.Provider> wrappers) {
        return loadDataSet(DataSetSpec.parse(dataSetName), loaders, wrappers);
    }

    /// Loads a dataset by spec, trying each loader in order until one matches, then applies exactly
    /// the given wrapper providers, in order, when the dataset is first materialised.
    ///
    /// @param spec     the dataset name and profile; it must not name wrappers, since `wrappers` replaces them
    /// @param loaders  the loaders to try, in priority order
    /// @param wrappers the wrappers to layer over the loaded dataset, outermost last; may be empty
    /// @return a lazy {@link DataSetInfo} handle, or empty if no loader recognises the name
    /// @throws IllegalArgumentException if the spec names wrappers as well
    public static Optional<DataSetInfo> loadDataSet(DataSetSpec spec,
                                                    Collection<DataSetLoader> loaders,
                                                    Collection<DataSetWrapper.Provider> wrappers) {
        if (spec.hasWrappers()) {
            throw new IllegalArgumentException("Dataset spec '" + spec + "' names wrappers, but wrapper providers were also given explicitly; use one or the other");
        }
        logger.info("loading dataset [{}]", spec);
        if (spec.getName().endsWith(".hdf5")) {
            throw new InvalidParameterException("DataSet names are not meant to be file names. Did you mean " + spec.getName().replace(".hdf5", "") + "? ");
        }

        for (DataSetLoader loader : loaders) {
            logger.trace("trying loader [{}]", loader.getClass().getSimpleName());
            Optional<DataSetInfo> dataSetLoaded = loader.loadDataSet(spec);
            if (dataSetLoaded.isPresent()) {
                logger.info("dataset [{}] found with loader [{}]", spec, loader.getClass().getSimpleName());
                return Optional.of(wrap(dataSetLoaded.get(), wrappers));
            }
        }
        logger.warn("Unable to find dataset [{}] with any dataset loader.", spec);
        return Optional.empty();
    }

    /// Resolves symbolic wrappers through {@link #wrapperProviders}, preserving order and handing
    /// each wrapper's options to its factory.
    ///
    /// @param wrappers wrappers as written in a spec
    /// @return the configured providers
    /// @throws IllegalArgumentException if a name is not registered or a factory rejects its options
    public static List<DataSetWrapper.Provider> resolveWrappers(List<DataSetSpec.WrapperSpec> wrappers) {
        List<DataSetWrapper.Provider> providers = new ArrayList<>(wrappers.size());
        for (DataSetSpec.WrapperSpec wrapper : wrappers) {
            DataSetWrapper.Factory factory = wrapperProviders.get(wrapper.getName());
            if (factory == null) {
                throw new IllegalArgumentException("Unknown dataset wrapper '" + wrapper.getName() + "'; known wrappers: " + wrapperProviders.keySet());
            }
            providers.add(factory.provider(wrapper.getOptions()));
        }
        return providers;
    }

    private static DataSetInfo wrap(DataSetInfo info, Collection<DataSetWrapper.Provider> wrappers) {
        if (wrappers.isEmpty()) {
            return info;
        }
        List<DataSetWrapper.Provider> providers = List.copyOf(wrappers);
        return new DataSetInfo(info, () -> {
            DataSet ds = info.getDataSet();
            for (DataSetWrapper.Provider provider : providers) {
                ds = provider.wrap(ds);
            }
            return ds;
        });
    }
}
