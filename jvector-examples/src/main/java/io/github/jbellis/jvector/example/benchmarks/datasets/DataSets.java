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
import java.util.Collection;
import java.util.List;
import java.util.Optional;

public class DataSets {
    public static final List<DataSetLoader> defaultLoaders = new ArrayList<>() {{
        add(new DataSetLoaderHDF5());
        add(new DataSetLoaderMFD());
    }};

    public static Optional<DataSet> loadDataSet(String dataSetName) {
        return loadDataSet(dataSetName, defaultLoaders);
    }

    public static Optional<DataSet> loadDataSet(String dataSetName, Collection<DataSetLoader> loaders) {
        for (DataSetLoader loader : loaders) {
            Optional<DataSet> dataSetLoaded = loader.loadDataSet(dataSetName);
            if (dataSetLoaded.isPresent()) {
                return dataSetLoaded;
            }
        }
        return Optional.empty();
    }
}
