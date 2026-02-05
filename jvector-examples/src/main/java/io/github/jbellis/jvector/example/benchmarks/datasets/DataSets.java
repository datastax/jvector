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
import java.util.List;
import java.util.Optional;

public class DataSets {
    private static final Logger logger = LoggerFactory.getLogger(DataSets.class);

    public static final List<DataSetLoader> defaultLoaders = new ArrayList<>() {{
        add(new DataSetLoaderHDF5());
        add(new DataSetLoaderMFD());
    }};

    public static Optional<DataSet> loadDataSet(String dataSetName) {
        return loadDataSet(dataSetName, defaultLoaders);
    }

    public static Optional<DataSet> loadDataSet(String dataSetName, Collection<DataSetLoader> loaders) {
        logger.info("loading dataset [{}]", dataSetName);
        if (dataSetName.endsWith(".hdf5")) {
            throw new InvalidParameterException("DataSet names are not meant to be file names. Did you mean " + dataSetName.replace(".hdf5", "") + "? ");
        }

        for (DataSetLoader loader : loaders) {
            logger.trace("trying loader [{}]", loader.getClass().getSimpleName());
            Optional<DataSet> dataSetLoaded = loader.loadDataSet(dataSetName);
            if (dataSetLoaded.isPresent()) {
                logger.info("dataset [{}] found with loader [{}]", dataSetName, loader.getClass().getSimpleName());
                return dataSetLoaded;
            }
        }
        logger.warn("Unable to find dataset [{}] with any dataset loader.", dataSetName);
        return Optional.empty();
    }
}
