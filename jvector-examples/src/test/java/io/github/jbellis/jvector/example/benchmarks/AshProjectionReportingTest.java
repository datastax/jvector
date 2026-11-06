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

import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.example.reporting.*;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import java.nio.file.Files;
import java.util.*;
import static org.junit.Assert.*;

public class AshProjectionReportingTest {
    @Rule public final TemporaryFolder temporary = new TemporaryFolder();
    @Test public void csvAddsOnlyTwoColumnsAndReportsPreparedViewWithoutInitializingIt() throws Exception {
        String key = RunConfig.ASH_QUERY_PROJECTION_PROPERTY, previous = System.getProperty(key);
        System.setProperty(key, "fp32");
        try {
            assertEquals(List.of("schema_version","run_id","run_uuid","system_id","dataset_name",
                    "M","efConstruction","neighborOverflow","addHierarchy","refineFinalGraph",
                    "feature_set","build_compressor","search_compressor","usePruning","topK","overquery","rerankK"),
                    ExperimentsSchemaV1.fixedColumns().subList(0,17));
            assertEquals(List.of("ash_query_projection_requested","ash_query_projection_effective"),
                    ExperimentsSchemaV1.fixedColumns().subList(17,19));
            var vectors = VectorizationProvider.getInstance().getVectorTypeSupport();
            var first = vectors.createFloatVector(new float[]{1,0,0,0});
            var second = vectors.createFloatVector(new float[]{0,1,0,0});
            var ds = new SimpleDataSet("precision-metadata", VectorSimilarityFunction.DOT_PRODUCT,
                    List.of(first,second), List.of(first), List.of(List.of(0)));
            var model = AsymmetricHashing.initialize(ds.getBaseRavv(), AsymmetricHashing.RANDOM, 48, 1, 2);
            assertNull(model.stiefelTransform.initializedQueryProjectionPrecision());
            var config = new RunConfig();
            config.benchmarks = Map.of("accuracy",List.of("recall"));
            config.logging = new RunConfig.RunLogging();
            config.logging.type = "csv";
            config.logging.dir = temporary.getRoot().toString();
            config.logging.benchmarks = config.benchmarks;
            var artifacts = RunArtifacts.open(config,List.of());
            var metrics = List.of(Metric.of("search.accuracy.recall_at_10","recall",".5f",1));
            artifacts.logRow("precision-metadata",32,100,1.2f,true,true,Set.of(FeatureId.FUSED_ASH),"ASH","None",false,10,1,10,model,metrics);
            assertNull(model.stiefelTransform.initializedQueryProjectionPrecision());
            model.stiefelTransform.projectQueryDirect(new float[]{1,0,0,0},new float[model.quantizedDim]);
            assertEquals("fp32",model.stiefelTransform.initializedQueryProjectionPrecision());
            artifacts.logRow("precision-metadata",32,100,1.2f,true,true,Set.of(FeatureId.FUSED_ASH),"ASH","None",false,10,2,20,model,metrics);
            artifacts.logRow("precision-metadata",32,100,1.2f,true,true,Set.of(FeatureId.FUSED_PQ),"PQ","None",false,10,1,10,null,metrics);
            var lines = Files.readAllLines(artifacts.experimentsWriter().path());
            assertEquals(4,lines.size());
            assertEquals("fp32",lines.get(1).split(",",-1)[17]);
            assertEquals("",lines.get(1).split(",",-1)[18]);
            assertEquals("fp32",lines.get(2).split(",",-1)[17]);
            assertEquals("fp32",lines.get(2).split(",",-1)[18]);
            assertEquals("",lines.get(3).split(",",-1)[17]);
            assertEquals("",lines.get(3).split(",",-1)[18]);
        } finally {
            if(previous==null)System.clearProperty(key);else System.setProperty(key,previous);
        }
    }
}
