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
package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.graph.similarity.QueryPreparer;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.util.Arrays;
import java.util.Objects;

/**
 * Caller-scheduled ASH preparation. Uses the same query precision and native
 * matrix as synchronous scoring, and calls an already formed batch directly.
 * Encoding and training representations are unaffected.
 */
@Experimental
public final class ASHQueryPreparer implements QueryPreparer {
    private final AsymmetricHashing ash;
    private final ASHScorer scorer;
    private static final class Rows {
        float[][] input = new float[8][];
        float[][] output = new float[8][];
        void ensure(int count) {
            if (count > input.length) {
                input = new float[count][];
                output = new float[count][];
            }
        }
    }
    // Values contain only cleared row-reference arrays, never a model or query.
    private final ThreadLocal<Rows> rows = ThreadLocal.withInitial(Rows::new);

    public ASHQueryPreparer(AsymmetricHashing ash) {
        this.ash = Objects.requireNonNull(ash, "ash");
        FusedASHLayout.validateBitsPerDimension(ash.bitsPerDimension);
        this.scorer = new ASHScorer(ash);
    }

    @Override
    public PreparedQuery newState() { return new State(ash, scorer.newQueryPrecompute()); }

    @Override
    public boolean supportsBatchPreparation() {
        return ash.stiefelTransform.supportsQueryBatchProjection();
    }

    @Override
    public void prepare(VectorFloat<?> query, PreparedQuery preparedState) {
        State state = state(ash, preparedState, false);
        validate(query, state);
        state.invalidate();
        state.copy(query);
        ash.stiefelTransform.projectQueryDirect(state.input, state.prepared.qProj);
        finish(state);
    }

    @Override
    public void prepareBatch(VectorFloat<?>[] queries, PreparedQuery[] preparedStates, int count) {
        Objects.requireNonNull(queries, "queries");
        Objects.requireNonNull(preparedStates, "preparedStates");
        Objects.checkFromIndexSize(0, count, queries.length);
        Objects.checkFromIndexSize(0, count, preparedStates.length);
        if (count == 0) return;
        if (count == 1) {
            // Preserve each provider's direct kernel and avoid batch staging.
            // prepare() validates before invalidating or overwriting the state.
            prepare(queries[0], preparedStates[0]);
            return;
        }
        validateBatch(queries, preparedStates, count);
        Rows buffers = rows.get();
        buffers.ensure(count);
        // Invalidate all outputs before any computation can fail.
        for (int i = 0; i < count; i++) ((State) preparedStates[i]).invalidate();
        try {
            for (int i = 0; i < count; i++) {
                State state = (State) preparedStates[i];
                state.copy(queries[i]);
                buffers.input[i] = state.input;
                buffers.output[i] = state.prepared.qProj;
            }
            ash.stiefelTransform.projectQueryBatch(buffers.input, buffers.output, count);
            for (int i = 0; i < count; i++) finish((State) preparedStates[i]);
        } catch (RuntimeException | Error failure) {
            for (int i = 0; i < count; i++) ((State) preparedStates[i]).ready = false;
            throw failure;
        } finally {
            Arrays.fill(buffers.input, 0, count, null);
            Arrays.fill(buffers.output, 0, count, null);
        }
    }

    private void validateBatch(VectorFloat<?>[] queries, PreparedQuery[] preparedStates, int count) {
        for (int i = 0; i < count; i++) {
            State state = state(ash, preparedStates[i], false);
            validate(queries[i], state);
            for (int j = 0; j < i; j++)
                if (preparedStates[j] == state)
                    throw new IllegalArgumentException("Batch output states must be distinct");
        }
        for (int i = 0; i < count; i++) {
            // Resolve an array-backed input once, rather than once per output.
            // Other implementations are copied through VectorFloat's accessor.
            if (!(queries[i] instanceof FloatArray)) continue;
            float[] input = ((FloatArray) queries[i]).array();
            for (int j = 0; j < count; j++) {
                if (i != j && input == ((State) preparedStates[j]).input)
                    throw new IllegalArgumentException("Batch input aliases another output state");
            }
        }
    }

    private void validate(VectorFloat<?> query, State state) {
        scorer.validateQuery(query);
        if (state.ash != ash) throw new IllegalArgumentException("Prepared query belongs to another ASH model");
    }

    private void finish(State state) {
        scorer.finishQuery(state.query, state.prepared);
        FusedASHLayout.buildQueryLut(state.prepared.qProj, ash.quantizedDim, ash.bitsPerDimension, state.lut);
        state.ready = true;
    }

    static State state(AsymmetricHashing ash, PreparedQuery query, boolean requireReady) {
        Objects.requireNonNull(query, "preparedQuery");
        if (!(query instanceof State) || ((State) query).ash != ash)
            throw new IllegalArgumentException("Prepared query belongs to another ASH model");
        State state = (State) query;
        if (requireReady) state.check(state.generation);
        return state;
    }

    @Override
    public long ramBytesUsed() {
        return RamUsageEstimator.shallowSizeOf(this) + RamUsageEstimator.shallowSizeOf(scorer)
                + RamUsageEstimator.shallowSizeOf(rows);
    }

    static final class State implements PreparedQuery {
        final AsymmetricHashing ash;
        final float[] input;
        final VectorFloat<?> query;
        final ASHScorer.QueryPrecompute prepared;
        final float[] lut;
        final long bytes;
        long generation;
        boolean ready;

        State(AsymmetricHashing ash, ASHScorer.QueryPrecompute prepared) {
            this.ash = ash;
            this.input = new float[ash.originalDimension];
            this.query = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(input);
            this.prepared = prepared;
            this.lut = new float[Math.multiplyExact(FusedASHLayout.codeGroups(ash.quantizedDim, ash.bitsPerDimension), 16)];
            // Initialize a provider's lazy view once, outside preparation/search.
            Object backing = query.get();
            long sum = RamUsageEstimator.shallowSizeOf(this) + RamUsageEstimator.sizeOf(input)
                    + RamUsageEstimator.shallowSizeOf(query) + RamUsageEstimator.shallowSizeOf(prepared)
                    + RamUsageEstimator.sizeOf(prepared.qProj) + RamUsageEstimator.sizeOf(lut)
                    + RamUsageEstimator.sizeOf(prepared.dotQMuByLandmark);
            if (backing != input) sum += RamUsageEstimator.shallowSizeOf(backing);
            this.bytes = sum;
        }

        void invalidate() { ready = false; generation++; }
        void copy(VectorFloat<?> source) {
            if (source instanceof FloatArray) {
                System.arraycopy(((FloatArray) source).array(), 0, input, 0, input.length);
            } else {
                for (int i = 0; i < input.length; i++) input[i] = source.get(i);
            }
        }
        void check(long expectedGeneration) {
            if (!ready || generation != expectedGeneration)
                throw new IllegalStateException("Prepared query is unprepared or has been reused");
        }
        @Override
        public VectorFloat<?> queryVector() { check(generation); return query; }
        @Override
        public long ramBytesUsed() { return bytes; }
    }
}
