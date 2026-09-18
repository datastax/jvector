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

package io.github.jbellis.jvector.util;

import java.util.logging.Logger;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class TestRuntimeMode {
    private static final Logger LOG = Logger.getLogger(TestRuntimeMode.class.getName());

    @Test
    public void terseAndVerboseSpellingsBothParse() {
        assertTrue(RuntimeMode.parseIsDevelopment("dev", LOG));
        assertTrue(RuntimeMode.parseIsDevelopment("development", LOG));
        assertTrue(RuntimeMode.parseIsDevelopment("DEVELOPMENT", LOG));
        assertTrue(RuntimeMode.parseIsDevelopment("  Dev  ", LOG));
        assertFalse(RuntimeMode.parseIsDevelopment("prod", LOG));
        assertFalse(RuntimeMode.parseIsDevelopment("production", LOG));
        assertFalse(RuntimeMode.parseIsDevelopment("PROD", LOG));
    }

    @Test
    public void defaultIsProduction() {
        assertFalse("unset must be production — the diagnostic walk is opt-in",
                    RuntimeMode.parseIsDevelopment(null, LOG));
        assertFalse(RuntimeMode.parseIsDevelopment("", LOG));
    }

    /**
     * A typo must NOT fall back to development: that would silently
     * reintroduce the per-insert full-walk pathology the gate exists to
     * prevent.
     */
    @Test
    public void unknownValuesResolveToProduction() {
        assertFalse(RuntimeMode.parseIsDevelopment("porduction", LOG));
        assertFalse(RuntimeMode.parseIsDevelopment("debug", LOG));
        assertFalse(RuntimeMode.parseIsDevelopment("true", LOG));
    }
}
