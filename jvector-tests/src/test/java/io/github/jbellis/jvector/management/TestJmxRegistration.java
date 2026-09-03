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

package io.github.jbellis.jvector.management;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import javax.management.MBeanServer;
import javax.management.ObjectName;
import java.lang.management.ManagementFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Regression test for the externally-observable JMX surface of {@link GraphIndexBuilderConfig}.
 *
 * <p>{@link GraphIndexBuilderConfig} no longer registers itself with the platform
 * {@link MBeanServer} directly &mdash; registration now flows through
 * {@link ManagementRegistry} to whichever backend {@link ManagementBackendProvider} selects,
 * which defaults to JMX. This test verifies that, from an external JMX client's point of view
 * (JConsole, jmxterm, or a monitoring agent), nothing changed: the same {@link ObjectName}
 * is registered and attribute get/set round-trips against the live singleton, exactly as
 * documented in {@code docs/release notes/4.1.0/703.feature.md}.
 *
 * <p>Assumes the default management backend (JMX) is active, i.e. that
 * {@code jvector.management.backend} has not been overridden away from {@code jmx}.
 */
public class TestJmxRegistration {

    private static final ObjectName OBJECT_NAME;
    static {
        try {
            OBJECT_NAME = new ObjectName("io.github.jbellis.jvector:type=GraphIndexBuilderConfig");
        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private boolean savedAddHierarchy;

    @Before
    public void setup() {
        savedAddHierarchy = GraphIndexBuilderConfig.getInstance().isAddHierarchy();
    }

    @After
    public void tearDown() {
        GraphIndexBuilderConfig.getInstance().setAddHierarchy(savedAddHierarchy);
    }

    @Test
    public void configSingletonIsRegisteredUnderDocumentedObjectName() throws Exception {
        // touch the singleton to guarantee it has been constructed (and thus registered)
        GraphIndexBuilderConfig.getInstance();

        MBeanServer server = ManagementFactory.getPlatformMBeanServer();
        assertTrue("expected " + OBJECT_NAME + " to be registered with the platform MBeanServer",
                server.isRegistered(OBJECT_NAME));
    }

    @Test
    public void attributeRoundTripsThroughRealMBeanServer() throws Exception {
        GraphIndexBuilderConfig.getInstance();
        MBeanServer server = ManagementFactory.getPlatformMBeanServer();

        server.setAttribute(OBJECT_NAME, new javax.management.Attribute("AddHierarchy", false));
        assertEquals(false, GraphIndexBuilderConfig.getInstance().isAddHierarchy());
        assertEquals(false, server.getAttribute(OBJECT_NAME, "AddHierarchy"));

        server.setAttribute(OBJECT_NAME, new javax.management.Attribute("AddHierarchy", true));
        assertEquals(true, GraphIndexBuilderConfig.getInstance().isAddHierarchy());
        assertEquals(true, server.getAttribute(OBJECT_NAME, "AddHierarchy"));
    }
}
