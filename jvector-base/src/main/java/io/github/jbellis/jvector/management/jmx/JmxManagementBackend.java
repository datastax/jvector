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

package io.github.jbellis.jvector.management.jmx;

import io.github.jbellis.jvector.management.ManagementEntry;
import io.github.jbellis.jvector.management.spi.ManagementBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.management.MBeanServer;
import javax.management.MalformedObjectNameException;
import javax.management.NotCompliantMBeanException;
import javax.management.ObjectName;
import javax.management.StandardMBean;
import java.lang.management.ManagementFactory;

/**
 * The default {@link ManagementBackend}: exposes each {@link ManagementEntry} as a JMX Standard
 * MBean on the platform {@link MBeanServer}, under the object name
 * {@code io.github.jbellis.jvector:type=<entry name>}.
 *
 * <p>This is the only class in JVector that depends on {@code javax.management}. It uses
 * {@link StandardMBean}'s explicit-interface constructor, which registers a resource against a
 * management interface regardless of naming &mdash; unlike the implicit Standard MBean
 * convention (where the interface must be named {@code <Impl>MBean}), the domain interfaces
 * this backend wraps (for example {@code GraphIndexBuilderSettings}) carry no JMX-specific
 * naming and no compile-time dependency on this package.
 *
 * <p>Registration and deregistration are best-effort: failures (for example, a restricted JVM
 * environment, or an object-name collision) are logged at {@code WARN} and otherwise ignored,
 * so JMX availability is never on the application's critical path.
 */
public class JmxManagementBackend implements ManagementBackend {

    private static final Logger logger = LoggerFactory.getLogger(JmxManagementBackend.class);
    private static final String DOMAIN = "io.github.jbellis.jvector";

    @Override
    public void bind(ManagementEntry entry) {
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            ObjectName objectName = objectName(entry);
            if (server.isRegistered(objectName)) {
                server.unregisterMBean(objectName);
            }
            server.registerMBean(wrap(entry), objectName);
            logger.info("Registered JMX MBean: {}", objectName);
        } catch (Exception e) {
            logger.warn("Failed to register JMX MBean for '{}': {}", entry.name(), e.getMessage());
        }
    }

    @Override
    public void unbind(ManagementEntry entry) {
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            ObjectName objectName = objectName(entry);
            if (server.isRegistered(objectName)) {
                server.unregisterMBean(objectName);
                logger.info("Unregistered JMX MBean: {}", objectName);
            }
        } catch (Exception e) {
            logger.warn("Failed to unregister JMX MBean for '{}': {}", entry.name(), e.getMessage());
        }
    }

    private static ObjectName objectName(ManagementEntry entry) throws MalformedObjectNameException {
        return new ObjectName(DOMAIN + ":type=" + entry.name());
    }

    private static <T> StandardMBean wrap(ManagementEntry entry) throws NotCompliantMBeanException {
        @SuppressWarnings("unchecked")
        Class<T> serviceInterface = (Class<T>) entry.serviceInterface();
        T resource = serviceInterface.cast(entry.resource());
        return new StandardMBean(resource, serviceInterface, false);
    }
}
