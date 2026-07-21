### JMX Runtime Configuration for Graph Index Builder

**Description**  
Introduces `GraphIndexBuilderConfig`, a JMX-managed singleton that exposes `GraphIndexBuilder`
construction parameters as runtime-tunable attributes. Before this change, options such as
`addHierarchy`, `refineFinalGraph`, and `parallelBuild` could only be set at the construction
call site and required a code change or application restart to modify. With this change, all
three parameters can be inspected and updated live via any standard JMX client — JConsole,
jvisualvm, jmxterm, or a monitoring agent — without restarting the JVM.

Changes take effect the next time a `GraphIndexBuilder` is constructed; they do not affect
indexes that are already being built or have already been built.

The implementation follows the Standard MBean pattern: `GraphIndexBuilderConfigMBean` declares
the managed attributes and `GraphIndexBuilderConfig` is the singleton implementation registered
under the object name `io.github.jbellis.jvector:type=GraphIndexBuilderConfig`. All attributes
are stored as `volatile` fields so writes from a JMX management thread are immediately visible
to application threads without additional synchronization. MBean registration is best-effort:
a registration failure (for example, in a restricted JVM environment) logs a warning but does
not disrupt normal operation — the singleton continues to supply its default values.

**Managed Attributes**

| Attribute | Type | Default | Description |
|---|---|---|---|
| `AddHierarchy` | `boolean` | `true` | When `true`, builds HNSW-style hierarchy layers on top of the base Vamana graph. `false` produces a flat level-0 (plain Vamana) index, which uses less memory and may build faster on small datasets. |
| `RefineFinalGraph` | `boolean` | `true` | When `true`, runs a second diversity-refinement pass over each node's edges after the initial build. Improves recall at the cost of additional build time. |
| `ParallelBuild` | `boolean` | `false` | When `true`, serializes level-0 node records concurrently via `OnDiskParallelGraphIndexWriter`. Both writers produce an identical on-disk format; switching this flag does not require re-indexing existing data. |

**How to Enable**

`GraphIndexBuilderConfig` is initialized automatically on first access and registers its MBean
with the platform MBeanServer. No application code changes are required to activate JMX
management — connecting a JMX client to a running JVector process is sufficient.

*Programmatic access* — read or set values directly from application code:

```java
import io.github.jbellis.jvector.management.GraphIndexBuilderConfig;

GraphIndexBuilderConfig config = GraphIndexBuilderConfig.getInstance();

// Read current values
boolean addHierarchy   = config.isAddHierarchy();
boolean refineFinal    = config.isRefineFinalGraph();
boolean parallelBuild  = config.isParallelBuild();

// Update at runtime (affects all subsequent GraphIndexBuilder constructions)
config.setAddHierarchy(false);
config.setParallelBuild(true);
```

*Non-deprecated constructor* — `GraphIndexBuilder` constructors that do not accept explicit
boolean flags read from `GraphIndexBuilderConfig` at construction time:

```java
// Reads addHierarchy and refineFinalGraph from JMX config
var builder = new GraphIndexBuilder(scoreProvider, dimension, M, beamWidth,
                                    neighborOverflow, alpha);
```

Constructors that accept explicit flags continue to honor the caller-supplied values and do
not consult the singleton, enabling call-site overrides when needed.

**Using JConsole**

JConsole is the standard JMX browser included with every JDK installation.

1. **Launch JConsole**

   ```
   jconsole
   ```

   In the connection dialog, select the target JVM process by name or PID and click
   **Connect**. If connecting to a remote process, use
   `<host>:<jmxPort>` after enabling remote JMX on the target JVM:

   ```
   -Dcom.sun.management.jmxremote
   -Dcom.sun.management.jmxremote.port=9999
   -Dcom.sun.management.jmxremote.authenticate=false
   -Dcom.sun.management.jmxremote.ssl=false
   ```

2. **Navigate to the MBean**

   Select the **MBeans** tab. In the left-hand tree expand:

   ```
   io.github.jbellis.jvector
     └── GraphIndexBuilderConfig
           └── Attributes
   ```

3. **Read an attribute**

   Click on **Attributes**. The right-hand panel lists all three attributes with their
   current values:

   ```
   AddHierarchy    true
   RefineFinalGraph  true
   ParallelBuild   false
   ```

4. **Set an attribute**

   Double-click the value cell next to the attribute you want to change, type the new
   value (`true` or `false`), and press **Enter**. The change takes effect immediately;
   the next `GraphIndexBuilder` constructed in that JVM will use the new value.

   Attribute changes are also logged at `INFO` level by JVector:

   ```
   INFO  GraphIndexBuilderConfig - JMX: addHierarchy changed true → false
   ```

**Using jmxterm (command-line alternative)**

```bash
# Connect to the target JVM by PID
java -jar jmxterm.jar
open <pid>

# Navigate to the MBean
bean io.github.jbellis.jvector:type=GraphIndexBuilderConfig

# Read all attributes
info -b

# Read a specific attribute
get AddHierarchy

# Set an attribute
set AddHierarchy false
set ParallelBuild true
```

**Notes**

- The `GraphIndexBuilderConfig` class and its MBean interface are annotated `@Experimental`.
  The attribute set and object name may change in a future release.
- JMX attribute changes are global — they apply to all `GraphIndexBuilder` instances created
  after the change in the same JVM. Per-graph overrides are not supported in this release;
  callers that need per-graph control should pass values explicitly to the appropriate
  constructor.
- The `parallelBuild` attribute requires the `OnDiskParallelGraphIndexWriter` to be on the
  classpath (it is part of `jvector-base`). When `parallelBuild` is `true`, the unified
  `RandomAccessOnDiskGraphIndexWriter.Builder` automatically selects the parallel writer at
  `build()` time.
