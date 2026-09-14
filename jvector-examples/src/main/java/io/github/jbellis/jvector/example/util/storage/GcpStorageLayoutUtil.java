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
package io.github.jbellis.jvector.example.util.storage;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Detects GCE runtime context via metadata service and classifies storage for each mounted filesystem.
 */
public final class GcpStorageLayoutUtil {
    private static final String GCE_METADATA_HOST_ENV = "GCE_METADATA_HOST";
    private static final String METADATA_HOST_DEFAULT = "metadata.google.internal";
    private static final String METADATA_PREFIX = "/computeMetadata/v1/";
    private static final String METADATA_FLAVOR_HEADER = "Metadata-Flavor";
    private static final String METADATA_FLAVOR_VALUE = "Google";
    private static final Duration METADATA_TIMEOUT = Duration.ofMillis(300);

    private static final Pattern NVME_PARTITION_SUFFIX = Pattern.compile("p\\d+$");
    private static final Pattern GENERIC_PARTITION_SUFFIX = Pattern.compile("\\d+$");
    private static final Set<String> NETWORK_FILESYSTEM_TYPES = Set.of("nfs", "nfs4", "efs", "cifs", "smbfs", "fuse.sshfs");

    private GcpStorageLayoutUtil() {
    }

    public enum StorageClass {
        PD_STANDARD_HDD,
        PD_THROUGHPUT_OPTIMIZED,
        PD_BALANCED_SSD,
        PD_SSD,
        PD_EXTREME_SSD,
        LOCAL_SSD,
        LOCAL_NVME,
        NETWORK_FILESYSTEM,
        MEMORY_TMPFS,
        PSEUDO_FILESYSTEM,
        UNKNOWN
    }

    public static final class StorageSnapshot {
        private final boolean runningOnGcp;
        private final String instanceId;
        private final String machineType;
        private final String zone;
        private final Map<String, MountStorageInfo> mountsByMountPoint;

        public StorageSnapshot(boolean runningOnGcp,
                               String instanceId,
                               String machineType,
                               String zone,
                               Map<String, MountStorageInfo> mountsByMountPoint) {
            this.runningOnGcp = runningOnGcp;
            this.instanceId = instanceId;
            this.machineType = machineType;
            this.zone = zone;
            this.mountsByMountPoint = Objects.requireNonNull(mountsByMountPoint, "mountsByMountPoint");
        }

        public boolean runningOnGcp() {
            return runningOnGcp;
        }

        public String instanceId() {
            return instanceId;
        }

        public String machineType() {
            return machineType;
        }

        public String zone() {
            return zone;
        }

        public Map<String, MountStorageInfo> mountsByMountPoint() {
            return mountsByMountPoint;
        }
    }

    public static final class MountStorageInfo {
        private final String mountPoint;
        private final String source;
        private final String filesystemType;
        private final StorageClass storageClass;
        private final String deviceName;
        private final String diskKind;
        private final String interfaceType;

        public MountStorageInfo(String mountPoint,
                                String source,
                                String filesystemType,
                                StorageClass storageClass,
                                String deviceName,
                                String diskKind,
                                String interfaceType) {
            this.mountPoint = mountPoint;
            this.source = source;
            this.filesystemType = filesystemType;
            this.storageClass = Objects.requireNonNull(storageClass, "storageClass");
            this.deviceName = deviceName;
            this.diskKind = diskKind;
            this.interfaceType = interfaceType;
        }

        public String mountPoint() {
            return mountPoint;
        }

        public String source() {
            return source;
        }

        public String filesystemType() {
            return filesystemType;
        }

        public StorageClass storageClass() {
            return storageClass;
        }

        public String deviceName() {
            return deviceName;
        }

        public String diskKind() {
            return diskKind;
        }

        public String interfaceType() {
            return interfaceType;
        }
    }

    public static StorageSnapshot inspectStorage() {
        var identity = fetchGcpIdentity();
        var mounts = readMountEntries();
        var diskData = identity.map(GcpStorageLayoutUtil::fetchGcpDiskData).orElse(GcpDiskData.empty());

        mounts.sort(Comparator.comparing(MountEntry::mountPoint));
        var byMountPoint = new LinkedHashMap<String, MountStorageInfo>(mounts.size());
        for (var mount : mounts) {
            var diskResolution = resolveDisk(mount.source(), diskData);
            var storageClass = classify(mount, diskResolution);
            byMountPoint.put(
                    mount.mountPoint(),
                    new MountStorageInfo(
                            mount.mountPoint(),
                            mount.source(),
                            mount.filesystemType(),
                            storageClass,
                            diskResolution.deviceName(),
                            diskResolution.diskKind(),
                            diskResolution.interfaceType()
                    )
            );
        }

        return new StorageSnapshot(
                identity.isPresent(),
                identity.map(GcpIdentity::instanceId).orElse(null),
                identity.map(GcpIdentity::machineType).orElse(null),
                identity.map(GcpIdentity::zone).orElse(null),
                Collections.unmodifiableMap(byMountPoint)
        );
    }

    public static Map<String, StorageClass> storageClassByMountPoint() {
        var snapshot = inspectStorage();
        var byMountPoint = new LinkedHashMap<String, StorageClass>(snapshot.mountsByMountPoint().size());
        for (var entry : snapshot.mountsByMountPoint().entrySet()) {
            byMountPoint.put(entry.getKey(), entry.getValue().storageClass());
        }
        return Collections.unmodifiableMap(byMountPoint);
    }

    private static Optional<GcpIdentity> fetchGcpIdentity() {
        var client = HttpClient.newBuilder()
                .connectTimeout(METADATA_TIMEOUT)
                .build();

        var instanceId = readMetadata(client, "instance/id");
        if (instanceId == null || instanceId.isBlank()) {
            return Optional.empty();
        }

        var machineType = parseLeafResource(readMetadata(client, "instance/machine-type"));
        var zone = parseLeafResource(readMetadata(client, "instance/zone"));
        return Optional.of(new GcpIdentity(instanceId.trim(), machineType, zone));
    }

    private static GcpDiskData fetchGcpDiskData(GcpIdentity ignoredIdentity) {
        var byDeviceName = fetchDisksByDeviceNameFromMetadata();
        var aliasesByNormalizedDevice = mapGoogleAliasesByNormalizedDevice();
        return new GcpDiskData(byDeviceName, aliasesByNormalizedDevice);
    }

    private static Map<String, GcpDiskInfo> fetchDisksByDeviceNameFromMetadata() {
        var client = HttpClient.newBuilder()
                .connectTimeout(METADATA_TIMEOUT)
                .build();

        var indexListing = readMetadata(client, "instance/disks/");
        if (indexListing == null || indexListing.isBlank()) {
            return Map.of();
        }

        var byDeviceName = new LinkedHashMap<String, GcpDiskInfo>();
        for (var rawLine : indexListing.split("\n")) {
            var line = rawLine.trim();
            if (line.isEmpty()) {
                continue;
            }
            var index = line.endsWith("/") ? line.substring(0, line.length() - 1) : line;
            var deviceName = readMetadata(client, "instance/disks/" + index + "/device-name");
            if (deviceName == null || deviceName.isBlank()) {
                continue;
            }

            var diskKind = safeLower(readMetadata(client, "instance/disks/" + index + "/type"));
            var interfaceType = safeUpper(readMetadata(client, "instance/disks/" + index + "/interface"));
            var diskTypeHint = readMetadata(client, "instance/disks/" + index + "/disk-type");
            byDeviceName.put(deviceName.trim(), new GcpDiskInfo(deviceName.trim(), diskKind, interfaceType, safeLower(diskTypeHint)));
        }
        return byDeviceName;
    }

    private static Map<String, List<String>> mapGoogleAliasesByNormalizedDevice() {
        var byIdDir = Path.of("/dev/disk/by-id");
        if (!Files.isDirectory(byIdDir)) {
            return Map.of();
        }

        var aliasesByDevice = new LinkedHashMap<String, List<String>>();
        try (Stream<Path> entries = Files.list(byIdDir)) {
            entries.filter(Files::isSymbolicLink).forEach(link -> {
                var alias = link.getFileName().toString();
                if (!alias.startsWith("google-")) {
                    return;
                }
                try {
                    var target = normalizeDevice(link.toRealPath().toString());
                    aliasesByDevice.computeIfAbsent(target, unused -> new ArrayList<>()).add(alias);
                } catch (IOException ignored) {
                    // continue
                }
            });
        } catch (IOException ignored) {
            return Map.of();
        }

        for (var aliases : aliasesByDevice.values()) {
            aliases.sort(String::compareTo);
        }
        return aliasesByDevice;
    }

    private static DiskResolution resolveDisk(String mountSource, GcpDiskData diskData) {
        if (mountSource == null || !mountSource.startsWith("/dev/")) {
            return DiskResolution.empty();
        }

        var normalized = normalizeDevice(mountSource);
        var aliases = diskData.aliasesByNormalizedDevice().getOrDefault(normalized, List.of());
        var primaryAlias = aliases.isEmpty() ? null : aliases.get(0);
        var inferredDeviceName = primaryAlias == null ? null : stripGooglePrefix(primaryAlias);
        GcpDiskInfo info = inferredDeviceName == null ? null : diskData.byDeviceName().get(inferredDeviceName);

        // Try all aliases in case the first one doesn't match a metadata device-name.
        if (info == null) {
            for (var alias : aliases) {
                var candidate = stripGooglePrefix(alias);
                if (candidate == null) {
                    continue;
                }
                info = diskData.byDeviceName().get(candidate);
                if (info != null) {
                    inferredDeviceName = candidate;
                    break;
                }
            }
        }

        var rotational = readRotationalFlag(normalized);
        if (info == null) {
            return new DiskResolution(normalized, inferredDeviceName, null, null, null, rotational);
        }
        return new DiskResolution(
                normalized,
                inferredDeviceName,
                info.diskKind(),
                info.interfaceType(),
                info.diskTypeHint(),
                rotational
        );
    }

    private static StorageClass classify(MountEntry mount, DiskResolution diskResolution) {
        var fsType = safeLower(mount.filesystemType());
        var source = mount.source();
        var sourceLower = safeLower(source);

        if ("tmpfs".equals(fsType)) {
            return StorageClass.MEMORY_TMPFS;
        }
        if (NETWORK_FILESYSTEM_TYPES.contains(fsType)) {
            return StorageClass.NETWORK_FILESYSTEM;
        }
        if (isPseudoFileSystem(fsType, sourceLower)) {
            return StorageClass.PSEUDO_FILESYSTEM;
        }

        if ("scratch".equals(diskResolution.diskKind())) {
            if ("NVME".equals(diskResolution.interfaceType()) || sourceLower.contains("nvme")) {
                return StorageClass.LOCAL_NVME;
            }
            return StorageClass.LOCAL_SSD;
        }
        if ("persistent".equals(diskResolution.diskKind())) {
            return classifyPersistentDisk(diskResolution);
        }

        // Best-effort fallback based on device name hints and local block characteristics.
        var hints = safeLower(diskResolution.deviceName()) + " "
                + safeLower(diskResolution.diskTypeHint()) + " "
                + sourceLower;
        if (hints.contains("local-ssd")) {
            return sourceLower.contains("nvme") ? StorageClass.LOCAL_NVME : StorageClass.LOCAL_SSD;
        }
        if (source != null && source.startsWith("/dev/")) {
            if (sourceLower.contains("nvme")) {
                return StorageClass.LOCAL_NVME;
            }
            if (Boolean.TRUE.equals(diskResolution.rotational())) {
                return StorageClass.PD_STANDARD_HDD;
            }
            return StorageClass.LOCAL_SSD;
        }
        return StorageClass.UNKNOWN;
    }

    private static StorageClass classifyPersistentDisk(DiskResolution diskResolution) {
        var hints = safeLower(diskResolution.deviceName()) + " " + safeLower(diskResolution.diskTypeHint());
        if (hints.contains("extreme")) {
            return StorageClass.PD_EXTREME_SSD;
        }
        if (hints.contains("throughput")) {
            return StorageClass.PD_THROUGHPUT_OPTIMIZED;
        }
        if (hints.contains("balanced")) {
            return StorageClass.PD_BALANCED_SSD;
        }
        if (hints.contains("pd-ssd") || hints.contains("ssd")) {
            return StorageClass.PD_SSD;
        }
        if (hints.contains("standard")) {
            return StorageClass.PD_STANDARD_HDD;
        }

        if (Boolean.TRUE.equals(diskResolution.rotational())) {
            return StorageClass.PD_STANDARD_HDD;
        }
        return StorageClass.PD_BALANCED_SSD;
    }

    private static String readMetadata(HttpClient client, String relativePath) {
        var host = Optional.ofNullable(System.getenv(GCE_METADATA_HOST_ENV)).orElse(METADATA_HOST_DEFAULT);
        var uri = URI.create("http://" + host + METADATA_PREFIX + relativePath);
        try {
            var request = HttpRequest.newBuilder(uri)
                    .timeout(METADATA_TIMEOUT)
                    .header(METADATA_FLAVOR_HEADER, METADATA_FLAVOR_VALUE)
                    .GET()
                    .build();
            var response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                return null;
            }
            var flavorHeader = response.headers().firstValue(METADATA_FLAVOR_HEADER).orElse("");
            if (!METADATA_FLAVOR_VALUE.equalsIgnoreCase(flavorHeader)) {
                return null;
            }
            return response.body();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        } catch (IOException e) {
            return null;
        }
    }

    private static String parseLeafResource(String value) {
        if (value == null) {
            return null;
        }
        var trimmed = value.trim();
        if (trimmed.isEmpty()) {
            return null;
        }
        var idx = trimmed.lastIndexOf('/');
        if (idx < 0 || idx == trimmed.length() - 1) {
            return trimmed;
        }
        return trimmed.substring(idx + 1);
    }

    private static List<MountEntry> readMountEntries() {
        var mountsPath = Files.isReadable(Path.of("/proc/self/mounts"))
                ? Path.of("/proc/self/mounts")
                : Path.of("/proc/mounts");

        if (!Files.isReadable(mountsPath)) {
            return new ArrayList<>();
        }

        var entries = new ArrayList<MountEntry>();
        try (Stream<String> lines = Files.lines(mountsPath)) {
            lines.forEach(line -> {
                var parts = line.split(" ");
                if (parts.length < 3) {
                    return;
                }
                var source = decodeMountToken(parts[0]);
                var mountPoint = decodeMountToken(parts[1]);
                var filesystemType = decodeMountToken(parts[2]);
                entries.add(new MountEntry(source, mountPoint, filesystemType));
            });
        } catch (IOException ignored) {
            return new ArrayList<>();
        }
        return entries;
    }

    private static Boolean readRotationalFlag(String normalizedDevice) {
        if (normalizedDevice == null || !normalizedDevice.startsWith("/dev/")) {
            return null;
        }
        var blockName = normalizedDevice.substring("/dev/".length());
        var rotaPath = Path.of("/sys/class/block", blockName, "queue", "rotational");
        if (!Files.isReadable(rotaPath)) {
            return null;
        }
        try {
            var value = Files.readString(rotaPath).trim();
            if ("1".equals(value)) {
                return Boolean.TRUE;
            }
            if ("0".equals(value)) {
                return Boolean.FALSE;
            }
        } catch (IOException ignored) {
            return null;
        }
        return null;
    }

    private static boolean isPseudoFileSystem(String fsType, String sourceLower) {
        return fsType.equals("proc")
                || fsType.equals("sysfs")
                || fsType.equals("devpts")
                || fsType.equals("devtmpfs")
                || fsType.equals("cgroup")
                || fsType.equals("cgroup2")
                || fsType.equals("autofs")
                || fsType.equals("mqueue")
                || fsType.equals("tracefs")
                || fsType.equals("pstore")
                || fsType.equals("securityfs")
                || fsType.equals("debugfs")
                || fsType.equals("configfs")
                || fsType.equals("fusectl")
                || fsType.equals("binfmt_misc")
                || fsType.equals("rpc_pipefs")
                || sourceLower.equals("proc")
                || sourceLower.equals("sysfs")
                || sourceLower.equals("tmpfs");
    }

    private static String normalizeDevice(String device) {
        if (device == null) {
            return null;
        }
        if (!device.startsWith("/dev/")) {
            return device;
        }
        if (device.startsWith("/dev/nvme")) {
            return NVME_PARTITION_SUFFIX.matcher(device).replaceAll("");
        }
        return GENERIC_PARTITION_SUFFIX.matcher(device).replaceAll("");
    }

    private static String decodeMountToken(String token) {
        return token
                .replace("\\040", " ")
                .replace("\\011", "\t")
                .replace("\\012", "\n")
                .replace("\\134", "\\");
    }

    private static String stripGooglePrefix(String alias) {
        if (alias == null || !alias.startsWith("google-") || alias.length() <= "google-".length()) {
            return null;
        }
        return alias.substring("google-".length());
    }

    private static String safeLower(String value) {
        return value == null ? "" : value.toLowerCase(Locale.ROOT);
    }

    private static String safeUpper(String value) {
        return value == null ? null : value.trim().toUpperCase(Locale.ROOT);
    }

    private static final class MountEntry {
        private final String source;
        private final String mountPoint;
        private final String filesystemType;

        private MountEntry(String source, String mountPoint, String filesystemType) {
            this.source = source;
            this.mountPoint = mountPoint;
            this.filesystemType = filesystemType;
        }

        private String source() {
            return source;
        }

        private String mountPoint() {
            return mountPoint;
        }

        private String filesystemType() {
            return filesystemType;
        }
    }

    private static final class GcpIdentity {
        private final String instanceId;
        private final String machineType;
        private final String zone;

        private GcpIdentity(String instanceId, String machineType, String zone) {
            this.instanceId = instanceId;
            this.machineType = machineType;
            this.zone = zone;
        }

        private String instanceId() {
            return instanceId;
        }

        private String machineType() {
            return machineType;
        }

        private String zone() {
            return zone;
        }
    }

    private static final class GcpDiskInfo {
        private final String deviceName;
        private final String diskKind;
        private final String interfaceType;
        private final String diskTypeHint;

        private GcpDiskInfo(String deviceName, String diskKind, String interfaceType, String diskTypeHint) {
            this.deviceName = deviceName;
            this.diskKind = diskKind;
            this.interfaceType = interfaceType;
            this.diskTypeHint = diskTypeHint;
        }

        private String deviceName() {
            return deviceName;
        }

        private String diskKind() {
            return diskKind;
        }

        private String interfaceType() {
            return interfaceType;
        }

        private String diskTypeHint() {
            return diskTypeHint;
        }
    }

    private static final class GcpDiskData {
        private final Map<String, GcpDiskInfo> byDeviceName;
        private final Map<String, List<String>> aliasesByNormalizedDevice;

        private GcpDiskData(Map<String, GcpDiskInfo> byDeviceName, Map<String, List<String>> aliasesByNormalizedDevice) {
            this.byDeviceName = Objects.requireNonNull(byDeviceName, "byDeviceName");
            this.aliasesByNormalizedDevice = Objects.requireNonNull(aliasesByNormalizedDevice, "aliasesByNormalizedDevice");
        }

        private Map<String, GcpDiskInfo> byDeviceName() {
            return byDeviceName;
        }

        private Map<String, List<String>> aliasesByNormalizedDevice() {
            return aliasesByNormalizedDevice;
        }

        private static GcpDiskData empty() {
            return new GcpDiskData(Map.of(), Map.of());
        }
    }

    private static final class DiskResolution {
        private final String normalizedDevice;
        private final String deviceName;
        private final String diskKind;
        private final String interfaceType;
        private final String diskTypeHint;
        private final Boolean rotational;

        private DiskResolution(String normalizedDevice,
                               String deviceName,
                               String diskKind,
                               String interfaceType,
                               String diskTypeHint,
                               Boolean rotational) {
            this.normalizedDevice = normalizedDevice;
            this.deviceName = deviceName;
            this.diskKind = diskKind;
            this.interfaceType = interfaceType;
            this.diskTypeHint = diskTypeHint;
            this.rotational = rotational;
        }

        private static DiskResolution empty() {
            return new DiskResolution(null, null, null, null, null, null);
        }

        private String normalizedDevice() {
            return normalizedDevice;
        }

        private String deviceName() {
            return deviceName;
        }

        private String diskKind() {
            return diskKind;
        }

        private String interfaceType() {
            return interfaceType;
        }

        private String diskTypeHint() {
            return diskTypeHint;
        }

        private Boolean rotational() {
            return rotational;
        }
    }
}
