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

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Best-effort storage inspection utility for non-cloud environments.
 * Supports Linux, macOS, and Windows using local OS signals and common mount metadata.
 */
public final class LocalStorageLayoutUtil {
    private static final Pattern LINUX_NVME_PARTITION_SUFFIX = Pattern.compile("p\\d+$");
    private static final Pattern GENERIC_PARTITION_SUFFIX = Pattern.compile("\\d+$");
    private static final Pattern MAC_MOUNT_PATTERN = Pattern.compile("^(.+) on (.+) \\((.+)\\)$");
    private static final Pattern MAC_DISK_SLICE_SUFFIX = Pattern.compile("s\\d+$");
    private static final Set<String> NETWORK_FILESYSTEM_TYPES =
            Set.of("nfs", "nfs4", "efs", "cifs", "smbfs", "fuse.sshfs", "afpfs", "webdav", "davfs");

    private LocalStorageLayoutUtil() {
    }

    public enum StorageClass {
        LOCAL_HDD,
        LOCAL_SSD,
        LOCAL_NVME,
        NETWORK_FILESYSTEM,
        MEMORY_TMPFS,
        PSEUDO_FILESYSTEM,
        UNKNOWN
    }

    public static final class StorageSnapshot {
        private final String osName;
        private final Map<String, MountStorageInfo> mountsByMountPoint;

        public StorageSnapshot(String osName, Map<String, MountStorageInfo> mountsByMountPoint) {
            this.osName = osName;
            this.mountsByMountPoint = Objects.requireNonNull(mountsByMountPoint, "mountsByMountPoint");
        }

        public String osName() {
            return osName;
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
        private final String osHint;

        public MountStorageInfo(String mountPoint,
                                String source,
                                String filesystemType,
                                StorageClass storageClass,
                                String osHint) {
            this.mountPoint = mountPoint;
            this.source = source;
            this.filesystemType = filesystemType;
            this.storageClass = Objects.requireNonNull(storageClass, "storageClass");
            this.osHint = osHint;
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

        public String osHint() {
            return osHint;
        }
    }

    public static StorageSnapshot inspectStorage() {
        var os = safeLower(System.getProperty("os.name"));
        List<MountEntry> mounts;
        if (isLinux(os)) {
            mounts = readLinuxMountEntries();
        } else if (isMac(os)) {
            mounts = readMacMountEntries();
        } else if (isWindows(os)) {
            mounts = readWindowsMountEntries();
        } else {
            mounts = readGenericMountEntries();
        }

        mounts.sort(Comparator.comparing(MountEntry::mountPoint));
        var byMountPoint = new LinkedHashMap<String, MountStorageInfo>(mounts.size());
        for (var mount : mounts) {
            StorageClass storageClass;
            String osHint;
            if (isLinux(os)) {
                storageClass = classifyLinux(mount);
                osHint = "linux";
            } else if (isMac(os)) {
                storageClass = classifyMac(mount);
                osHint = "macos";
            } else if (isWindows(os)) {
                storageClass = classifyWindows(mount);
                osHint = "windows";
            } else {
                storageClass = classifyGeneric(mount);
                osHint = "generic";
            }

            byMountPoint.put(
                    mount.mountPoint(),
                    new MountStorageInfo(
                            mount.mountPoint(),
                            mount.source(),
                            mount.filesystemType(),
                            storageClass,
                            osHint
                    )
            );
        }

        return new StorageSnapshot(
                System.getProperty("os.name"),
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

    private static List<MountEntry> readLinuxMountEntries() {
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
                entries.add(new MountEntry(
                        decodeMountToken(parts[0]),
                        decodeMountToken(parts[1]),
                        decodeMountToken(parts[2])
                ));
            });
        } catch (IOException ignored) {
            return new ArrayList<>();
        }
        return entries;
    }

    private static List<MountEntry> readMacMountEntries() {
        var entries = new ArrayList<MountEntry>();
        for (String line : runCommandLines("mount")) {
            var matcher = MAC_MOUNT_PATTERN.matcher(line);
            if (!matcher.matches()) {
                continue;
            }
            var source = matcher.group(1).trim();
            var mountPoint = matcher.group(2).trim();
            var options = matcher.group(3).trim();
            var fsType = options.split(",")[0].trim();
            entries.add(new MountEntry(source, mountPoint, fsType));
        }
        if (entries.isEmpty()) {
            return readGenericMountEntries();
        }
        return entries;
    }

    private static List<MountEntry> readWindowsMountEntries() {
        var entries = new ArrayList<MountEntry>();
        var roots = File.listRoots();
        if (roots == null) {
            return entries;
        }
        for (var root : roots) {
            if (root == null) {
                continue;
            }
            var path = root.toPath();
            String fsType = "unknown";
            try {
                fsType = Files.getFileStore(path).type();
            } catch (IOException ignored) {
                // keep default
            }
            entries.add(new MountEntry(root.getPath(), root.getPath(), fsType));
        }
        return entries;
    }

    private static List<MountEntry> readGenericMountEntries() {
        var entries = new ArrayList<MountEntry>();
        var roots = File.listRoots();
        if (roots == null) {
            return entries;
        }
        for (var root : roots) {
            if (root == null) {
                continue;
            }
            String fsType = "unknown";
            try {
                fsType = Files.getFileStore(root.toPath()).type();
            } catch (IOException ignored) {
                // keep default
            }
            entries.add(new MountEntry(root.getPath(), root.getPath(), fsType));
        }
        return entries;
    }

    private static StorageClass classifyLinux(MountEntry mount) {
        var fsType = safeLower(mount.filesystemType());
        var source = mount.source();
        var sourceLower = safeLower(source);

        if ("tmpfs".equals(fsType) || "ramfs".equals(fsType)) {
            return StorageClass.MEMORY_TMPFS;
        }
        if (NETWORK_FILESYSTEM_TYPES.contains(fsType) || sourceLower.startsWith("//")) {
            return StorageClass.NETWORK_FILESYSTEM;
        }
        if (isPseudoFileSystem(fsType, sourceLower)) {
            return StorageClass.PSEUDO_FILESYSTEM;
        }

        if (source != null && source.startsWith("/dev/")) {
            var normalized = normalizeLinuxDevice(sourceLower);
            if (normalized.contains("nvme")) {
                return StorageClass.LOCAL_NVME;
            }

            Boolean rotational = readLinuxRotationalFlag(normalized);
            if (Boolean.TRUE.equals(rotational)) {
                return StorageClass.LOCAL_HDD;
            }
            if (Boolean.FALSE.equals(rotational)) {
                return StorageClass.LOCAL_SSD;
            }
            return StorageClass.UNKNOWN;
        }
        return StorageClass.UNKNOWN;
    }

    private static StorageClass classifyMac(MountEntry mount) {
        var fsType = safeLower(mount.filesystemType());
        var source = mount.source();
        var sourceLower = safeLower(source);

        if ("devfs".equals(fsType) || "autofs".equals(fsType) || "procfs".equals(fsType)) {
            return StorageClass.PSEUDO_FILESYSTEM;
        }
        if ("tmpfs".equals(fsType) || "ramfs".equals(fsType)) {
            return StorageClass.MEMORY_TMPFS;
        }
        if (NETWORK_FILESYSTEM_TYPES.contains(fsType) || sourceLower.startsWith("//")) {
            return StorageClass.NETWORK_FILESYSTEM;
        }

        if (source != null && source.startsWith("/dev/")) {
            var diskInfo = readMacDiskInfo(source);
            if (diskInfo.protocolNvme) {
                return StorageClass.LOCAL_NVME;
            }
            if (diskInfo.solidState != null) {
                return diskInfo.solidState ? StorageClass.LOCAL_SSD : StorageClass.LOCAL_HDD;
            }
            if (sourceLower.contains("nvme")) {
                return StorageClass.LOCAL_NVME;
            }
            return StorageClass.UNKNOWN;
        }
        return StorageClass.UNKNOWN;
    }

    private static StorageClass classifyWindows(MountEntry mount) {
        var fsType = safeLower(mount.filesystemType());
        var source = mount.source();
        var sourceLower = safeLower(source);

        if (NETWORK_FILESYSTEM_TYPES.contains(fsType)
                || fsType.contains("smb")
                || fsType.contains("cifs")
                || sourceLower.startsWith("\\\\")) {
            return StorageClass.NETWORK_FILESYSTEM;
        }
        if (fsType.contains("tmp") || fsType.contains("ram")) {
            return StorageClass.MEMORY_TMPFS;
        }

        // Generic stub: fixed drives are treated as local SSD class when media specifics are unavailable.
        if (source != null && source.matches("^[A-Za-z]:\\\\.*")) {
            return StorageClass.LOCAL_SSD;
        }
        return StorageClass.UNKNOWN;
    }

    private static StorageClass classifyGeneric(MountEntry mount) {
        var fsType = safeLower(mount.filesystemType());
        if ("tmpfs".equals(fsType) || "ramfs".equals(fsType)) {
            return StorageClass.MEMORY_TMPFS;
        }
        if (NETWORK_FILESYSTEM_TYPES.contains(fsType)) {
            return StorageClass.NETWORK_FILESYSTEM;
        }
        return StorageClass.UNKNOWN;
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

    private static String normalizeLinuxDevice(String device) {
        if (!device.startsWith("/dev/")) {
            return device;
        }
        if (device.startsWith("/dev/nvme")) {
            return LINUX_NVME_PARTITION_SUFFIX.matcher(device).replaceAll("");
        }
        return GENERIC_PARTITION_SUFFIX.matcher(device).replaceAll("");
    }

    private static Boolean readLinuxRotationalFlag(String normalizedDevice) {
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

    private static MacDiskInfo readMacDiskInfo(String sourceDevice) {
        var base = sourceDevice;
        var slash = sourceDevice.lastIndexOf('/');
        if (slash >= 0 && slash + 1 < sourceDevice.length()) {
            var leaf = sourceDevice.substring(slash + 1);
            if (leaf.startsWith("disk")) {
                leaf = MAC_DISK_SLICE_SUFFIX.matcher(leaf).replaceAll("");
                base = "/dev/" + leaf;
            }
        }

        Boolean solidState = null;
        boolean protocolNvme = false;
        for (String line : runCommandLines("diskutil", "info", base)) {
            var trimmed = line.trim();
            var lower = safeLower(trimmed);
            if (lower.startsWith("solid state:")) {
                solidState = lower.endsWith("yes");
            } else if (lower.startsWith("protocol:")) {
                protocolNvme = lower.contains("nvme");
            } else if (lower.startsWith("device / media name:") && lower.contains("nvme")) {
                protocolNvme = true;
            }
        }
        return new MacDiskInfo(solidState, protocolNvme);
    }

    private static List<String> runCommandLines(String... command) {
        var lines = new ArrayList<String>();
        var pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);
        try {
            var process = pb.start();
            try (var reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    lines.add(line);
                }
            }
            process.waitFor();
        } catch (IOException | InterruptedException ignored) {
            if (ignored instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
        }
        return lines;
    }

    private static String decodeMountToken(String token) {
        return token
                .replace("\\040", " ")
                .replace("\\011", "\t")
                .replace("\\012", "\n")
                .replace("\\134", "\\");
    }

    private static boolean isLinux(String osNameLower) {
        return osNameLower.contains("linux");
    }

    private static boolean isMac(String osNameLower) {
        return osNameLower.contains("mac") || osNameLower.contains("darwin");
    }

    private static boolean isWindows(String osNameLower) {
        return osNameLower.contains("win");
    }

    private static String safeLower(String value) {
        return value == null ? "" : value.toLowerCase(Locale.ROOT);
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

    private static final class MacDiskInfo {
        private final Boolean solidState;
        private final boolean protocolNvme;

        private MacDiskInfo(Boolean solidState, boolean protocolNvme) {
            this.solidState = solidState;
            this.protocolNvme = protocolNvme;
        }
    }
}
