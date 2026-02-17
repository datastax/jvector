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

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.ec2.Ec2Client;
import software.amazon.awssdk.services.ec2.model.DescribeInstancesRequest;
import software.amazon.awssdk.services.ec2.model.DescribeVolumesRequest;
import software.amazon.awssdk.services.ec2.model.InstanceBlockDeviceMapping;
import software.amazon.awssdk.services.ec2.model.Volume;

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
 * Detects EC2 runtime context via IMDSv2 and classifies storage for each mounted filesystem.
 */
public final class StorageLayoutUtil {
    private static final String AWS_EC2_METADATA_DISABLED = "AWS_EC2_METADATA_DISABLED";
    private static final URI IMDS_TOKEN_URI = URI.create("http://169.254.169.254/latest/api/token");
    private static final URI IMDS_IDENTITY_URI = URI.create("http://169.254.169.254/latest/dynamic/instance-identity/document");
    private static final Duration IMDS_TIMEOUT = Duration.ofMillis(300);
    private static final String IMDS_TOKEN_HEADER = "X-aws-ec2-metadata-token";
    private static final String IMDS_TOKEN_TTL_HEADER = "X-aws-ec2-metadata-token-ttl-seconds";

    private static final Pattern JSON_FIELD_PATTERN = Pattern.compile("\"([^\"]+)\"\\s*:\\s*\"([^\"]+)\"");
    private static final Pattern VOL_ID_PATTERN = Pattern.compile("vol-?[0-9a-fA-F]+");
    private static final Pattern NVME_PARTITION_SUFFIX = Pattern.compile("p\\d+$");
    private static final Pattern GENERIC_PARTITION_SUFFIX = Pattern.compile("\\d+$");
    private static final Set<String> NETWORK_FILESYSTEM_TYPES = Set.of("nfs", "nfs4", "efs", "cifs", "smbfs", "fuse.sshfs");

    private StorageLayoutUtil() {
    }

    public enum StorageClass {
        // Slowest EBS tiers
        EBS_COLD_HDD,
        EBS_THROUGHPUT_HDD,
        EBS_MAGNETIC,

        // Faster EBS SSD tiers
        EBS_GP2,
        EBS_GP3,
        EBS_PROVISIONED_IOPS_SSD,

        // Local instance storage
        INSTANCE_STORE_SSD,
        INSTANCE_STORE_NVME,

        // Non-block storage
        NETWORK_FILESYSTEM,
        MEMORY_TMPFS,
        PSEUDO_FILESYSTEM,
        UNKNOWN
    }

    public static final class StorageSnapshot {
        private final boolean runningOnEc2;
        private final String instanceId;
        private final String instanceType;
        private final String region;
        private final Map<String, MountStorageInfo> mountsByMountPoint;

        public StorageSnapshot(boolean runningOnEc2,
                               String instanceId,
                               String instanceType,
                               String region,
                               Map<String, MountStorageInfo> mountsByMountPoint) {
            this.runningOnEc2 = runningOnEc2;
            this.instanceId = instanceId;
            this.instanceType = instanceType;
            this.region = region;
            this.mountsByMountPoint = Objects.requireNonNull(mountsByMountPoint, "mountsByMountPoint");
        }

        public boolean runningOnEc2() {
            return runningOnEc2;
        }

        public String instanceId() {
            return instanceId;
        }

        public String instanceType() {
            return instanceType;
        }

        public String region() {
            return region;
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
        private final String volumeId;
        private final String volumeType;

        public MountStorageInfo(String mountPoint,
                                String source,
                                String filesystemType,
                                StorageClass storageClass,
                                String volumeId,
                                String volumeType) {
            this.mountPoint = mountPoint;
            this.source = source;
            this.filesystemType = filesystemType;
            this.storageClass = Objects.requireNonNull(storageClass, "storageClass");
            this.volumeId = volumeId;
            this.volumeType = volumeType;
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

        public String volumeId() {
            return volumeId;
        }

        public String volumeType() {
            return volumeType;
        }
    }

    public static StorageSnapshot inspectStorage() {
        var identity = fetchEc2Identity();
        var mounts = readMountEntries();
        var ec2Data = identity.map(StorageLayoutUtil::fetchEc2VolumeData).orElse(Ec2VolumeData.empty());

        mounts.sort(Comparator.comparing(MountEntry::mountPoint));
        var byMountPoint = new LinkedHashMap<String, MountStorageInfo>(mounts.size());
        for (var mount : mounts) {
            var resolvedVolumeId = resolveVolumeId(mount.source(), ec2Data);
            var volumeType = resolvedVolumeId == null ? null : ec2Data.volumeTypeById().get(resolvedVolumeId);
            var storageClass = classify(mount, resolvedVolumeId, volumeType);
            byMountPoint.put(
                    mount.mountPoint(),
                    new MountStorageInfo(
                            mount.mountPoint(),
                            mount.source(),
                            mount.filesystemType(),
                            storageClass,
                            resolvedVolumeId,
                            volumeType
                    )
            );
        }

        return new StorageSnapshot(
                identity.isPresent(),
                identity.map(Ec2Identity::instanceId).orElse(null),
                identity.map(Ec2Identity::instanceType).orElse(null),
                identity.map(Ec2Identity::region).orElse(null),
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

    private static Optional<Ec2Identity> fetchEc2Identity() {
        var imdsDisabled = System.getenv(AWS_EC2_METADATA_DISABLED);
        if (imdsDisabled != null && "true".equalsIgnoreCase(imdsDisabled)) {
            return Optional.empty();
        }

        var client = HttpClient.newBuilder()
                .connectTimeout(IMDS_TIMEOUT)
                .build();
        try {
            var tokenRequest = HttpRequest.newBuilder(IMDS_TOKEN_URI)
                    .timeout(IMDS_TIMEOUT)
                    .header(IMDS_TOKEN_TTL_HEADER, "60")
                    .method("PUT", HttpRequest.BodyPublishers.noBody())
                    .build();
            var tokenResponse = client.send(tokenRequest, HttpResponse.BodyHandlers.ofString());
            if (tokenResponse.statusCode() != 200) {
                return Optional.empty();
            }

            var token = tokenResponse.body();
            if (token == null || token.isBlank()) {
                return Optional.empty();
            }

            var identityRequest = HttpRequest.newBuilder(IMDS_IDENTITY_URI)
                    .timeout(IMDS_TIMEOUT)
                    .header(IMDS_TOKEN_HEADER, token)
                    .GET()
                    .build();
            var identityResponse = client.send(identityRequest, HttpResponse.BodyHandlers.ofString());
            if (identityResponse.statusCode() != 200) {
                return Optional.empty();
            }

            return parseIdentity(identityResponse.body());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return Optional.empty();
        } catch (IOException e) {
            return Optional.empty();
        }
    }

    private static Optional<Ec2Identity> parseIdentity(String json) {
        if (json == null || json.isBlank()) {
            return Optional.empty();
        }
        var values = new LinkedHashMap<String, String>();
        var matcher = JSON_FIELD_PATTERN.matcher(json);
        while (matcher.find()) {
            values.put(matcher.group(1), matcher.group(2));
        }

        var instanceId = values.get("instanceId");
        var instanceType = values.get("instanceType");
        var region = values.get("region");
        if (instanceId == null || instanceType == null || region == null) {
            return Optional.empty();
        }
        return Optional.of(new Ec2Identity(instanceId, instanceType, region));
    }

    private static Ec2VolumeData fetchEc2VolumeData(Ec2Identity identity) {
        var deviceNameToVolumeId = new LinkedHashMap<String, String>();
        var volumeTypeById = new LinkedHashMap<String, String>();
        var nvmeDeviceToVolumeId = mapNvmeDevicesToVolumeIds();

        try (var ec2 = Ec2Client.builder().region(Region.of(identity.region())).build()) {
            var instanceRequest = DescribeInstancesRequest.builder()
                    .instanceIds(identity.instanceId())
                    .build();
            var instanceResponse = ec2.describeInstances(instanceRequest);
            var reservations = instanceResponse.reservations();
            if (reservations != null) {
                for (var reservation : reservations) {
                    for (var instance : reservation.instances()) {
                        for (InstanceBlockDeviceMapping mapping : instance.blockDeviceMappings()) {
                            if (mapping.ebs() == null || mapping.ebs().volumeId() == null || mapping.deviceName() == null) {
                                continue;
                            }
                            deviceNameToVolumeId.put(normalizeDevice(mapping.deviceName()), mapping.ebs().volumeId());
                        }
                    }
                }
            }

            if (!deviceNameToVolumeId.isEmpty()) {
                var volumeResponse = ec2.describeVolumes(DescribeVolumesRequest.builder()
                        .volumeIds(deviceNameToVolumeId.values())
                        .build());
                for (Volume volume : volumeResponse.volumes()) {
                    if (volume.volumeId() != null && volume.volumeType() != null) {
                        volumeTypeById.put(volume.volumeId(), volume.volumeTypeAsString());
                    }
                }
            }
        } catch (RuntimeException ignored) {
            // If IAM permissions or service calls fail, we still return mount classifications.
        }

        return new Ec2VolumeData(deviceNameToVolumeId, nvmeDeviceToVolumeId, volumeTypeById);
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

    private static Map<String, String> mapNvmeDevicesToVolumeIds() {
        var byIdDir = Path.of("/dev/disk/by-id");
        if (!Files.isDirectory(byIdDir)) {
            return Map.of();
        }

        var mapping = new LinkedHashMap<String, String>();
        try (Stream<Path> entries = Files.list(byIdDir)) {
            entries.filter(Files::isSymbolicLink).forEach(link -> {
                var name = link.getFileName().toString();
                if (!name.startsWith("nvme-Amazon_Elastic_Block_Store_")) {
                    return;
                }
                var volumeId = extractVolumeId(name);
                if (volumeId == null) {
                    return;
                }

                try {
                    var target = normalizeDevice(link.toRealPath().toString());
                    mapping.put(target, volumeId);
                } catch (IOException ignored) {
                    // continue
                }
            });
        } catch (IOException ignored) {
            return Map.of();
        }
        return mapping;
    }

    private static String resolveVolumeId(String mountSource, Ec2VolumeData ec2Data) {
        if (mountSource == null || !mountSource.startsWith("/dev/")) {
            return null;
        }

        var normalized = normalizeDevice(mountSource);
        var byNvme = ec2Data.nvmeDeviceToVolumeId().get(normalized);
        if (byNvme != null) {
            return byNvme;
        }
        return ec2Data.deviceNameToVolumeId().get(normalized);
    }

    private static StorageClass classify(MountEntry mount, String volumeId, String volumeType) {
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

        if (volumeId != null) {
            return mapEbsVolumeType(volumeType);
        }

        if (source != null && source.startsWith("/dev/")) {
            if (sourceLower.contains("nvme")) {
                return StorageClass.INSTANCE_STORE_NVME;
            }
            return StorageClass.INSTANCE_STORE_SSD;
        }
        return StorageClass.UNKNOWN;
    }

    private static StorageClass mapEbsVolumeType(String volumeType) {
        if (volumeType == null) {
            return StorageClass.EBS_GP3;
        }

        switch (safeLower(volumeType)) {
            case "sc1":
                return StorageClass.EBS_COLD_HDD;
            case "st1":
                return StorageClass.EBS_THROUGHPUT_HDD;
            case "standard":
                return StorageClass.EBS_MAGNETIC;
            case "io1":
            case "io2":
                return StorageClass.EBS_PROVISIONED_IOPS_SSD;
            case "gp2":
                return StorageClass.EBS_GP2;
            case "gp3":
                return StorageClass.EBS_GP3;
            default:
                return StorageClass.EBS_GP3;
        }
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

    private static String decodeMountToken(String token) {
        return token
                .replace("\\040", " ")
                .replace("\\011", "\t")
                .replace("\\012", "\n")
                .replace("\\134", "\\");
    }

    private static String extractVolumeId(String value) {
        var matcher = VOL_ID_PATTERN.matcher(value);
        if (!matcher.find()) {
            return null;
        }
        var raw = matcher.group();
        if (raw.startsWith("vol-")) {
            return raw.toLowerCase(Locale.ROOT);
        }
        return "vol-" + raw.substring(3).toLowerCase(Locale.ROOT);
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

    private static final class Ec2Identity {
        private final String instanceId;
        private final String instanceType;
        private final String region;

        private Ec2Identity(String instanceId, String instanceType, String region) {
            this.instanceId = instanceId;
            this.instanceType = instanceType;
            this.region = region;
        }

        private String instanceId() {
            return instanceId;
        }

        private String instanceType() {
            return instanceType;
        }

        private String region() {
            return region;
        }
    }

    private static final class Ec2VolumeData {
        private final Map<String, String> deviceNameToVolumeId;
        private final Map<String, String> nvmeDeviceToVolumeId;
        private final Map<String, String> volumeTypeById;

        private Ec2VolumeData(Map<String, String> deviceNameToVolumeId,
                              Map<String, String> nvmeDeviceToVolumeId,
                              Map<String, String> volumeTypeById) {
            Objects.requireNonNull(deviceNameToVolumeId, "deviceNameToVolumeId");
            Objects.requireNonNull(nvmeDeviceToVolumeId, "nvmeDeviceToVolumeId");
            Objects.requireNonNull(volumeTypeById, "volumeTypeById");
            this.deviceNameToVolumeId = deviceNameToVolumeId;
            this.nvmeDeviceToVolumeId = nvmeDeviceToVolumeId;
            this.volumeTypeById = volumeTypeById;
        }

        private Map<String, String> deviceNameToVolumeId() {
            return deviceNameToVolumeId;
        }

        private Map<String, String> nvmeDeviceToVolumeId() {
            return nvmeDeviceToVolumeId;
        }

        private Map<String, String> volumeTypeById() {
            return volumeTypeById;
        }

        private static Ec2VolumeData empty() {
            return new Ec2VolumeData(Map.of(), Map.of(), Map.of());
        }
    }
}
