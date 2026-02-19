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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Cloud wrapper that chooses AWS or GCP storage inspection and maps the provider-specific classes
 * into cloud-agnostic storage tiers.
 */
public final class CloudStorageLayoutUtil {
    private CloudStorageLayoutUtil() {
    }

    public enum CloudProvider {
        AWS_EC2,
        GCP_GCE,
        LOCAL_OR_UNKNOWN
    }

    public enum StorageClass {
        BLOCK_HDD_COLD,
        BLOCK_HDD_THROUGHPUT,
        BLOCK_HDD_STANDARD,
        BLOCK_SSD_BALANCED,
        BLOCK_SSD_GENERAL,
        BLOCK_SSD_HIGH_IOPS,
        LOCAL_SSD,
        LOCAL_NVME,
        NETWORK_FILESYSTEM,
        MEMORY_TMPFS,
        PSEUDO_FILESYSTEM,
        UNKNOWN
    }

    public static final class StorageSnapshot<T> {
        private final T cloudSpecificSnapshot;
        private final CloudProvider provider;
        private final boolean runningInCloud;
        private final String instanceId;
        private final String instanceTypeOrMachineType;
        private final String regionOrZone;
        private final Map<String, MountStorageInfo> mountsByMountPoint;

        public StorageSnapshot(T cloudSpecificSnapshot,
                               CloudProvider provider,
                               boolean runningInCloud,
                               String instanceId,
                               String instanceTypeOrMachineType,
                               String regionOrZone,
                               Map<String, MountStorageInfo> mountsByMountPoint) {
            this.cloudSpecificSnapshot = cloudSpecificSnapshot;
            this.provider = Objects.requireNonNull(provider, "provider");
            this.runningInCloud = runningInCloud;
            this.instanceId = instanceId;
            this.instanceTypeOrMachineType = instanceTypeOrMachineType;
            this.regionOrZone = regionOrZone;
            this.mountsByMountPoint = Objects.requireNonNull(mountsByMountPoint, "mountsByMountPoint");
        }

        public T cloudSpecificSnapshot() {
            return cloudSpecificSnapshot;
        }

        public CloudProvider provider() {
            return provider;
        }

        public boolean runningInCloud() {
            return runningInCloud;
        }

        public String instanceId() {
            return instanceId;
        }

        public String instanceTypeOrMachineType() {
            return instanceTypeOrMachineType;
        }

        public String regionOrZone() {
            return regionOrZone;
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
        private final String providerSpecificClass;

        public MountStorageInfo(String mountPoint,
                                String source,
                                String filesystemType,
                                StorageClass storageClass,
                                String providerSpecificClass) {
            this.mountPoint = mountPoint;
            this.source = source;
            this.filesystemType = filesystemType;
            this.storageClass = Objects.requireNonNull(storageClass, "storageClass");
            this.providerSpecificClass = providerSpecificClass;
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

        public String providerSpecificClass() {
            return providerSpecificClass;
        }
    }

    public static StorageSnapshot<?> inspectStorage() {
        var awsSnapshot = StorageLayoutUtil.inspectStorage();
        if (awsSnapshot.runningOnEc2()) {
            return fromAws(awsSnapshot, CloudProvider.AWS_EC2, true);
        }

        var gcpSnapshot = GcpStorageLayoutUtil.inspectStorage();
        if (gcpSnapshot.runningOnGcp()) {
            return fromGcp(gcpSnapshot);
        }

        // Not in a detected cloud environment. Use OS-specific local storage inspection.
        var localSnapshot = LocalStorageLayoutUtil.inspectStorage();
        return fromLocal(localSnapshot);
    }

    public static Map<String, StorageClass> storageClassByMountPoint() {
        var snapshot = inspectStorage();
        var byMountPoint = new LinkedHashMap<String, StorageClass>(snapshot.mountsByMountPoint().size());
        for (var entry : snapshot.mountsByMountPoint().entrySet()) {
            byMountPoint.put(entry.getKey(), entry.getValue().storageClass());
        }
        return Collections.unmodifiableMap(byMountPoint);
    }

    private static StorageSnapshot<StorageLayoutUtil.StorageSnapshot> fromAws(StorageLayoutUtil.StorageSnapshot snapshot,
                                                                               CloudProvider provider,
                                                                               boolean runningInCloud) {
        var byMountPoint = new LinkedHashMap<String, MountStorageInfo>(snapshot.mountsByMountPoint().size());
        for (var entry : snapshot.mountsByMountPoint().entrySet()) {
            var mount = entry.getValue();
            byMountPoint.put(
                    entry.getKey(),
                    new MountStorageInfo(
                            mount.mountPoint(),
                            mount.source(),
                            mount.filesystemType(),
                            mapAwsClass(mount.storageClass()),
                            mount.storageClass().name()
                    )
            );
        }

        return new StorageSnapshot<>(
                snapshot,
                provider,
                runningInCloud,
                snapshot.instanceId(),
                snapshot.instanceType(),
                snapshot.region(),
                Collections.unmodifiableMap(byMountPoint)
        );
    }

    private static StorageSnapshot<GcpStorageLayoutUtil.StorageSnapshot> fromGcp(GcpStorageLayoutUtil.StorageSnapshot snapshot) {
        var byMountPoint = new LinkedHashMap<String, MountStorageInfo>(snapshot.mountsByMountPoint().size());
        for (var entry : snapshot.mountsByMountPoint().entrySet()) {
            var mount = entry.getValue();
            byMountPoint.put(
                    entry.getKey(),
                    new MountStorageInfo(
                            mount.mountPoint(),
                            mount.source(),
                            mount.filesystemType(),
                            mapGcpClass(mount.storageClass()),
                            mount.storageClass().name()
                    )
            );
        }

        return new StorageSnapshot<>(
                snapshot,
                CloudProvider.GCP_GCE,
                true,
                snapshot.instanceId(),
                snapshot.machineType(),
                snapshot.zone(),
                Collections.unmodifiableMap(byMountPoint)
        );
    }

    private static StorageSnapshot<LocalStorageLayoutUtil.StorageSnapshot> fromLocal(LocalStorageLayoutUtil.StorageSnapshot snapshot) {
        var byMountPoint = new LinkedHashMap<String, MountStorageInfo>(snapshot.mountsByMountPoint().size());
        for (var entry : snapshot.mountsByMountPoint().entrySet()) {
            var mount = entry.getValue();
            byMountPoint.put(
                    entry.getKey(),
                    new MountStorageInfo(
                            mount.mountPoint(),
                            mount.source(),
                            mount.filesystemType(),
                            mapLocalClass(mount.storageClass()),
                            mount.storageClass().name()
                    )
            );
        }

        return new StorageSnapshot<>(
                snapshot,
                CloudProvider.LOCAL_OR_UNKNOWN,
                false,
                null,
                snapshot.osName(),
                snapshot.osName(),
                Collections.unmodifiableMap(byMountPoint)
        );
    }

    private static StorageClass mapAwsClass(StorageLayoutUtil.StorageClass storageClass) {
        switch (storageClass) {
            case EBS_COLD_HDD:
                return StorageClass.BLOCK_HDD_COLD;
            case EBS_THROUGHPUT_HDD:
                return StorageClass.BLOCK_HDD_THROUGHPUT;
            case EBS_MAGNETIC:
                return StorageClass.BLOCK_HDD_STANDARD;
            case EBS_GP2:
                return StorageClass.BLOCK_SSD_BALANCED;
            case EBS_GP3:
                return StorageClass.BLOCK_SSD_GENERAL;
            case EBS_PROVISIONED_IOPS_SSD:
                return StorageClass.BLOCK_SSD_HIGH_IOPS;
            case INSTANCE_STORE_SSD:
                return StorageClass.LOCAL_SSD;
            case INSTANCE_STORE_NVME:
                return StorageClass.LOCAL_NVME;
            case NETWORK_FILESYSTEM:
                return StorageClass.NETWORK_FILESYSTEM;
            case MEMORY_TMPFS:
                return StorageClass.MEMORY_TMPFS;
            case PSEUDO_FILESYSTEM:
                return StorageClass.PSEUDO_FILESYSTEM;
            case UNKNOWN:
            default:
                return StorageClass.UNKNOWN;
        }
    }

    private static StorageClass mapGcpClass(GcpStorageLayoutUtil.StorageClass storageClass) {
        switch (storageClass) {
            case PD_STANDARD_HDD:
                return StorageClass.BLOCK_HDD_STANDARD;
            case PD_THROUGHPUT_OPTIMIZED:
                return StorageClass.BLOCK_HDD_THROUGHPUT;
            case PD_BALANCED_SSD:
                return StorageClass.BLOCK_SSD_BALANCED;
            case PD_SSD:
                return StorageClass.BLOCK_SSD_GENERAL;
            case PD_EXTREME_SSD:
                return StorageClass.BLOCK_SSD_HIGH_IOPS;
            case LOCAL_SSD:
                return StorageClass.LOCAL_SSD;
            case LOCAL_NVME:
                return StorageClass.LOCAL_NVME;
            case NETWORK_FILESYSTEM:
                return StorageClass.NETWORK_FILESYSTEM;
            case MEMORY_TMPFS:
                return StorageClass.MEMORY_TMPFS;
            case PSEUDO_FILESYSTEM:
                return StorageClass.PSEUDO_FILESYSTEM;
            case UNKNOWN:
            default:
                return StorageClass.UNKNOWN;
        }
    }

    private static StorageClass mapLocalClass(LocalStorageLayoutUtil.StorageClass storageClass) {
        switch (storageClass) {
            case LOCAL_HDD:
                return StorageClass.BLOCK_HDD_STANDARD;
            case LOCAL_SSD:
                return StorageClass.LOCAL_SSD;
            case LOCAL_NVME:
                return StorageClass.LOCAL_NVME;
            case NETWORK_FILESYSTEM:
                return StorageClass.NETWORK_FILESYSTEM;
            case MEMORY_TMPFS:
                return StorageClass.MEMORY_TMPFS;
            case PSEUDO_FILESYSTEM:
                return StorageClass.PSEUDO_FILESYSTEM;
            case UNKNOWN:
            default:
                return StorageClass.UNKNOWN;
        }
    }
}
