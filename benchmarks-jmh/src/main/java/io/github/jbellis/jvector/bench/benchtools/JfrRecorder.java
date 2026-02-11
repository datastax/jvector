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

package io.github.jbellis.jvector.bench.benchtools;

import jdk.jfr.Configuration;
import jdk.jfr.Recording;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;

/**
 * Manages the lifecycle of a JFR (Java Flight Recorder) recording for benchmarks.
 */
public final class JfrRecorder {
    private static final Logger log = LoggerFactory.getLogger(JfrRecorder.class);

    private Recording recording;
    private String fileName;

    /**
     * Creates the output directory, configures a "profile" recording, starts it, and returns the absolute path.
     *
     * @param outputDir directory to write the JFR file into
     * @param fileName  name of the JFR file (e.g. {@code "compactor-foo.jfr"})
     * @return the absolute path of the recording file
     * @throws IOException    if the directory cannot be created
     * @throws ParseException if the JFR "profile" configuration cannot be loaded
     */
    public Path start(Path outputDir, String fileName) throws IOException, ParseException {
        Files.createDirectories(outputDir);
        Path jfrPath = outputDir.resolve(fileName).toAbsolutePath();
        recording = new Recording(Configuration.getConfiguration("profile"));
        recording.setToDisk(true);
        recording.setDestination(jfrPath);
        recording.start();
        this.fileName = fileName;
        System.out.println("JFR recording started, saving to: " + jfrPath);
        log.info("JFR recording started, saving to: {}", jfrPath);
        return jfrPath;
    }

    /** Stops and closes the recording, logging the saved path. */
    public void stop() {
        if (recording != null) {
            recording.stop();
            recording.close();
            Path jfrPath = recording.getDestination();
            recording = null;
            System.out.println("JFR recording saved to: " + jfrPath);
            log.info("JFR recording saved to: {}", jfrPath);
        }
    }

    /** Returns {@code true} if a recording is currently in progress. */
    public boolean isActive() {
        return recording != null;
    }

    /** Returns the current file name, or {@code null} if no recording has been started. */
    public String getFileName() {
        return fileName;
    }
}
