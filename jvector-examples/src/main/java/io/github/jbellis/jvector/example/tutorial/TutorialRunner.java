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

package io.github.jbellis.jvector.example.tutorial;

import java.io.IOException;

public class TutorialRunner {
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please pick an example");
        }
        String[] forwardArgs = new String[args.length - 1];
        for (int i = 0; i < forwardArgs.length; i++) {
            forwardArgs[i] = args[i + 1];
        }

        switch (args[0]) {
            case "intro":
                VectorIntro.main(forwardArgs);
                break;
            case "disk":
                DiskIntro.main(forwardArgs);
                break;
            case "ltm":
                LargerThanMemory.main(forwardArgs);
                break;
            default:
                throw new IllegalArgumentException("Unknown example" + args[0]);
        }
    }
}
