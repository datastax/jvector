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

package io.github.jbellis.jvector.example.testrig.commands;

import io.nosqlbench.command.datasets.CMD_datasets;
import picocli.AutoComplete;
import picocli.CommandLine;

@CommandLine.Command(name = "testrig",
    header = "JVector Test Rig",
    mixinStandardHelpOptions = true,
    description = "JVector Test Rig",
    subcommands = {CommandLine.HelpCommand.class, AutoComplete.GenerateCompletion.class,
                   Run_CMD.class, CMD_datasets.class})
public class TestRig_CMD {

  public static void main(String[] args) {
    @SuppressWarnings("InstantiationOfUtilityClass") TestRig_CMD command = new TestRig_CMD();
             CommandLine commandLine =
             new CommandLine(command).setCaseInsensitiveEnumValuesAllowed(true)
        .setOptionsCaseInsensitive(true);
    int exitCode = commandLine.execute(args);
    System.exit(exitCode);
  }

}
