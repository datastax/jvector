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
