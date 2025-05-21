package io.github.jbellis.jvector.example.testrig.commands;

import io.nosqlbench.command.datasets.CMD_datasets;
import picocli.AutoComplete;
import picocli.CommandLine;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;

@CommandLine.Command(name = "testrig",
    mixinStandardHelpOptions = true,
    description = "JVector Test Rig",
    subcommands = {CommandLine.HelpCommand.class, AutoComplete.GenerateCompletion.class,
                   Run.class, CMD_datasets.class})
public class TestRig {

  @CommandLine.Option(names = {"-d", "--dataset"}, description = "Dataset to use")
  private String dataset;

  @CommandLine.Option(names = {"--catalog"},
      description = "A directory, remote url, or other catalog container",
      defaultValue = "https://jvector-datasets-public.s3.us-east-1.amazonaws.com/")
  private List<URL> catalogs = new ArrayList<>();

  @CommandLine.Parameters(description = "test execution commands")
  private List<String> commands = new ArrayList<>();

  private static enum Verbs {
    datasets;
  }

  public static void main(String[] args) {
    TestRig command = new TestRig();
             CommandLine commandLine =
             new CommandLine(command).setCaseInsensitiveEnumValuesAllowed(true)
        .setOptionsCaseInsensitive(true);
    int exitCode = commandLine.execute(args);
    System.exit(exitCode);
  }

}
