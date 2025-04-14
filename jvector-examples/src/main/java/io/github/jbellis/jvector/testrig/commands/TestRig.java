package io.github.jbellis.jvector.testrig.commands;

import io.nosqlbench.nbvectors.commands.datasets.CMD_datasets;
import picocli.AutoComplete;
import picocli.CommandLine;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

@CommandLine.Command(name = "testrig", mixinStandardHelpOptions = true, description = """
    JVector Test Rig

    To generate command-line completion scripts use the
    `generate-completion` command, and redirect the output to a file,
    then source the file into your shell. (Known to work with bash,
    and likely works with other shells too)

    You may need to add the directory containing your testrig script
    to your path if it is not already there, so that completion matching
    for associated commands works reliably.
    """,

    subcommands = {
        CommandLine.HelpCommand.class, AutoComplete.GenerateCompletion.class, Run.class,
        CMD_datasets.class
    })
public class TestRig implements Callable<Integer> {

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
    CommandLine commandLine = new CommandLine(command).setCaseInsensitiveEnumValuesAllowed(true)
        .setOptionsCaseInsensitive(true);
    int exitCode = commandLine.execute(args);
    System.exit(exitCode);
  }

  @Override
  public Integer call() throws Exception {

    System.out.println("Test Rig had no commands.");

    return 0;
  }
}
