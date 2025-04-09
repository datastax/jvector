package io.github.jbellis.jvector.testrig.commands;

import io.github.jbellis.jvector.testrig.BenchHarness;
import io.nosqlbench.vectordata.VectorTestData;
import io.nosqlbench.vectordata.download.DatasetEntry;
import picocli.CommandLine;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

@CommandLine.Command(name = "run", description = "Run a testrig command")
public class Run implements Callable<Integer> {

  public static class ExpanderExample implements Iterable<String> {
    @Override
    public java.util.Iterator<String> iterator() {
      return VectorTestData.catalogs().find().datasets().stream().map(DatasetEntry::name).toList()
          .iterator();
    }
  }

  @CommandLine.Option(names = {"--catalog"},
      description = "A directory, remote url, or other catalog container",
      defaultValue = "https://jvector-datasets-public.s3.us-east-1.amazonaws.com/")
  List<URL> catalogs = new ArrayList<>();

  @CommandLine.Option(names = {"-d", "--dataset"},
      description = "Dataset to use",
      completionCandidates = ExpanderExample.class)
  private List<String> dsnames;

  @CommandLine.Option(names = {"-p", "--profile"},
      description = "Profile to use",
      defaultValue = "default")
  private String profile = "default";

  public static void main(String[] args) {
    TestRig command = new TestRig();
    CommandLine commandLine = new CommandLine(command).setCaseInsensitiveEnumValuesAllowed(true)
        .setOptionsCaseInsensitive(true);
    int exitCode = commandLine.execute(args);
    System.exit(exitCode);
  }

  @Override
  public Integer call() throws Exception {
    System.out.println("Test Rig run with datasets: " + dsnames);
    for (String dsname : dsnames) {
      String[] nameparts = dsname.split(":+", 2);
      String _name = null, _profile = this.profile;
      switch (nameparts.length) {
        case 2:
          _profile = nameparts[1];
        case 1:
          _name = nameparts[0];
          break;
      }

      System.out.println("Using testdata source " + _name);
      System.out.println("Using profile " + _profile);

      DatasetEntry ds = VectorTestData.catalogs().find().findExact(_name).orElseThrow();
      BenchHarness harness = new BenchHarness(ds, _profile);
      harness.run();

    }
    return 0;
  }
}
