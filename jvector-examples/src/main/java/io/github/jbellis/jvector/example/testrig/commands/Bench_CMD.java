package io.github.jbellis.jvector.example.testrig.commands;

import io.github.jbellis.jvector.example.testrig.BenchHarness;
import io.nosqlbench.vectordata.VectorTestData;
import io.nosqlbench.vectordata.discovery.TestDataSources;
import io.nosqlbench.vectordata.downloader.Catalog;
import io.nosqlbench.vectordata.downloader.DatasetEntry;
import picocli.CommandLine;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

@CommandLine.Command(name = "bench", description = "run example benchmarks")
public class Bench_CMD implements Callable<Integer> {

  public static class ExpanderExample implements Iterable<String> {
    @Override
    public java.util.Iterator<String> iterator() {
      return VectorTestData.catalogs().catalog().datasets().stream().map(DatasetEntry::name)
          .collect(Collectors.toList()).iterator();
    }
  }

  @CommandLine.Option(names = {"--catalog"},
      description = "A directory, remote url, or other catalog container")
  List<URL> catalogs = new ArrayList<>();

    @CommandLine.Option(names = {"--optional-catalog"},
            description = "A configuration file to use",
            split = ",",
            defaultValue = "~/.config/jvector/catalogs.yaml,~/.config/vectordata/catalogs.yaml")
    List<String> optionalCatalogs;

  @CommandLine.Option(names = {"-d", "--dataset"},
      description = "Dataset to use",
      completionCandidates = ExpanderExample.class)
  private List<String> dsnames;

  @CommandLine.Option(names = {"-p", "--profile"},
      description = "Profile to use",
      defaultValue = "default")
  private String profile = "default";

  @CommandLine.Option(names = {"-c", "--concurrency"},
      description = "Number of concurrent threads",
      defaultValue = "1")
  private int concurrency = 1;

  public static void main(String[] args) {
    Bench_CMD command = new Bench_CMD();
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

      Catalog catalog = new TestDataSources().addOptionalCatalogs(optionalCatalogs).catalog();
      DatasetEntry ds = catalog.findExact(_name).orElseThrow();
      //          VectorTestData.catalogs().catalog().findExact(_name).orElseThrow();
      BenchHarness harness = new BenchHarness(ds, _profile, concurrency);

      harness.run();

    }
    return 0;
  }
}
