package io.github.jbellis.jvector.example.benchmarks;

public abstract class AbstractQueryBenchmark implements QueryBenchmark {
    private String fmt;

    AbstractQueryBenchmark(String fmt) {
        this.fmt = fmt;
    }

    @Override
    public void setPrintPrecision(String fmt) {
        this.fmt = fmt;
    }

    @Override
    public String getPrintPrecision() {
        return fmt;
    }
}
