package io.github.jbellis.jvector.disk;

import java.io.DataOutput;
import java.io.IOException;

public interface IndexWriter extends DataOutput {
    long position() throws IOException;
}
