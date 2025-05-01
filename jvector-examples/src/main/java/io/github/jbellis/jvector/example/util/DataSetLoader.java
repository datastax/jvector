package io.github.jbellis.jvector.example.util;

import java.io.IOException;

public class DataSetLoader {
    public static DataSet loadDataSet(String fileName) throws IOException {
        DataSet ds;
        if (fileName.endsWith(".hdf5")) {
            DownloadHelper.maybeDownloadHdf5(fileName);
            ds = Hdf5Loader.load(fileName);
        } else {
            var mfd = DownloadHelper.maybeDownloadFvecs(fileName);
            ds = mfd.load();
        }
        return ds;
    }
}
