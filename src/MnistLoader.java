package src;

import java.io.*;
import java.util.zip.GZIPInputStream;

/**
 * Parses the official MNIST file format (IDX) into plain Java arrays.
 *
 * MNIST is distributed as 4 files (often gzipped, .gz extension):
 *   train-images-idx3-ubyte(.gz)  - 60000 training images
 *   train-labels-idx1-ubyte(.gz)  - 60000 training labels
 *   t10k-images-idx3-ubyte(.gz)   - 10000 test images
 *   t10k-labels-idx1-ubyte(.gz)   - 10000 test labels
 *
 * This class reads them directly -- it works whether the files are still
 * gzipped or already decompressed, since it auto-detects by file extension.
 */
public class MnistLoader {

    /**
     * Reads an IDX image file.
     * Returns a double[numImages][784] array. Each row is one image,
     * flattened row-major (pixel 0 = top-left), with values normalized
     * from 0-255 to 0.0-1.0 -- this normalization matters because neural
     * networks train far more reliably on small input values than on
     * raw 0-255 integers.
     */
    public static double[][] loadImages(String filePath) throws IOException {
        try (DataInputStream in = openStream(filePath)) {
            int magic = in.readInt();
            if (magic != 2051) {
                throw new IOException("Not an MNIST image file (bad magic number: " + magic + ")");
            }

            int numImages = in.readInt();
            int numRows = in.readInt();
            int numCols = in.readInt();
            int imageSize = numRows * numCols; // 28*28 = 784

            double[][] images = new double[numImages][imageSize];
            for (int i = 0; i < numImages; i++) {
                for (int p = 0; p < imageSize; p++) {
                    int pixel = in.readUnsignedByte(); // 0-255
                    images[i][p] = pixel / 255.0;       // normalize to 0.0-1.0
                }
            }
            return images;
        }
    }

    /**
     * Reads an IDX label file.
     * Returns a double[numLabels][10] array where each row is a one-hot
     * vector. E.g. label 3 becomes [0,0,0,1,0,0,0,0,0,0].
     * One-hot is the format NeuralNetwork.backward() expects as "target".
     */
    public static double[][] loadLabelsOneHot(String filePath) throws IOException {
        try (DataInputStream in = openStream(filePath)) {
            int magic = in.readInt();
            if (magic != 2049) {
                throw new IOException("Not an MNIST label file (bad magic number: " + magic + ")");
            }

            int numLabels = in.readInt();
            double[][] labels = new double[numLabels][10];
            for (int i = 0; i < numLabels; i++) {
                int digit = in.readUnsignedByte(); // 0-9
                labels[i][digit] = 1.0;
            }
            return labels;
        }
    }

    /**
     * Same as loadLabelsOneHot, but returns plain int digits instead of
     * one-hot vectors. Handy for comparing against predict() during testing.
     */
    public static int[] loadLabelsRaw(String filePath) throws IOException {
        try (DataInputStream in = openStream(filePath)) {
            int magic = in.readInt();
            if (magic != 2049) {
                throw new IOException("Not an MNIST label file (bad magic number: " + magic + ")");
            }

            int numLabels = in.readInt();
            int[] labels = new int[numLabels];
            for (int i = 0; i < numLabels; i++) {
                labels[i] = in.readUnsignedByte();
            }
            return labels;
        }
    }

    // Opens the file as a DataInputStream, transparently decompressing
    // if the filename ends in .gz. DataInputStream reads multi-byte
    // numbers (readInt, etc.) in big-endian order, which is exactly the
    // byte order MNIST files use.
    private static DataInputStream openStream(String filePath) throws IOException {
        InputStream raw = new BufferedInputStream(new FileInputStream(filePath));
        if (filePath.endsWith(".gz")) {
            raw = new GZIPInputStream(raw);
        }
        return new DataInputStream(raw);
    }

    /** Quick sanity check you can run directly: prints shapes and one label. */
    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.out.println("Usage: java MnistLoader <images-file> <labels-file>");
            return;
        }
        double[][] images = loadImages(args[0]);
        int[] labels = loadLabelsRaw(args[1]);

        System.out.println("Loaded " + images.length + " images, each length " + images[0].length);
        System.out.println("Loaded " + labels.length + " labels");
        System.out.println("First label: " + labels[0]);

        // Print the first image as ASCII art so you can visually confirm
        // the pixel data and label actually match.
        for (int row = 0; row < 28; row++) {
            StringBuilder sb = new StringBuilder();
            for (int col = 0; col < 28; col++) {
                double v = images[0][row * 28 + col];
                sb.append(v > 0.5 ? '#' : (v > 0.1 ? '.' : ' '));
            }
            System.out.println(sb);
        }
    }
}
