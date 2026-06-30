package src;

import java.io.IOException;

/**
 * Trains a NeuralNetwork on MNIST and saves the learned weights to disk.
 * Run this once during development -- it is NOT part of the drawing app.
 * After it finishes, weights.bin is loaded by the recognition code at
 * runtime; no training happens when the Swing app starts.
 */
public class MnistTrainer {

    public static void main(String[] args) throws IOException {
        // --- 1. Load data ---
        System.out.println("Loading MNIST data...");
        double[][] trainImages = MnistLoader.loadImages("data/train-images.idx3-ubyte");
        double[][] trainLabels = MnistLoader.loadLabelsOneHot("data/train-labels.idx1-ubyte");
        double[][] testImages = MnistLoader.loadImages("data/t10k-images.idx3-ubyte");
        int[] testLabels = MnistLoader.loadLabelsRaw("data/t10k-labels.idx1-ubyte");
        System.out.println("Loaded " + trainImages.length + " training images, "
                + testImages.length + " test images.");

        // --- 2. Build the network ---
        // 784 inputs (28x28 pixels), 128 hidden neurons, 10 outputs (digits 0-9).
        NeuralNetwork net = new NeuralNetwork(784, 128, 10);

        // --- 3. Train ---
        int epochs = 10;
        double learningRate = 0.05;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle the training order each epoch so the network doesn't
            // learn anything from the fixed ordering of the dataset.
            int[] order = shuffledIndices(trainImages.length);

            for (int idx : order) {
                // The training part, uses net, a NeuralNetwork class's object
                net.forward(trainImages[idx]);
                net.backward(trainLabels[idx], learningRate);
            }

            double accuracy = evaluate(net, testImages, testLabels);
            System.out.printf("Epoch %d/%d - test accuracy: %.2f%%%n",
                    epoch + 1, epochs, accuracy * 100);
        }

        // --- 4. Save weights for the recognition app to load later ---
        net.saveWeights("weights.bin");
        System.out.println("Saved trained weights to weights.bin");
    }

    // Fisher-Yates shuffle of indices 0..n-1.
    private static int[] shuffledIndices(int n) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        java.util.Random rand = new java.util.Random();
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        return indices;
    }

    // Runs predict() on every test image and returns the fraction correct.
    private static double evaluate(NeuralNetwork net, double[][] testImages, int[] testLabels) {
        int correct = 0;
        for (int i = 0; i < testImages.length; i++) {
            int predicted = net.predict(testImages[i]);
            if (predicted == testLabels[i]) correct++;
        }
        return (double) correct / testImages.length;
    }
}
