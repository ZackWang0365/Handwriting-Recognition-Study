package src;

import java.io.*;
import java.util.Random;

/**
 * A minimal feedforward neural network: 784 inputs -> hidden layer -> 10 outputs.
 * One hidden layer, ReLU activation on the hidden layer, softmax on the output.
 *
 * This class knows nothing about MNIST or images -- it just operates on
 * plain double[] vectors. That separation keeps it reusable: the same class
 * works whether the 784-length input comes from MNIST files or from your
 * drawing canvas.
 */
public class NeuralNetwork {

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    // Weight matrices and bias vectors. These are exactly what
    // "training" adjusts and what "inference" reads from.
    // W1: hiddenSize x inputSize   b1: hiddenSize
    // W2: outputSize x hiddenSize  b2: outputSize
    private double[][] W1;
    private double[] b1;
    private double[][] W2;
    private double[] b2;

    // Cached values from the most recent forward pass.
    // We need these during backpropagation, so we keep them as fields
    // rather than passing them around.
    private double[] lastInput;
    private double[] lastHiddenPreActivation; // before ReLU
    private double[] lastHidden;               // after ReLU
    private double[] lastOutput;               // after softmax

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        W1 = new double[hiddenSize][inputSize];
        b1 = new double[hiddenSize];
        W2 = new double[outputSize][hiddenSize];
        b2 = new double[outputSize];

        randomizeWeights();
    }

    // Small random weights to start. Starting at exactly 0 would make every
    // neuron in a layer learn the same thing (no diversity to break symmetry).
    private void randomizeWeights() {
        Random rand = new Random();
        double w1Scale = Math.sqrt(2.0 / inputSize);
        double w2Scale = Math.sqrt(2.0 / hiddenSize);

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] = rand.nextGaussian() * w1Scale;
            }
        }
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] = rand.nextGaussian() * w2Scale;
            }
        }
        // Biases start at 0 -- this is fine, asymmetry already comes from W1/W2.
    }

    /**
     * Forward pass: input -> hidden (ReLU) -> output (softmax).
     * Returns the output probabilities (length = outputSize).
     * Also caches intermediate values needed by backward().
     */
    public double[] forward(double[] input) {
        lastInput = input;

        // hiddenPreActivation = W1 * input + b1
        lastHiddenPreActivation = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double sum = b1[i]; //define one by one, the entries of b1
            for (int j = 0; j < inputSize; j++) {
                sum += W1[i][j] * input[j];//sum, or entries of b1, are the result of matrix mult plus itself
            }
            lastHiddenPreActivation[i] = sum;//result of matrix mult and b1 itself give to lastHiddenPreActivation
        }

        // hidden = ReLU(hiddenPreActivation)
        lastHidden = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            lastHidden[i] = Math.max(0, lastHiddenPreActivation[i]);
        }

        // outputPreActivation = W2 * hidden + b2
        double[] outputPreActivation = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = b2[i];
            for (int j = 0; j < hiddenSize; j++) {
                sum += W2[i][j] * lastHidden[j];
            }
            outputPreActivation[i] = sum;
        }

        // output = softmax(outputPreActivation)
        lastOutput = softmax(outputPreActivation);
        return lastOutput;
    }

    private double[] softmax(double[] logits) {
        // Subtract max for numerical stability (avoids overflow in Math.exp).
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logits) max = Math.max(max, v);

        double[] expValues = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < logits.length; i++) {
            expValues[i] = Math.exp(logits[i] - max);
            sum += expValues[i];
        }
        for (int i = 0; i < logits.length; i++) {
            expValues[i] /= sum;
        }
        return expValues;
    }

    /**
     * Backward pass (backpropagation) + weight update, all in one step.
     * target: one-hot encoded correct answer (e.g. digit 3 -> [0,0,0,1,0,0,0,0,0,0])
     * learningRate: how big a step to take when adjusting weights.
     *
     * Must be called right after forward() on the same input, since it uses
     * the cached lastInput/lastHidden/lastOutput.
     */
    public void backward(double[] target, double learningRate) {
        // --- Output layer gradient ---
        // For softmax + cross-entropy loss combined, the gradient w.r.t.
        // the output pre-activation simplifies neatly to (predicted - target).
        double[] outputError = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputError[i] = lastOutput[i] - target[i];
        }

        // Gradients for W2, b2.
        double[][] gradW2 = new double[outputSize][hiddenSize];
        double[] gradB2 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            gradB2[i] = outputError[i];
            for (int j = 0; j < hiddenSize; j++) {
                gradW2[i][j] = outputError[i] * lastHidden[j];
            }
        }

        // --- Hidden layer gradient ---
        // Propagate error backward through W2, then through the ReLU derivative.
        double[] hiddenError = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double sum = 0;
            for (int i = 0; i < outputSize; i++) {
                sum += outputError[i] * W2[i][j];
            }
            // ReLU derivative: 1 if pre-activation was > 0, else 0.
            double reluDerivative = lastHiddenPreActivation[j] > 0 ? 1.0 : 0.0;
            hiddenError[j] = sum * reluDerivative;
        }

        // Gradients for W1, b1.
        double[][] gradW1 = new double[hiddenSize][inputSize];
        double[] gradB1 = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            gradB1[j] = hiddenError[j];
            for (int k = 0; k < inputSize; k++) {
                gradW1[j][k] = hiddenError[j] * lastInput[k];
            }
        }

        // --- Apply gradient descent update: weight -= learningRate * gradient ---
        for (int i = 0; i < outputSize; i++) {
            b2[i] -= learningRate * gradB2[i];
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] -= learningRate * gradW2[i][j];
            }
        }
        for (int j = 0; j < hiddenSize; j++) {
            b1[j] -= learningRate * gradB1[j];
            for (int k = 0; k < inputSize; k++) {
                W1[j][k] -= learningRate * gradW1[j][k];
            }
        }
    }

    /** Convenience: returns the predicted digit (0-9) for a given input. */
    public int predict(double[] input) {
        double[] output = forward(input);
        int bestIndex = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[bestIndex]) bestIndex = i;
        }
        return bestIndex;
    }

    /** Saves all weights/biases to a simple binary file. */
    public void saveWeights(String filePath) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(filePath)))) {
            writeMatrix(out, W1);
            writeVector(out, b1);
            writeMatrix(out, W2);
            writeVector(out, b2);
        }
    }

    /** Loads weights/biases previously saved with saveWeights(). */
    public void loadWeights(String filePath) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {
            readMatrix(in, W1);
            readVector(in, b1);
            readMatrix(in, W2);
            readVector(in, b2);
        }
    }

    private void writeMatrix(DataOutputStream out, double[][] m) throws IOException {
        for (double[] row : m) {
            for (double v : row) out.writeDouble(v);
        }
    }

    private void writeVector(DataOutputStream out, double[] v) throws IOException {
        for (double x : v) out.writeDouble(x);
    }

    private void readMatrix(DataInputStream in, double[][] m) throws IOException {
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                m[i][j] = in.readDouble();
            }
        }
    }

    private void readVector(DataInputStream in, double[] v) throws IOException {
        for (int i = 0; i < v.length; i++) {
            v[i] = in.readDouble();
        }
    }
}
