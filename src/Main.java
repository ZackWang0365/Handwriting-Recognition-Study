package src;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * Entry point. Builds the window, loads the trained network once at
 * startup, and wires up the buttons. Run this class (it has main) --
 * not DrawingCanvas.
 */
public class Main {

    public static void main(String[] args) {
        // Load the network once, when the app starts -- not on every click.
        // This is inference-only: no training happens here, just reading the
        // weights that MnistTrainer already learned and saved.
        NeuralNetwork net = new NeuralNetwork(784, 128, 10);
        try {
            net.loadWeights("weights.bin");
        } catch (IOException e) {
            System.err.println("Could not load weights.bin -- run MnistTrainer first.");
            e.printStackTrace();
            return;
        }

        JFrame frame = new JFrame("Handwriting Digit Recognizer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        DrawingCanvas canvas = new DrawingCanvas(280, 280);

        JLabel resultLabel = new JLabel("Draw a digit, then click Recognize");
        resultLabel.setFont(resultLabel.getFont().deriveFont(18f));
        resultLabel.setHorizontalAlignment(SwingConstants.CENTER);

        JButton clearBtn = new JButton("Clear");
        clearBtn.addActionListener(e -> {
            canvas.clear();
            resultLabel.setText("Draw a digit, then click Recognize");
        });

        JButton recognizeBtn = new JButton("Recognize");
        recognizeBtn.addActionListener(e -> {
            BufferedImage img = canvas.getCanvasImage();//get user's drawing

            // this is where preprocessing + model
            // inference will go in the next step.
            double[] input = ImagePreprocessor.toMnistInput(img);

            if (input == null) {
                resultLabel.setText("Canvas is empty -- draw something first");
                return;
            }
            
            System.out.println("Captured image: " + img.getWidth() + "x" + img.getHeight());
            int predicted = net.predict(input);
            resultLabel.setText("Predicted digit: " + predicted);
        });

        // Setup user interface
        JPanel controls = new JPanel();
        controls.add(clearBtn);
        controls.add(recognizeBtn);

        frame.setLayout(new BorderLayout());
        frame.add(canvas, BorderLayout.CENTER);
        frame.add(controls, BorderLayout.SOUTH);
        frame.add(resultLabel, BorderLayout.NORTH);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
