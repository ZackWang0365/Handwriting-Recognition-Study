package src;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Converts a drawn canvas image into the exact format the network expects:
 * a 784-length double[] matching MNIST's pixel convention.
 *
 * MNIST images are: 28x28, digit drawn in WHITE (high values) on a BLACK
 * background (low values), with the digit already cropped and centered
 * by its CENTER OF MASS (not just the bounding box's geometric center --
 * this matters a lot for asymmetric digits like 6, 7, 9).
 */
public class ImagePreprocessor {

    /**
     * Main entry point. Returns null if the canvas is blank (nothing drawn) --
     * callers should check for that before calling predict().
     */
    public static double[] toMnistInput(BufferedImage canvasImage) {
        int width = canvasImage.getWidth();
        int height = canvasImage.getHeight();

        // Step 1: find the bounding box of the ink (anything not close to white).
        int minX = width, minY = height, maxX = -1, maxY = -1;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = grayscaleAt(canvasImage, x, y);
                if (gray < 240) { // not pure white -> part of a stroke
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (maxX < 0) {
            return null; // nothing drawn
        }

        // Step 2: pad the bounding box and make it square, so the digit's
        // aspect ratio isn't distorted when we scale it down. This uses the
        // bounding box's geometric center just to decide the CROP region --
        // that's fine for sizing, the real centering fix happens in step 4.
        int boxWidth = maxX - minX + 1;
        int boxHeight = maxY - minY + 1;
        int padding = (int) (Math.max(boxWidth, boxHeight) * 0.3);

        int side = Math.max(boxWidth, boxHeight) + padding * 2;
        int bboxCenterX = (minX + maxX) / 2;
        int bboxCenterY = (minY + maxY) / 2;

        int cropX = bboxCenterX - side / 2;
        int cropY = bboxCenterY - side / 2;

        // Step 3: scale that square region down into a 28x28 image.
        BufferedImage scaled = new BufferedImage(28, 28, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = scaled.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, 28, 28);
        g2.drawImage(canvasImage,
                0, 0, 28, 28,
                cropX, cropY, cropX + side, cropY + side,
                null);
        g2.dispose();

        // Step 4: re-center by CENTER OF MASS, not bounding box.
        // Asymmetric digits (6, 7, 9) have their ink weight concentrated
        // toward one end -- centering by the bounding box's geometry alone
        // leaves them visually off-center compared to how MNIST normalizes
        // its training images, which biases the network's prediction.
        double[] grayValues = new double[28 * 28]; // 0 (background) - 1 (ink), pre-shift
        double sumX = 0, sumY = 0, sumMass = 0;
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int gray = grayscaleAt(scaled, x, y);
                double ink = (255 - gray) / 255.0; // 0 = background, 1 = full ink
                grayValues[y * 28 + x] = ink;
                sumX += x * ink;
                sumY += y * ink;
                sumMass += ink;
            }
        }

        double[] input = new double[28 * 28];
        if (sumMass > 0) {
            double centroidX = sumX / sumMass;
            double centroidY = sumY / sumMass;
            // How far to shift so the centroid lands on (13.5, 13.5), the
            // center of a 28x28 grid.
            int shiftX = (int) Math.round(13.5 - centroidX);
            int shiftY = (int) Math.round(13.5 - centroidY);

            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int srcX = x - shiftX;
                    int srcY = y - shiftY;
                    if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
                        input[y * 28 + x] = grayValues[srcY * 28 + srcX];
                    } else {
                        input[y * 28 + x] = 0.0; // shifted-in area is background
                    }
                }
            }
        } else {
            input = grayValues;
        }

        return input;
    }

    // Averages the RGB channels at (x,y) into a single 0-255 grayscale value.
    private static int grayscaleAt(BufferedImage img, int x, int y) {
        int rgb = img.getRGB(x, y);
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        return (r + g + b) / 3;
    }

    /**
     * Debug helper: saves the exact 28x28 input the network sees as a PNG,
     * scaled up 10x so it's easy to inspect. Call this from Main.java's
     * Recognize button temporarily if predictions still look wrong --
     * if the saved image doesn't look like a clean, centered digit, the
     * bug is in preprocessing; if it does look right, the bug is more
     * likely in training (e.g. needs more epochs).
     */
    public static void saveDebugImage(double[] input, String filePath) throws java.io.IOException {
        int scale = 10;
        BufferedImage debugImg = new BufferedImage(28 * scale, 28 * scale, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = debugImg.createGraphics();
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                double ink = input[y * 28 + x]; // 0 = background, 1 = ink
                int gray = (int) (255 - ink * 255);
                g2.setColor(new Color(gray, gray, gray));
                g2.fillRect(x * scale, y * scale, scale, scale);
            }
        }
        g2.dispose();
        javax.imageio.ImageIO.write(debugImg, "png", new java.io.File(filePath));
    }
}
