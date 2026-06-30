package src;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;

/**
 * Minimal handwriting canvas.
 * Draw with the mouse; call clear() to wipe it; call getCanvasImage()
 * to grab the current drawing for preprocessing/recognition later.
 *
 * This class only defines the panel itself -- see Main.java for the
 * code that creates the window and runs the app.
 */
public class DrawingCanvas extends JPanel {

    // The actual pixel data we draw into. This is the "source of truth".
    private BufferedImage canvasImage;

    // Tracks the last mouse position so we can draw a line segment
    // from there to the new position on every drag event.
    private int lastX, lastY;

    public DrawingCanvas(int width, int height) {
        setPreferredSize(new Dimension(width, height));
        setBackground(Color.WHITE);

        // Create the in-memory image and fill it white to start.
        canvasImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = canvasImage.createGraphics(); //build a stroke g2 corresponding to canvas canvasImage
        g2.setColor(Color.WHITE); //this stroke is used to set background color
        g2.fillRect(0, 0, width, height);
        g2.dispose();

        // mousePressed: record where the stroke starts.
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                lastX = e.getX();
                lastY = e.getY();
            }
        });

        // mouseDragged: fires repeatedly while the button is held and the
        // mouse moves. We draw a line from the last point to the new point
        // directly onto canvasImage, then ask Swing to repaint the panel.
        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                int x = e.getX();
                int y = e.getY();

                Graphics2D g2 = canvasImage.createGraphics();//build a stroke g2 corresponding to canvas canvasImage
                g2.setColor(Color.BLACK);//this stroke is used to write
                g2.setStroke(new BasicStroke(
                        8, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                g2.drawLine(lastX, lastY, x, y);
                g2.dispose();

                lastX = x;
                lastY = y;

                repaint(); // triggers paintComponent below, the reason is that paintComponent is protected method
            }
        });
    }

    // Swing calls this whenever the panel needs to be redrawn.
    // We don't draw strokes here -- we just blit the stored image.
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(canvasImage, 0, 0, null);
    }

    public void clear() {
        Graphics2D g2 = canvasImage.createGraphics();
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvasImage.getWidth(), canvasImage.getHeight());
        g2.dispose();
        repaint();
    }

    public BufferedImage getCanvasImage() {
        return canvasImage;
    }
}
