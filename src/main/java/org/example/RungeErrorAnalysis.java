package org.example;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.ArrayList;
import java.util.List;

public class RungeErrorAnalysis {

    static final double A = 2;
    static final double B = -1;
    static final double C = -1;

    public static double calculateErrorNorm(RealVector yNumerical, RealVector yAnalytical) {
        return yNumerical.subtract(yAnalytical).getNorm();
    }

    public static RealVector analyticalSolution(double x) {
        return new ArrayRealVector(new double[]{
                Math.exp(x),
                A - C * Math.sin(x),
                A + C * Math.sin(x),
                Math.exp(x)
        });
    }

    public static RealVector rungeKuttaStep(RungeKutta.VectorFunction f, double x, RealVector y, double h, RealMatrix a, double[] b, double[] c) {
        int s = b.length;
        RealVector[] K = new RealVector[s];
        for (int i = 0; i < s; i++) {
            RealVector sum = new ArrayRealVector(y.getDimension());
            for (int j = 0; j < i; j++) {
                sum = sum.add(K[j].mapMultiply(h * a.getEntry(i, j)));
            }
            K[i] = f.apply(x + c[i] * h, y.add(sum));
        }

        RealVector nextY = y;
        for (int j = 0; j < s; j++) {
            nextY = nextY.add(K[j].mapMultiply(h * b[j]));
        }
        return nextY;
    }

    public static RealVector solveWithConstantStep(RungeKutta.VectorFunction f, RealVector y0, double[] segment, double h, RealMatrix a, double[] b, double[] c) {
        double x = segment[0];
        RealVector y = y0.copy();

        while (x < segment[1]) {
            y = rungeKuttaStep(f, x, y, h, a, b, c);
            x += h;
        }
        return y;
    }

    public static double findOptimalStep(RungeKutta.VectorFunction f, RealVector y0, double[] segment, RealMatrix a, double[] b, double[] c, double tolerance) {
        double h = 1.0; // Initial guess for step size
        double factor = 0.5; // Reduction factor
        double prevErrorNorm = Double.MAX_VALUE;

        while (true) {
            RealVector yNumerical = solveWithConstantStep(f, y0, segment, h, a, b, c);
            RealVector yAnalytical = analyticalSolution(segment[1]);
            double errorNorm = calculateErrorNorm(yNumerical, yAnalytical);

            if (errorNorm <= tolerance || errorNorm >= prevErrorNorm) {
                return h;
            }

            prevErrorNorm = errorNorm;
            h *= factor; // Reduce the step size
        }
    }

    public static void main(String[] args) {
        double[] segment = {0, 5};
        RealVector y0 = new ArrayRealVector(new double[]{1, A, A, 1});

        double[][] aData = {{0, 0, 0}, {0.5, 0, 0}, {-1, 2, 0}};
        double[] bData = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
        double[] cData = {0, 0.5, 1};

        RealMatrix a = new Array2DRowRealMatrix(aData);
        double[] b = bData;
        double[] c = cData;

        RungeKutta.VectorFunction f = (x, y) -> {
            RealVector result = new ArrayRealVector(y.getDimension());
            result.setEntry(0, y.getEntry(0)); // dy/dx = y (пример для простоты)
            result.setEntry(1, y.getEntry(1)); // Просто для демонстрации.
            return result;
        };

        List<Double> stepSizes = new ArrayList<>();
        List<Double> errorNorms = new ArrayList<>();

        for (int k = 0; k <= 6; k++) {
            double h = 1.0 / Math.pow(2, k);
            RealVector yNumerical = solveWithConstantStep(f, y0, segment, h, a, b, c);
            RealVector yAnalytical = analyticalSolution(segment[1]);
            double errorNorm = calculateErrorNorm(yNumerical, yAnalytical);

            stepSizes.add(h);
            errorNorms.add(errorNorm);
        }

        double optimalStep = findOptimalStep(f, y0, segment, a, b, c, 1e-6);
        System.out.println("Optimal step size: " + optimalStep);

        XYSeries errorSeries = new XYSeries("Error Norm");
        for (int i = 0; i < stepSizes.size(); i++) {
            errorSeries.add(Math.log(stepSizes.get(i)), Math.log(errorNorms.get(i)));
        }

        XYSeries referenceLine = new XYSeries("Reference (Slope 2)");
        for (int i = 0; i < stepSizes.size(); i++) {
            double logH = Math.log(stepSizes.get(i));
            referenceLine.add(logH, 2 * logH + Math.log(errorNorms.get(0)));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(errorSeries);
        dataset.addSeries(referenceLine);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Log-Log Error Analysis",
                "Log(h)",
                "Log(Error Norm)",
                dataset
        );

        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }
}
