package org.example;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.util.function.BiFunction;


public class RungeKutta {

    static final double A = 2;
    static final double B = -1;
    static final double C = -1;

    public static interface VectorFunction extends BiFunction<Double, RealVector, RealVector> {}

    public static RealVector f(double x, RealVector y) {
        double y1 = 2 * x * y.getEntry(3) * Math.pow(y.getEntry(1), 1 / B);
        double y2 = 2 * B * x * Math.exp((B / C) * (y.getEntry(2) - A)) * y.getEntry(3);
        double y3 = 2 * C * x * y.getEntry(3);
        double y4 = -2 * x * Math.log(y.getEntry(0));
        return new ArrayRealVector(new double[]{y1, y2, y3, y4});
    }

    public static RealVector answer(double x) {
        double y1 = Math.exp(Math.sin(x * x));
        double y2 = Math.exp(-1 * Math.sin(x * x));
        double y3 = C * Math.sin(x * x) + A;
        double y4 = Math.cos(x * x);
        return new ArrayRealVector(new double[]{y1, y2, y3, y4});
    }



    public static RealVector[] localMargin(RealVector y, RealVector y2, int p) {
        double factor1 = 1.0 / (1.0 - Math.pow(2, -p));
        double factor2 = 1.0 / (Math.pow(2, p) - 1.0);

        RealVector margin1 = y.subtract(y2).mapMultiply(factor1);
        RealVector margin2 = y.subtract(y2).mapMultiply(factor2);

        return new RealVector[]{margin1, margin2};
    }

    public static RealVector getK(VectorFunction f, double x, RealVector y, double h, int s, RealMatrix a, double[] c) {
        RealVector[] K = new RealVector[s];
        for (int i = 0; i < s; i++) {
            RealVector sum = new ArrayRealVector(y.getDimension());
            for (int j = 0; j < i; j++) {
                sum = sum.add(K[j].mapMultiply(h * a.getEntry(i, j)));
            }
            K[i] = f.apply(x + c[i] * h, y.add(sum));
        }

        // Efficiently combine K vectors using a temporary array
        double[] combinedK = new double[s * y.getDimension()];
        for (int i = 0; i < s; i++) {
            System.arraycopy(K[i].toArray(), 0, combinedK, i * y.getDimension(), y.getDimension());
        }
        return new ArrayRealVector(combinedK);
    }


    public static void runge(VectorFunction f, RealVector cauchyCond, double[] segment, double h, int s, int p, RealMatrix a, double[] b, double[] c, String pngName) throws IOException {
        int n = (int) Math.ceil((segment[1] - segment[0]) / h);

        double[] x = new double[n + 1];
        RealMatrix y = new Array2DRowRealMatrix(n + 1, cauchyCond.getDimension() - 1);


        double[] x2 = new double[n/2 + 1];
        RealMatrix y2 = new Array2DRowRealMatrix(n/2 + 1, cauchyCond.getDimension() - 1);

        x[0] = cauchyCond.getEntry(0);
        y.setRowVector(0, cauchyCond.getSubVector(1, cauchyCond.getDimension() - 1));

        x2[0] = cauchyCond.getEntry(0);
        y2.setRowVector(0, cauchyCond.getSubVector(1, cauchyCond.getDimension() - 1));

        RealMatrix r = new Array2DRowRealMatrix(n/2 + 1, cauchyCond.getDimension() - 1);
        RealMatrix r2 = new Array2DRowRealMatrix(n/2 + 1, cauchyCond.getDimension() - 1);



        for (int i = 0; i < n; i++) {
            RealVector K = getK(f, x[i], y.getRowVector(i), h, s, a, c);

            RealVector nextY = y.getRowVector(i);
            int kDim = y.getRowVector(i).getDimension(); // Dimension of each K_i

            for (int j = 0; j < s; j++) {
                // Correctly extract the subvector for each stage
                nextY = nextY.add(K.getSubVector(j * kDim, kDim).mapMultiply(h * b[j]));
            }

            y.setRowVector(i + 1, nextY);
            x[i + 1] = x[i] + h;



            if (i % 2 == 1) {
                RealVector K2 = getK(f, x2[i/2], y2.getRowVector(i/2), 2 * h, s, a, c);

                RealVector nextY2 = y2.getRowVector(i/2);
                for (int j = 0; j < s; j++) {
                    nextY2 = nextY2.add(K2.getSubVector(j * kDim, kDim).mapMultiply(2 * h * b[j]));
                }

                y2.setRowVector(i/2 + 1, nextY2);
                x2[i/2 + 1] = x2[i/2] + 2 * h;

                RealVector[] rx = localMargin(y.getRowVector(i + 1), y2.getRowVector(i/2 + 1), p);
                r.setRowVector(i/2 + 1, rx[1]);
                r2.setRowVector(i/2 + 1, rx[0]);
            }

        }

        // ... (plotting logic using JFreeChart)
        plotResults(x, y, x2, y2, r, r2, pngName);

    }



    private static void plotResults(double[] x, RealMatrix y, double[] x2, RealMatrix y2, RealMatrix r, RealMatrix r2, String pngName) throws IOException {
        XYSeriesCollection dataset1 = new XYSeriesCollection();
        XYSeriesCollection dataset2 = new XYSeriesCollection();
        XYSeriesCollection dataset3 = new XYSeriesCollection();
        XYSeriesCollection dataset4 = new XYSeriesCollection();

        for (int j = 0; j < y.getColumnDimension(); j++) {
            XYSeries series1 = new XYSeries("y" + (j + 1));
            for (int i = 0; i < x.length; i++) {
                series1.add(x[i], y.getEntry(i, j));
            }
            dataset1.addSeries(series1);

            XYSeries series3 = new XYSeries("y" + (j + 1));
            for (int i = 0; i < x2.length; i++) {
                series3.add(x2[i], y2.getEntry(i, j));
            }
            dataset3.addSeries(series3);

            XYSeries series2 = new XYSeries("r" + (j + 1));
            for (int i = 0; i < x2.length; i++) {
                series2.add(x2[i], r.getEntry(i, j));
            }
            dataset2.addSeries(series2);

            XYSeries series4 = new XYSeries("r2" + (j + 1));
            for (int i = 0; i < x2.length; i++) {
                series4.add(x2[i], r2.getEntry(i, j));
            }
            dataset4.addSeries(series4);
        }

        JFreeChart chart1 = ChartFactory.createXYLineChart("Half Step Method", "x", "y", dataset1, PlotOrientation.VERTICAL, true, true, false);
        JFreeChart chart2 = ChartFactory.createXYLineChart("Half Step Method Error Margin", "x", "r", dataset2, PlotOrientation.VERTICAL, true, true, false);
        JFreeChart chart3 = ChartFactory.createXYLineChart("Normal Step Method", "x", "y", dataset3, PlotOrientation.VERTICAL, true, true, false);
        JFreeChart chart4 = ChartFactory.createXYLineChart("Normal Step Method Error Margin", "x", "r2", dataset4, PlotOrientation.VERTICAL, true, true, false);

        int width = 800;
        int height = 600;
        ChartUtils.saveChartAsPNG(new File(pngName + "_half_step.png"), chart1, width, height);
        ChartUtils.saveChartAsPNG(new File(pngName + "_half_step_error.png"), chart2, width, height);
        ChartUtils.saveChartAsPNG(new File(pngName + "_normal_step.png"), chart3, width, height);
        ChartUtils.saveChartAsPNG(new File(pngName + "_normal_step_error.png"), chart4, width, height);
    }


    public static void rungeKuttaMethod(RealMatrix a, double[] b, double[] c, String pngName) throws IOException {
        RealVector cauchyConditions = new ArrayRealVector(new double[]{0, 1, 1, A, 1});
        double[] segm = new double[]{0, 5};
        runge(RungeKutta::f, cauchyConditions, segm, 1e-2 / 2.0, b.length, b.length, a, b, c, pngName);
    }

    public static void main(String[] args) throws IOException {
        double[][] aData = {{0, 0}, {1.0 / 2.0, 0}, {-1, 2}};
        double[] bData = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
        double[] cData = {0, 1.0 / 2.0, 1};

        RealMatrix a = new Array2DRowRealMatrix(aData);
        double[] b = bData;
        double[] c = cData;

        rungeKuttaMethod(a, b, c, "odessa");

        double[][] aData2 = {{0}, {1.0 / 2.0}};
        double[] bData2 = {0, 1};
        double[] cData2 = {0, 1.0 / 2.0};

        RealMatrix a2 = new Array2DRowRealMatrix(aData2);
        double[] b2 = bData2;
        double[] c2 = cData2;

        rungeKuttaMethod(a2, b2, c2, "kyiv");

    }
}