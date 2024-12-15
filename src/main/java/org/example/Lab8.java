package org.example;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.apache.commons.math3.linear.RealVector;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

public class Lab8 {

    private static final double c2 = 0.6;
    private static final double A = 1;
    private static final double B = 3;
    private static final double C = 3;
    private static final double x0 = 0;
    private static final double x1 = 5;
    private static final RealVector y0 = new ArrayRealVector(new double[]{1, 1, A, 1});
    private static final double atol = 1e-12;


    // Правая часть системы
    public static RealVector diffEquation(double x, RealVector y) {
        double[] dy = new double[4];
        dy[0] = 2 * x * Math.pow(Math.max(y.getEntry(1), 1e-5), 1 / B) * y.getEntry(3);
        dy[1] = 2 * B * x * Math.exp(Math.max(Math.min((B / C) * (y.getEntry(2) - A), 20), -20)) * y.getEntry(3);
        dy[2] = 2 * C * x * y.getEntry(3);
        dy[3] = -2 * x * Math.log(Math.max(y.getEntry(0), 1e-5));
        return new ArrayRealVector(dy);
    }

    // Реальное решение задачи Коши
    public static RealVector realY(double x) {
        double[] y = new double[4];
        y[0] = Math.exp(Math.sin(x * x));
        y[1] = Math.exp(B * Math.sin(x * x));
        y[2] = C * Math.sin(x * x) + A;
        y[3] = Math.cos(x * x);
        return new ArrayRealVector(y);
    }

    // Метод Рунге-Кутты 2 порядка
    public static Result rungeKutta(double xStart, RealVector yStart, double xEnd, double h) {
        double a21 = c2;
        double b2 = 1 / (2 * c2);
        double b1 = 1 - b2;

        double xCurr = xStart;
        RealVector yCurr = yStart.copy();
        int calcCount = 0;

        while (xCurr < xEnd) {
            RealVector k1 = diffEquation(xCurr, yCurr).mapMultiply(h);
            RealVector k2 = diffEquation(xCurr + h * c2, yCurr.add(k1.mapMultiply(a21))).mapMultiply(h);

            calcCount += 2;

            xCurr = Math.min(xCurr + h, xEnd);
            yCurr = yCurr.add(k1.mapMultiply(b1)).add(k2.mapMultiply(b2));
        }
        return new Result(yCurr, calcCount);
    }


    // Метод Симпсона (метод-оппонент)
    public static Result simpson(double xStart, RealVector yStart, double xEnd, double h) {
        double xCurr = xStart;
        RealVector yCurr = yStart.copy();
        int calcCount = 0;

        while (xCurr < xEnd) {
            RealVector k1 = diffEquation(xCurr, yCurr).mapMultiply(h);
            RealVector k2 = diffEquation(xCurr + h / 2, yCurr.add(k1.mapMultiply(0.5))).mapMultiply(h);
            RealVector k3 = diffEquation(xCurr + h, yCurr.subtract(k1).add(k2.mapMultiply(2))).mapMultiply(h);

            calcCount += 3;

            xCurr = Math.min(xCurr + h, xEnd);
            yCurr = yCurr.add(k1.mapMultiply(1.0 / 6)).add(k2.mapMultiply(4.0 / 6)).add(k3.mapMultiply(1.0 / 6));
        }
        return new Result(yCurr, calcCount);
    }

    private static class Result {
        public RealVector y;
        public int calcCount;

        public Result(RealVector y, int calcCount) {
            this.y = y;
            this.calcCount = calcCount;
        }

        @Override
        public String toString() {
            return "(" + y + ", " + calcCount + ")";
        }
    }

    public static void main(String[] args) {
        // Пример использования
        Result rkResult = rungeKutta(x0, y0, x1, 0.005);
        Result simpsonResult = simpson(x0, y0, x1, 0.005);
        RealVector realY = realY(x1);


        System.out.println("Приближение по ЯМРК-2 в x1: " + rkResult);
        System.out.println("Приближение по методу Симпсона в x1: " + simpsonResult);
        System.out.println("Реальное значение решения в x1: " + realY);

        List<Double> hValues = new ArrayList<>();
        List<Double> rkNorms = new ArrayList<>();
        List<Double> simpsonNorms = new ArrayList<>();

        int p = 15;
        for (int i = 0; i < p; i++) {
            double h = 1.0 / Math.pow(2, i);
            hValues.add(h);

            rkResult = rungeKutta(x0, y0, x1, h);
            simpsonResult = simpson(x0, y0, x1, h);
            realY = realY(x1);

            rkNorms.add(rkResult.y.subtract(realY).getNorm());
            simpsonNorms.add(simpsonResult.y.subtract(realY).getNorm());
        }

//
//        // Построение графика с помощью XChart
//        XYChart chart = QuickChart.getChart("Норма погрешности", "h", "Норма",
//                "Рунге-Кутты", hValues, rkNorms);
//        new SwingWrapper<>(chart).displayChart(); //ААААА


        // Пункт 4: Оценка полной погрешности по правилу Рунге и h_opt
        double tol = 1e-5;
        double hOptRk = 1.0;
        int degreeRk = 2;
        int calcResRk = 0;

        while (true) {
            rkResult = rungeKutta(x0, y0, x1, hOptRk);
            Result rkResultHalf = rungeKutta(x0, y0, x1, hOptRk / 2);

            calcResRk += rkResult.calcCount + rkResultHalf.calcCount;

            RealVector error = rkResult.y.subtract(rkResultHalf.y).mapDivide(1 - Math.pow(2, -degreeRk));
            double errorNorm = error.getNorm();

            if (errorNorm < tol) {
                System.out.println("Оптимальный шаг (Рунге-Кутты): " + hOptRk);
                System.out.println("Решение (Рунге-Кутты): " + rkResult.y);
                System.out.println("Норма погрешности (Рунге-Кутты): " + errorNorm);
                System.out.println("Вычисления (Рунге-Кутты): " + calcResRk);
                break;
            } else {
                hOptRk /= 2;
            }
        }

        // Аналогично для метода Симпсона (с tol1 и degree_opp)
        double tol1 = 1e-2; // Изменено для более быстрой сходимости
        double hOptSimpson = 1.0;
        int degreeSimpson = 3;
        int calcResSimpson = 0;

        while (true) {
            simpsonResult = simpson(x0, y0, x1, hOptSimpson);
            Result simpsonResultHalf = simpson(x0, y0, x1, hOptSimpson / 2);

            calcResSimpson += simpsonResult.calcCount + simpsonResultHalf.calcCount;

            RealVector error = simpsonResult.y.subtract(simpsonResultHalf.y).mapDivide(1 - Math.pow(2, -degreeSimpson));
            double errorNorm = error.getNorm();

            if (errorNorm < tol1) {
                System.out.println("Оптимальный шаг (Симпсон): " + hOptSimpson);
                System.out.println("Решение (Симпсон): " + simpsonResult.y);
                System.out.println("Норма погрешности (Симпсон): " + errorNorm);
                System.out.println("Вычисления (Симпсон): " + calcResSimpson);
                break;
            } else {
                hOptSimpson /= 2;
            }
        }


        List<Double> nodesRk = new ArrayList<>();
        List<Double> normsRk = new ArrayList<>();

        for (double i = x0; i <= x1; i += hOptRk) {
            nodesRk.add(i);
            RealVector real = realY(i);
            Result rk = rungeKutta(x0, y0, i, hOptRk);
            normsRk.add(rk.y.subtract(real).getNorm());
        }

        List<Double> nodesSimpson = new ArrayList<>();
        List<Double> normsSimpson = new ArrayList<>();

        for (double i = x0; i <= x1; i += hOptSimpson) {
            nodesSimpson.add(i);
            RealVector real = realY(i);
            Result simpson = simpson(x0, y0, i, hOptSimpson);
            normsSimpson.add(simpson.y.subtract(real).getNorm());
        }

        // Построение графика с использованием XChart
        XYChart chartComparison = new XYChart(800, 600);
        chartComparison.setTitle("Сравнение норм для оптимального шага");
        chartComparison.setXAxisTitle("x");
        chartComparison.setYAxisTitle("Норма ошибки");

        // Добавление данных для метода Рунге-Кутты
        chartComparison.addSeries("Рунге-Кутты", nodesRk, normsRk);

        // Добавление данных для метода Симпсона
        chartComparison.addSeries("Симпсон", nodesSimpson, normsSimpson);

        // Отображение графика
        new SwingWrapper<>(chartComparison).displayChart();

        // Сохранение графика в файл
        try {
            BitmapEncoder.saveBitmap(chartComparison, "./comparison_chart", BitmapFormat.PNG);
            System.out.println("График comparison_chart.png сохранён.");
        } catch (Exception e) {
            System.err.println("Ошибка при сохранении графика: " + e.getMessage());
        }

        // Подготовка данных для графика нормы от x
        List<Double> nodesRkForNorms = new ArrayList<>();
        List<Double> normsRkForNorms = new ArrayList<>();
        for (double i = x0; i <= x1; i += hOptRk) {
            nodesRkForNorms.add(i);
            RealVector real = realY(i);
            Result rk = rungeKutta(x0, y0, i, hOptRk);
            normsRkForNorms.add(rk.y.subtract(real).getNorm());
        }

        // Построение и отображение графика нормы от x
        XYChart chartNorms = QuickChart.getChart("Норма от x", "x", "Норма",
                "Рунге-Кутты", nodesRkForNorms, normsRkForNorms);
        new SwingWrapper<>(chartNorms).displayChart();

        // Сохранение графика нормы от
        try {
            BitmapEncoder.saveBitmap(chartNorms, "norms_chart", BitmapFormat.PNG);
            System.out.println("График norms_chart.png сохранён.");
        } catch (Exception e) {
            System.err.println("Ошибка при сохранении графика: " + e.getMessage());
        }

        // Пункт 4: Алгоритм с автоматическим выбором шага
        double[] rtols = {1e-4, 1e-5, 1e-6, 1e-7, 1e-8};
        Map<Double, Integer> rkCalcCounts = new HashMap<>();
        Map<Double, Integer> simpsonCalcCounts = new HashMap<>();

        for (double rtol : rtols) {
            AdaptiveResult rkAdaptiveResult = adaptiveStepAlgorithm(Lab8::rungeKutta, "Рунге-Кутты", rtol);
            rkCalcCounts.put(rtol, rkAdaptiveResult.calcCount);
            AdaptiveResult simpsonAdaptiveResult = adaptiveStepAlgorithm(Lab8::simpson, "Симпсон", rtol);
            simpsonCalcCounts.put(rtol, simpsonAdaptiveResult.calcCount);
        }

        // Построение графика зависимости числа обращений к правой части от rtol
        XYChart chartCalcCounts = new XYChart(800, 600);
        chartCalcCounts.setTitle("Зависимость числа обращений к правой части от rtol");
        chartCalcCounts.setXAxisTitle("rtol");
        chartCalcCounts.setYAxisTitle("Число обращений");

        // Устанавливаем логарифмический масштаб для осей
        chartCalcCounts.getStyler().setXAxisLogarithmic(true);
        chartCalcCounts.getStyler().setYAxisLogarithmic(true);

        List<Double> rtolValues = new ArrayList<>(rkCalcCounts.keySet());
        List<Integer> rkCounts = new ArrayList<>(rkCalcCounts.values());
        List<Integer> simpsonCounts = new ArrayList<>(simpsonCalcCounts.values());

        chartCalcCounts.addSeries("Рунге-Кутты", rtolValues, rkCounts);
        chartCalcCounts.addSeries("Симпсон", rtolValues, simpsonCounts);

        new SwingWrapper<>(chartCalcCounts).displayChart();

        try {
            BitmapEncoder.saveBitmap(chartCalcCounts, "./calc_counts_chart", BitmapFormat.PNG);
            System.out.println("График calc_counts_chart.png сохранён.");
        } catch (Exception e) {
            System.err.println("Ошибка при сохранении графика: " + e.getMessage());
        }
    }

    private static AdaptiveResult adaptiveStepAlgorithm(AdaptiveMethod method, String methodName, double rtol) {
        double h = 0.1;
        double x = x0;
        RealVector y = y0.copy();
        List<Double> xValues = new ArrayList<>();
        List<Double> hValues = new ArrayList<>();
        List<Double> norms = new ArrayList<>();
        List<String> stepTypes = new ArrayList<>();
        int calcCount = 0;

        while (x < x1) {
            Result result = method.apply(x, y, x + h, h);
            RealVector y1 = result.y;
            calcCount += result.calcCount;

            Result resultHalf = method.apply(x, y, x + h, h / 2);
            RealVector y2 = resultHalf.y;
            calcCount += resultHalf.calcCount;

            double norm;
            if (methodName.equals("Рунге-Кутты")) {
                norm = y1.subtract(y2).mapDivide(1 - Math.pow(2, -2)).getNorm();
            } else {
                norm = y1.subtract(y2).mapDivide(1 - Math.pow(2, -3)).getNorm();
            }

            xValues.add(x);
            hValues.add(h);
            norms.add(realY(x).subtract(y).getNorm());

            if (norm > rtol * Math.pow(2, 2)) {
                h /= 2;
                stepTypes.add("Уменьшение");
            } else if (norm < rtol / Math.pow(2, 3)) {
                h = Math.min(2 * h, 0.5);
                y = y1;
                x = x + h;
                stepTypes.add("Увеличение");
            } else {
                y = y1;
                x = x + h;
                stepTypes.add("Принятие");
            }
        }

        System.out.println("Алгоритм с автоматическим выбором шага (" + methodName + "), rtol = " + rtol + ":");
        System.out.println("Количество вычислений: " + calcCount);

        // График решения
        XYChart chartSolution = QuickChart.getChart("Решение (" + methodName + "), rtol = " + rtol, "x", "y",
                "Решение", xValues, norms);
        new SwingWrapper<>(chartSolution).displayChart();

        try {
            BitmapEncoder.saveBitmap(chartSolution, "./solution_chart_" + methodName + "_" + rtol, BitmapFormat.PNG);
            System.out.println("График solution_chart_" + methodName + "_" + rtol + ".png сохранён.");
        } catch (Exception e) {
            System.err.println("Ошибка при сохранении графика: " + e.getMessage());
        }


        // График длины шага
        XYChart chartStep = new XYChart(800, 600);
        chartStep.setTitle("Длина шага от x (" + methodName + "), rtol = " + rtol);
        chartStep.setXAxisTitle("x");
        chartStep.setYAxisTitle("Длина шага");

        // Добавляем данные для графика длины шага
        chartStep.addSeries("Длина шага", xValues, hValues);

        // Отображаем график
        new SwingWrapper<>(chartStep).displayChart();

        try {
            BitmapEncoder.saveBitmap(chartStep, "./step_chart_" + methodName + "_" + rtol, BitmapFormat.PNG);
            System.out.println("График step_chart_" + methodName + "_" + rtol + ".png сохранён.");
        } catch (Exception e) {
            System.err.println("Ошибка при сохранении графика: " + e.getMessage());
        }

        // График нормы погрешности
        XYChart chartNorm = QuickChart.getChart("Норма погрешности (" + methodName + "), rtol = " + rtol, "x", "Норма",
                "Норма", xValues, norms);
        new SwingWrapper<>(chartNorm).displayChart();

        try {
            BitmapEncoder.saveBitmap(chartNorm, "./norm_chart_" + methodName + "_" + rtol, BitmapFormat.PNG);
            System.out.println("График norm_chart_" + methodName + "_" + rtol + ".png сохранён.");
        } catch (Exception e) {
            System.err.println("Ошибка при сохранении графика: " + e.getMessage());
        }

        return new AdaptiveResult(calcCount);
    }

    private interface AdaptiveMethod {
        Result apply(double xStart, RealVector yStart, double xEnd, double h);
    }

    private static class AdaptiveResult {
        public int calcCount;

        public AdaptiveResult(int calcCount) {
            this.calcCount = calcCount;
        }
    }
}