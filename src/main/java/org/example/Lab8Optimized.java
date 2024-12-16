package org.example;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.style.Styler;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

//Вариант 4
public class Lab8Optimized {
    private static final double c2 = 0.2;
    private static final double A = 2;
    private static final double B = -1;
    private static final double C = -1;
    private static final double x0 = 0;
    private static final double x1 = 5;
    private static final RealVector y0 = new ArrayRealVector(new double[]{1, 1, A, 1});
    private static final double atol = 1e-12;
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();

    // Используем ThreadPool для переиспользования потоков
    private static final ExecutorService executorService = Executors.newFixedThreadPool(NUM_THREADS);

    // Кэш для хранения промежуточных результатов
    private static final Map<String, RealVector> calculationCache = new ConcurrentHashMap<>();
    private static final Map<String, Double> optimalStepCache = new ConcurrentHashMap<>();


    public static RealVector diffEquation(double x, RealVector y) {
        double[] dy = new double[4];
        dy[0] = 2 * x * Math.pow(Math.max(y.getEntry(1), 1e-5), 1 / B) * y.getEntry(3);
        dy[1] = 2 * B * x * Math.exp(Math.max(Math.min((B / C) * (y.getEntry(2) - A), 20), -20)) * y.getEntry(3);
        dy[2] = 2 * C * x * y.getEntry(3);
        dy[3] = -2 * x * Math.log(Math.max(y.getEntry(0), 1e-5));
        return new ArrayRealVector(dy);
    }

    public static RealVector realY(double x) {
        String cacheKey = "real_" + x;
        return calculationCache.computeIfAbsent(cacheKey, k -> {
            double[] y = new double[4];
            y[0] = Math.exp(Math.sin(x * x));
            y[1] = Math.exp(B * Math.sin(x * x));
            y[2] = C * Math.sin(x * x) + A;
            y[3] = Math.cos(x * x);
            return new ArrayRealVector(y);
        });
    }

    public static Result rungeKutta(double xStart, RealVector yStart, double xEnd, double h) {
        double a21 = c2;
        double b2 = 1 / (2 * c2);
        double b1 = 1 - b2;

        String cacheKey = String.format("rk_%f_%f_%f", xStart, xEnd, h);
        RealVector cachedResult = calculationCache.get(cacheKey);
        if (cachedResult != null) {
            return new Result(cachedResult, 2); // Возвращаем кэшированный результат
        }

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

        calculationCache.put(cacheKey, yCurr);
        return new Result(yCurr, calcCount);
    }

    public static Result simpson(double xStart, RealVector yStart, double xEnd, double h) {
        String cacheKey = String.format("simpson_%f_%f_%f", xStart, xEnd, h);
        RealVector cachedResult = calculationCache.get(cacheKey);
        if (cachedResult != null) {
            return new Result(cachedResult, 3);
        }

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

        calculationCache.put(cacheKey, yCurr);
        return new Result(yCurr, calcCount);
    }


    public static class Result {
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

    private static void runParallelCalculations(double[] rtols) {
        try {
            List<CompletableFuture<Map<String, AdaptiveResult>>> futures = Arrays.stream(rtols)
                    .mapToObj(rtol -> CompletableFuture.supplyAsync(() -> {
                        Map<String, AdaptiveResult> results = new HashMap<>();
                        results.put("Рунге-Кутты", adaptiveStepAlgorithm(Lab8Optimized::rungeKutta, "Рунге-Кутты", rtol));
                        results.put("Симпсон", adaptiveStepAlgorithm(Lab8Optimized::simpson, "Симпсон", rtol));
                        return results;
                    }, executorService))
                    .collect(Collectors.toList());

            // Ждем завершения всех вычислений
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

            // Собираем результаты
            Map<Double, Map<String, Integer>> results = new HashMap<>();
            for (int i = 0; i < rtols.length; i++) {
                Map<String, AdaptiveResult> methodResults = futures.get(i).get();
                Map<String, Integer> calcCounts = new HashMap<>();
                calcCounts.put("Рунге-Кутты", methodResults.get("Рунге-Кутты").calcCount);
                calcCounts.put("Симпсон", methodResults.get("Симпсон").calcCount);
                results.put(rtols[i], calcCounts);
            }

            // Визуализация результатов
            visualizeResults(results);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void visualizeResults(Map<Double, Map<String, Integer>> results) {
        XYChart chart = new XYChart(800, 600);
        chart.setTitle("Зависимость числа обращений к правой части от rtol");
        chart.setXAxisTitle("rtol");
        chart.setYAxisTitle("Число обращений");
        chart.getStyler().setXAxisLogarithmic(true);
        chart.getStyler().setYAxisLogarithmic(true);

        List<Double> rtolValues = new ArrayList<>(results.keySet());
        Collections.sort(rtolValues);

        List<Integer> rkCounts = rtolValues.stream()
                .map(rtol -> results.get(rtol).get("Рунге-Кутты"))
                .collect(Collectors.toList());

        List<Integer> simpsonCounts = rtolValues.stream()
                .map(rtol -> results.get(rtol).get("Симпсон"))
                .collect(Collectors.toList());

        chart.addSeries("Рунге-Кутты", rtolValues, rkCounts);
        chart.addSeries("Симпсон", rtolValues, simpsonCounts);

        try {
            BitmapEncoder.saveBitmap(chart, "./parallel_calc_counts_chart", BitmapFormat.PNG);
            System.out.println("График parallel_calc_counts_chart.png сохранён.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static AdaptiveResult adaptiveStepAlgorithm(AdaptiveMethod method, String methodName, double rtol) {
        double h = 0.1;
        double x = x0;
        RealVector y = y0.copy();
        int calcCount = 0;
        List<Double> xValues = new ArrayList<>();
        List<Double> hValues = new ArrayList<>();
        List<Double> norms = new ArrayList<>();
        List<String> stepTypes = new ArrayList<>();
        List<Double> fullError = new ArrayList<>();
        List<RealVector> yValues = new ArrayList<>(); // Добавлено для хранения значений y
        int degree = methodName.equals("Рунге-Кутты") ? 2 : 3;

        while (x < x1) {
            h = Math.min(h, x1 - x);
            Result result = method.apply(x, y, x + h, h);
            RealVector y1 = result.y;
            calcCount += result.calcCount;

            Result resultHalf = method.apply(x, y, x + h, h / 2);
            RealVector y2 = resultHalf.y;
            calcCount += resultHalf.calcCount;

            double norm = y1.subtract(y2).mapDivide(1 - Math.pow(2, -degree)).getNorm();

            xValues.add(x);
            hValues.add(h);
            norms.add(realY(x).subtract(y).getNorm());
            fullError.add(fullErrorrate(methodName, x,y,h));
            yValues.add(y); // Сохраняем значение y

            if (norm > Math.pow(2, degree) * rtol) {
                h /= 2;
                stepTypes.add("Уменьшение");
            } else if (rtol < norm && norm <= Math.pow(2, degree) * rtol) {
                x = x + h;
                y = y2;
                h /= 2;
                stepTypes.add("Принятие и уменьшение");
            } else if (rtol / Math.pow(2, degree + 1) < norm && norm <= rtol) {
                x = x + h;
                y = y1;
                stepTypes.add("Принятие");
            } else {
                x = x + h;
                y = y1;
                h = Math.min(2 * h, 0.5);
                stepTypes.add("Принятие и увеличение");
            }
        }

        if (rtol == 1e-6) {
            // График решения
            saveAdaptiveSolutionChart(methodName, rtol, xValues, yValues);


            // График длины шага
            XYChart chartStep = new XYChart(800, 600);
            chartStep.setTitle("Длина шага от x (" + methodName + "), rtol = " + rtol);
            chartStep.setXAxisTitle("x");
            chartStep.setYAxisTitle("Длина шага");

            // Добавляем данные для графика длины шага
            List<Double> acceptedX = new ArrayList<>();
            List<Double> acceptedH = new ArrayList<>();
            List<Double> rejectedX = new ArrayList<>();
            List<Double> rejectedH = new ArrayList<>();

            for (int i = 0; i < xValues.size(); i++) {
                if (stepTypes.get(i).equals("Принятие") || stepTypes.get(i).equals("Увеличение") || stepTypes.get(i).equals("Принятие и уменьшение")) {
                    acceptedX.add(xValues.get(i));
                    acceptedH.add(hValues.get(i));
                } else {
                    rejectedX.add(xValues.get(i));
                    rejectedH.add(hValues.get(i));
                }
            }

            chartStep.addSeries("Принятые шаги", acceptedX, acceptedH);
            chartStep.addSeries(".", rejectedX, rejectedH);
            saveChart(chartStep, "./step_chart_" + methodName + "_" + rtol);


            //График зависимости нормы точной полной погрешности от независимой переменной

            XYChart chartFullError = QuickChart.getChart("Зависимость полной погрешности (" + methodName + "), rtol = " + rtol, "x", "Норма",
                    "Норма", xValues, fullError);
            saveChart(chartFullError, "./full_error_chart_" + methodName + "_" + rtol);
        }


        return new AdaptiveResult(calcCount);
    }

    private static double fullErrorrate(String methodName, double x, RealVector y, double h) {
        double tol = 1e-5;
        int degree = methodName.equals("Рунге-Кутты") ? 2 : 3;
        double hOpt = 1.0;
        Result rkResult = null;
        Result rkResultHalf = null;
        double hLocal = h;

        if (methodName.equals("Рунге-Кутты")) {
            while (true) {
                rkResult = rungeKutta(x, y, x + hLocal, hLocal);
                rkResultHalf = rungeKutta(x, y, x + hLocal, hLocal / 2);
                RealVector error = rkResult.y.subtract(rkResultHalf.y).mapDivide(1 - Math.pow(2, -degree));
                double errorNorm = error.getNorm();

                if (errorNorm < tol) {
                    break;
                } else {
                    hLocal /= 2;
                }
            }
        } else {
            while (true) {
                rkResult = simpson(x, y, x + hLocal, hLocal);
                rkResultHalf = simpson(x, y, x + hLocal, hLocal / 2);
                RealVector error = rkResult.y.subtract(rkResultHalf.y).mapDivide(1 - Math.pow(2, -degree));
                double errorNorm = error.getNorm();

                if (errorNorm < tol) {
                    break;
                } else {
                    hLocal /= 2;
                }
            }
        }

        RealVector error = rkResult.y.subtract(rkResultHalf.y).mapDivide(1 - Math.pow(2, -degree));
        return error.getNorm();
    }


    private static void saveChart(XYChart chart, String fileName) {
        try {
            BitmapEncoder.saveBitmap(chart, fileName, BitmapFormat.PNG);
            System.out.println("График " + fileName + ".png сохранён.");
        } catch (Exception e) {
            System.err.println("Ошибка при сохранении графика: " + e.getMessage());
        }
    }

    private static void saveAdaptiveSolutionChart(String methodName, double rtol, List<Double> xValues, List<RealVector> yValues) {
        XYChart chart = new XYChart(800, 600);
        chart.setTitle("Решение с адаптивным шагом (" + methodName + "), rtol = " + rtol);
        chart.setXAxisTitle("x");
        chart.setYAxisTitle("y");

        List<Double> y1Values = new ArrayList<>();
        List<Double> y2Values = new ArrayList<>();
        List<Double> y3Values = new ArrayList<>();
        List<Double> y4Values = new ArrayList<>();

        for (RealVector y : yValues) {
            y1Values.add(y.getEntry(0));
            y2Values.add(y.getEntry(1));
            y3Values.add(y.getEntry(2));
            y4Values.add(y.getEntry(3));
        }

        chart.addSeries("y1(x)", xValues, y1Values);
        chart.addSeries("y2(x)", xValues, y2Values);
        chart.addSeries("y3(x)", xValues, y3Values);
        chart.addSeries("y4(x)", xValues, y4Values);
        saveChart(chart, "./adaptive_solution_chart_" + methodName + "_" + rtol);
    }

    private static void visualizeRealSolution() {
        List<Double> xValues = new ArrayList<>();
        List<Double> y1Values = new ArrayList<>();
        List<Double> y2Values = new ArrayList<>();
        List<Double> y3Values = new ArrayList<>();
        List<Double> y4Values = new ArrayList<>();

        for (double x = x0; x <= x1; x += 0.01) {
            RealVector y = realY(x);
            xValues.add(x);
            y1Values.add(y.getEntry(0));
            y2Values.add(y.getEntry(1));
            y3Values.add(y.getEntry(2));
            y4Values.add(y.getEntry(3));
        }

        XYChart chart = new XYChart(800, 600);
        chart.setTitle("Исходное решение");
        chart.setXAxisTitle("x");
        chart.setYAxisTitle("y");
        chart.addSeries("y1(x)", xValues, y1Values);
        chart.addSeries("y2(x)", xValues, y2Values);
        chart.addSeries("y3(x)", xValues, y3Values);
        chart.addSeries("y4(x)", xValues, y4Values);
        saveChart(chart, "./real_solution_chart");
    }

    @FunctionalInterface
    private interface AdaptiveMethod {
        Result apply(double xStart, RealVector yStart, double xEnd, double h);
    }

    private static class AdaptiveResult {
        public final int calcCount;

        public AdaptiveResult(int calcCount) {
            this.calcCount = calcCount;
        }
    }

    public static void main(String[] args) {
        try {
            // Параллельные вычисления для разных значений rtol
            double[] rtols = {1e-4, 1e-5, 1e-6, 1e-7, 1e-8};
            runParallelCalculations(rtols);

            // Закрываем пул потоков после завершения всех вычислений
            executorService.shutdown();
            if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }

            // Вычисление и визуализация для постоянного шага
            visualizeConstantStep();

            // Вычисление и визуализация для оптимального шага
            visualizeOptimalStep();

            // Визуализация исходного решения
            visualizeRealSolution();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void visualizeConstantStep() {
        List<Double> hValues = IntStream.rangeClosed(0, 6)
                .mapToDouble(k -> 1.0 / Math.pow(2, k))
                .boxed()
                .collect(Collectors.toList());

        List<Double> rkNorms = new ArrayList<>();
        List<Double> simpsonNorms = new ArrayList<>();

        for (double h : hValues) {
            Result rkResult = rungeKutta(x0, y0, x1, h);
            Result simpsonResult = simpson(x0, y0, x1, h);
            RealVector realY = realY(x1);

            rkNorms.add(rkResult.y.subtract(realY).getNorm());
            simpsonNorms.add(simpsonResult.y.subtract(realY).getNorm());
        }

        XYChart chart = new XYChart(800, 600);
        chart.setTitle("Норма погрешности от h (постоянный шаг)");
        chart.setXAxisTitle("h");
        chart.setYAxisTitle("Норма");
        chart.getStyler().setXAxisLogarithmic(true);
        chart.getStyler().setYAxisLogarithmic(true);

        chart.addSeries("Рунге-Кутты", hValues, rkNorms);
        chart.addSeries("Симпсон", hValues, simpsonNorms);

        saveChart(chart, "./constant_step_chart");

        // Построение графика зависимости нормы точной полной погрешности от длины шага
        XYChart chartNormVsH = new XYChart(800, 600);
        chartNormVsH.setTitle("Зависимость нормы погрешности от h");
        chartNormVsH.setXAxisTitle("h");
        chartNormVsH.setYAxisTitle("Норма погрешности");
        chartNormVsH.getStyler().setXAxisLogarithmic(true);
        chartNormVsH.getStyler().setYAxisLogarithmic(true);

        chartNormVsH.addSeries("Рунге-Кутты", hValues, rkNorms);
        chartNormVsH.addSeries("Симпсон", hValues, simpsonNorms);

        // Добавляем прямую с наклоном 2 для метода Рунге-Кутты
        List<Double> hValuesForLine = new ArrayList<>(hValues);
        List<Double> rkLineValues = hValuesForLine.stream().map(h -> h * h).collect(Collectors.toList());
        chartNormVsH.addSeries("Прямая с наклоном 2", hValuesForLine, rkLineValues);

        // Добавляем прямую с наклоном 3 для метода Симпсона
        List<Double> simpsonLineValues = hValuesForLine.stream().map(h -> h * h * h).collect(Collectors.toList());
        chartNormVsH.addSeries("Прямая с наклоном 3", hValuesForLine, simpsonLineValues);

        saveChart(chartNormVsH, "./constant_step_norm_vs_h_chart");
    }

    private static void visualizeOptimalStep() {
        double tol = 1e-5;
        int degreeRk = 2;
        int degreeSimpson = 3;
        String cacheKeyRk = "hOptRk";
        String cacheKeySimpson = "hOptSimpson";
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

        double hOptRk = optimalStepCache.computeIfAbsent(cacheKeyRk, k -> {
            double h = 1.0;
            Result rkResult = null;
            Result rkResultHalf = null;
            while (true) {
                rkResult = rungeKutta(x0, y0, x1, h);
                rkResultHalf = rungeKutta(x0, y0, x1, h / 2);

                RealVector error = rkResult.y.subtract(rkResultHalf.y).mapDivide(1 - Math.pow(2, -degreeRk));
                double errorNorm = error.getNorm();

                if (errorNorm < tol) {
                    System.out.println("Оптимальный шаг (Рунге-Кутты): " + h);
                    return h;
                } else {
                    h /= 2;
                }
            }
        });

        double hOptSimpson = optimalStepCache.computeIfAbsent(cacheKeySimpson, k -> {
            double h = 1.0;
            Result simpsonResult = null;
            Result simpsonResultHalf = null;
            while (true) {
                simpsonResult = simpson(x0, y0, x1, h);
                simpsonResultHalf = simpson(x0, y0, x1, h / 2);

                RealVector error = simpsonResult.y.subtract(simpsonResultHalf.y).mapDivide(1 - Math.pow(2, -degreeSimpson));
                double errorNorm = error.getNorm();

                if (errorNorm < tol) {
                    System.out.println("Оптимальный шаг (Симпсона): " + h);
                    return h;
                } else {
                    h /= 2;
                }
            }
        });

        List<Double> nodesRk = new ArrayList<>();
        List<Double> normsRk = new ArrayList<>();
        List<Double> nodesSimpson = new ArrayList<>();
        List<Double> normsSimpson = new ArrayList<>();

        List<CompletableFuture<Void>> futures = new ArrayList<>();
        for (double i = x0; i <= x1; i += hOptRk) {
            double x = i;
            futures.add(CompletableFuture.runAsync(() -> {
                RealVector real = realY(x);
                Result rk = rungeKutta(x0, y0, x, hOptRk);
                synchronized (nodesRk) {
                    nodesRk.add(x);
                    normsRk.add(rk.y.subtract(real).getNorm());
                }
            }, executor));
        }
        for (double i = x0; i <= x1; i += hOptSimpson) {
            double x = i;
            futures.add(CompletableFuture.runAsync(() -> {
                RealVector real = realY(x);
                Result simpson = simpson(x0, y0, x, hOptSimpson);
                synchronized (nodesSimpson) {
                    nodesSimpson.add(x);
                    normsSimpson.add(simpson.y.subtract(real).getNorm());
                }
            }, executor));
        }
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }


        // Построение графика зависимости нормы точной полной погрешности от независимой переменной при решении с hopt
        XYChart chartNormsVsX = new XYChart(800, 600);
        chartNormsVsX.setTitle("Зависимость нормы погрешности от x при hopt");
        chartNormsVsX.setXAxisTitle("x");
        chartNormsVsX.setYAxisTitle("Норма погрешности");

        chartNormsVsX.addSeries("Рунге-Кутты", nodesRk, normsRk);
        chartNormsVsX.addSeries("Симпсон", nodesSimpson, normsSimpson);
        saveChart(chartNormsVsX, "./optimal_step_norm_vs_x_chart");
    }
}