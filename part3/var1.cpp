#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

// Функция инициализации матрицы A и вектора b.
// Матрица A заполняется так, что A(i, j) = 2.0, если i == j, иначе 1.0.
// Вектор b: для i = 0 и i = 1, b[i] = N + 1, для остальных b[i] = N.
void initializeData(int N, std::vector<double> &A, std::vector<double> &b) {
    A.resize(N * N);
    b.resize(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
    }
    for (int i = 0; i < N; i++) {
        b[i] = (i == 0 || i == 1) ? (N + 1) : N;
    }
}

int main(int argc, char *argv[]) {
    int N = 4000;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    // Выделение памяти для матрицы A, векторов b, x и r.
    // x - начальное приближение (инициализируем нулями)
    // r - для хранения результата умножения A*x
    std::vector<double> A, b, x(N, 0.0), r(N, 0.0);
    initializeData(N, A, b);

    // Параметры итерационного метода
    double tau = 1e-4;  // шаг
    double eps = 1e-7;  // требуемая относительная точность
    int maxIter = 10000;// максимальное число итераций

    // Вычисление нормы вектора b
    double normB = 0.0;
    for (int i = 0; i < N; i++) {
        normB += b[i] * b[i];
    }
    normB = std::sqrt(normB);

    // Замер времени начала вычислений
    auto start = std::chrono::high_resolution_clock::now();

    int iter;
    for (iter = 0; iter < maxIter; iter++) {
// 1) Вычисление произведения матрицы A на вектор x: r = A*x
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A[i * N + j] * x[j];
            }
            r[i] = sum;
        }

        // 2) Вычисление суммы квадратов невязки: res_sq = ||b - r||^2
        double res_sq = 0.0;
#pragma omp parallel for reduction(+ : res_sq)
        for (int i = 0; i < N; i++) {
            double diff = b[i] - r[i];
            res_sq += diff * diff;
        }
        double normRes = std::sqrt(res_sq);
        double relRes = normRes / normB;

        // Вывод информации каждые 100 итераций (вывод может выполняться одним потоком)
        if (iter % 100 == 0) {
            std::cout << "Iter " << iter << ", relative residual = " << relRes << std::endl;
        }
        if (relRes < eps) {
            break;
        }

// 3) Обновление вектора x: x = x + tau*(b - r)
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x[i] += tau * (b[i] - r[i]);
        }
    }

    int maxThreads = omp_get_max_threads();
    std::cout << "Maximum available threads: " << maxThreads << std::endl;

    // Замер времени окончания вычислений
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\nConverged after " << iter << " iterations." << std::endl;
    std::cout << "Total elapsed time: " << elapsed.count() << " seconds." << std::endl;

    char *ompThreads = std::getenv("OMP_NUM_THREADS");
    if (ompThreads != nullptr) {
        std::cout << "OMP_NUM_THREADS: " << ompThreads << std::endl;
    } else {
        std::cout << "OMP_NUM_THREADS is not set." << std::endl;
    }
    
    std::cout << "\nComputed solution x (first 10 entries):" << std::endl;
    for (int i = 0; i < std::min(N, 10); i++) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }
    std::cout << "\nExpected solution x (vector of ones):" << std::endl;
    for (int i = 0; i < std::min(N, 10); i++) {
        std::cout << "1 ";
    }
    std::cout << std::endl;

    return 0;
}
