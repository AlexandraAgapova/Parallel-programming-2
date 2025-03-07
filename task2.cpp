#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <omp.h>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

const int NUM_THREADS = 40;  // 1,2,4,7,8,16,20,40 

// f(x) = exp(-x*x)
double func(double x) {
    return exp(-x * x);
}

// Последовательное численное интегрирование
double integrate(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += func(a + h * (i + 0.5));
    }
    sum *= h;
    return sum;
}

// Параллельное интегрирование
double integrate_omp(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;

    omp_set_num_threads(NUM_THREADS);
    
    #pragma omp parallel
    {
        double local_sum = 0.0;
        int threadid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        
        for (int i = lb; i <= ub; i++) {
            local_sum += func(a + h * (i + 0.5));
        }
        // Атомарное добавление локальной суммы в глобальную
        #pragma omp atomic
        sum += local_sum;
    }
    
    sum *= h;
    return sum;
}

// Функция для измерения времени
double wtime() {
    return omp_get_wtime();
}

// Запуск последовательной версии
double run_serial() {
    double t = wtime();
    double res = integrate(func, a, b, nsteps);
    t = wtime() - t;
    printf("Result (serial): %.12f; error: %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

// Запуск параллельной версии
double run_parallel() {
    double t = wtime();
    double res = integrate_omp(func, a, b, nsteps);
    t = wtime() - t;
    printf("Result (parallel): %.12f; error: %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

int main() {
    printf("Интегрирование f(x) на интервале [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    
    double tserial = run_serial();
    double tparallel = run_parallel();
    
    printf("Время выполнения (serial): %.6f секунд\n", tserial);
    printf("Время выполнения (parallel): %.6f секунд\n", tparallel);
    printf("Ускорение: %.2f\n", tserial / tparallel);
    
    return 0;
}
