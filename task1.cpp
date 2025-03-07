#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iomanip>

const int MATRIX_SIZE = 40000; // 20000 или 40000
const int NUM_THREADS = 40;    // 1, 2, 4, 7, 8, 16, 20, 40
const int NUM_RUNS = 10;   

std::vector<std::vector<double>> matrix(MATRIX_SIZE, std::vector<double>(MATRIX_SIZE));
std::vector<double> vec(MATRIX_SIZE);
std::vector<double> result(MATRIX_SIZE);

void initializeMatrix(int start, int end) {
    for (int i = start; i < end; i++)
        for (int j = 0; j < MATRIX_SIZE; j++)
            matrix[i][j] = static_cast<double>(i + j);
}

void initializeVector(int start, int end) {
    for (int i = start; i < end; i++)
        vec[i] = static_cast<double>(i);
}

void multiply(int start, int end) {
    for (int i = start; i < end; i++) {
        double sum = 0.0;
        for (int j = 0; j < MATRIX_SIZE; j++)
            sum += matrix[i][j] * vec[j];
        result[i] = sum;
    }
}

int main() {
    std::vector<double> runTimes;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        
        std::vector<std::thread> threads;
        int chunk = MATRIX_SIZE / NUM_THREADS;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Параллельная инициализация матрицы
        for (int t = 0; t < NUM_THREADS; t++)
            threads.emplace_back(initializeMatrix, t * chunk, (t + 1) * chunk);
        for (auto &t : threads)
            t.join();
        threads.clear();
        
        // Параллельная инициализация вектора
        for (int t = 0; t < NUM_THREADS; t++)
            threads.emplace_back(initializeVector, t * chunk, (t + 1) * chunk);
        for (auto &t : threads)
            t.join();
        threads.clear();
        
        // Параллельное умножение матрицы на вектор
        for (int t = 0; t < NUM_THREADS; t++)
            threads.emplace_back(multiply, t * chunk, (t + 1) * chunk);
        for (auto &t : threads)
            t.join();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
        runTimes.push_back(elapsed_time);
        
        std::cout << "Run " << std::setw(2) << run+1 << ": " 
                  << std::fixed << std::setprecision(6) 
                  << elapsed_time << " seconds" << std::endl;
    }
    
    // Копируем результаты для сортировки, чтобы отсеять крайние значения
    std::vector<double> sortedTimes = runTimes;
    std::sort(sortedTimes.begin(), sortedTimes.end());
    
    // Если запусков больше 2, отбрасываем минимальное и максимальное значение
    double sum = 0.0;
    int count = 0;
    if (NUM_RUNS > 2) {
        for (int i = 1; i < NUM_RUNS - 1; i++) {
            sum += sortedTimes[i];
            count++;
        }
    } else {
        for (double t : sortedTimes) {
            sum += t;
            count++;
        }
    }
    double averageTime = (count > 0) ? (sum / count) : 0.0;
    
    std::cout << "\nТаблица замеров времени:" << std::endl;
    for (int i = 0; i < NUM_RUNS; i++) {
        std::cout << "Run " << std::setw(2) << i+1 << ": " 
                  << std::fixed << std::setprecision(6) 
                  << runTimes[i] << " seconds" << std::endl;
    }
    std::cout << "\nСреднее время: " 
              << std::fixed << std::setprecision(6) 
              << averageTime << " seconds" << std::endl;
    
    return 0;
}
