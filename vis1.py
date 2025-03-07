import matplotlib.pyplot as plt
import numpy as np

threads = np.array([1, 2, 4, 7, 8, 16, 20, 40])
speedup_20k = np.array([1.299753, 0.586714, 0.322875, 0.217232, 0.19188, 0.141028, 0.136801, 0.133019]) 
speedup_40k = np.array([4.804039, 2.536103, 1.279579, 0.843520, 0.795286, 0.519911, 0.538549, 0.534682])

calculated_speedup_20k = speedup_20k[0] / speedup_20k
calculated_speedup_40k = speedup_40k[0] / speedup_40k

plt.figure(figsize=(8, 6))
plt.plot(threads, calculated_speedup_20k, marker='o', label="Размер 20000x20000")
plt.plot(threads, calculated_speedup_40k, marker='s', label="Размер 40000x40000")
plt.plot([0, max(threads)], [0, max(threads)], "--", color="gray", label="Идеальное ускорение (S=p)")

plt.xlabel("Количество потоков")
plt.ylabel("Ускорение")
plt.legend()
plt.title("Масштабируемость умножения матрицы на вектор")
plt.grid()

output_file = "res1.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"График сохранён в {output_file}")
