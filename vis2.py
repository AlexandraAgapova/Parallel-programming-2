import matplotlib.pyplot as plt
import numpy as np

threads = np.array([1, 2, 4, 7, 8, 16, 20, 40])
speedup = np.array([0.83, 1.66, 2.03, 3.09, 5.80, 9.31, 15.46, 9.08]) 


plt.figure(figsize=(8, 6))
plt.plot(threads, speedup, marker='s', label="Nstep = 40000000")
plt.plot([0, max(threads)], [0, max(threads)], "--", color="gray", label="Идеальное ускорение")

plt.xlabel("Количество потоков")
plt.ylabel("Ускорение")
plt.legend()
plt.title("Масштабируемость численного интегрирования")
plt.grid()

output_file = "res2.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"График сохранён в {output_file}")
