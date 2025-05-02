import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO

data = StringIO("""threads,time_v1,time_v2
1,226.851,228.695
2,118.275,111.8
4,58.8803,60.6097
8,30.5549,30.3276
""")

df = pd.read_csv(data)

# Вычисление ускорения и эффективности
df["speedup_v1"] = df["time_v1"].iloc[0] / df["time_v1"]
df["speedup_v2"] = df["time_v2"].iloc[0] / df["time_v2"]
df["efficiency_v1"] = df["speedup_v1"] / df["threads"]
df["efficiency_v2"] = df["speedup_v2"] / df["threads"]

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

axes[0].plot(df["threads"], df["time_v1"], marker='o', label="Время (вариант 1)")
axes[0].plot(df["threads"], df["time_v2"], marker='o', label="Время (вариант 2)")
axes[0].set_title("Зависимость времени от числа потоков")
axes[0].set_xlabel("Число потоков")
axes[0].set_ylabel("Время (сек)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df["threads"], df["speedup_v1"], marker='o', label="Ускорение (вариант 1)")
axes[1].plot(df["threads"], df["speedup_v2"], marker='o', label="Ускорение (вариант 2)")
axes[1].set_title("Ускорение")
axes[1].set_xlabel("Число потоков")
axes[1].set_ylabel("Speedup")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(df["threads"], df["efficiency_v1"], marker='o', label="Эффективность (вариант 1)")
axes[2].plot(df["threads"], df["efficiency_v2"], marker='o', label="Эффективность (вариант 2)")
axes[2].set_title("Эффективность")
axes[2].set_xlabel("Число потоков")
axes[2].set_ylabel("Efficiency")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("result.png")
