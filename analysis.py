# analysis.py
# Marimo interactive notebook for TDS Assignment
# Contact: 22f2001145@ds.study.iitm.ac.in
# Purpose: Demonstrate variable dependencies, interactive widgets, and dynamic markdown output.
# Data flow:
#   Cell 1 → creates dataset (df)
#   Cell 2 → depends on df and slider widget
#   Cell 3 → depends on smoothed values from Cell 2

import marimo as mo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Cell 1 — Base dataset (downstream cells depend on df)
# This cell defines x, y, and df. All dependent calculations must use this df.
np.random.seed(123)
x = np.linspace(0, 50, 150)
y = 0.3 * x + 8 * np.sin(0.4 * x) + np.random.normal(0, 4, x.shape)
df = pd.DataFrame({"x": x, "y": y})

mo.md("### 📊 Dataset Preview")
mo.md(df.head().to_markdown())

# %% Cell 2 — Interactive slider + dependent computation
# This cell depends on 'df'. Modifying the slider triggers recomputation.
slider = mo.ui.slider(1, 25, value=5, label="Smoothing window")

w = int(slider.value)
smoothed = df["y"].rolling(window=w, center=True, min_periods=1).mean()

mo.md(
    f"""
## 🔧 Smoothing Summary  
- Selected window: **{w}**  
- Mean (original): **{df['y'].mean():.2f}**  
- Mean (smoothed): **{smoothed.mean():.2f}**  
- Std (original): **{df['y'].std():.2f}**  
- Std (smoothed): **{smoothed.std():.2f}**
"""
)

# Plot original vs smoothed
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(df["x"], df["y"], alpha=0.5, label="Original")
ax.plot(df["x"], smoothed, label=f"Smoothed (w={w})", linewidth=2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(alpha=0.2)
ax.legend()

mo.display(fig)

# %% Cell 3 — Dependent result using smoothed data
peak_idx = smoothed.idxmax()
peak_x = df.loc[peak_idx, "x"]
peak_y = smoothed.max()

mo.md(f"### 📈 Peak after smoothing\n**Peak at x={peak_x:.2f}, y={peak_y:.2f}**")
