import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)

# Categories and benchmark data
categories = [
    "Adversarial Multi-Hop\n(ANLI / WANLI)",
    "Customer Support\nIntent Triage",
    "Guardrail & Policy\nVerification",
    "Binary Condition\nGating (Noul)",
    "Continuous Severity\nRating (Score)",
    "Macro Benchmark\nAverage",
]

von_scores = [91.23, 94.60, 93.10, 92.80, 89.40, 92.23]
jev_scores = [88.30, 91.80, 89.50, 90.20, 86.10, 89.18]

x = np.arange(len(categories))
width = 0.36

# Set dark technical aesthetic
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)
fig.patch.set_facecolor("#0b0f17")
ax.set_facecolor("#0b0f17")

# Render bars
rects1 = ax.bar(
    x - width / 2,
    von_scores,
    width,
    label="Von-1.0 (Ours - ModernBERT RLCD)",
    color="#00e5ff",
    edgecolor="#00b4d8",
    linewidth=1.2,
    zorder=3,
)
rects2 = ax.bar(
    x + width / 2,
    jev_scores,
    width,
    label="TypeSafe Jev (Cloud API)",
    color="#475569",
    edgecolor="#64748b",
    linewidth=1.2,
    zorder=3,
)

# Axis styling
ax.set_ylabel("Validation Accuracy (%)", fontsize=13, fontweight="bold", color="#f1f5f9", labelpad=12)
ax.set_title(
    "Von-1.0 vs TypeSafe Jev: Decision Task Breakdown",
    fontsize=16,
    fontweight="bold",
    color="#ffffff",
    pad=20,
)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, color="#cbd5e1", fontweight="500")
ax.set_ylim(75, 100)
ax.grid(axis="y", linestyle="--", alpha=0.15, color="#94a3b8", zorder=0)

# Legend styling
legend = ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="#1e293b",
    edgecolor="#334155",
    fontsize=11,
)
for text in legend.get_texts():
    text.set_color("#f8fafc")

# Spines
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_color("#334155")

# Bar value labels with delta highlights
for i in range(len(categories)):
    v_val = von_scores[i]
    j_val = jev_scores[i]
    delta = v_val - j_val

    # Von bar label
    ax.annotate(
        f"{v_val:.1f}%",
        xy=(x[i] - width / 2, v_val),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#00e5ff",
    )

    # Jev bar label
    ax.annotate(
        f"{j_val:.1f}%",
        xy=(x[i] + width / 2, j_val),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="normal",
        color="#94a3b8",
    )

    # Delta badge above pair
    ax.annotate(
        f"+{delta:.1f}%",
        xy=(x[i] - width / 2, max(v_val, j_val)),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#10b981",
    )

plt.tight_layout()
out_path = "assets/benchmark_comparison.png"
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Benchmark plot successfully generated at: {out_path}")
