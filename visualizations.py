import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

PALETTE = ["#2D6A4F", "#52B788", "#95D5B2", "#D8F3DC",
           "#B7E4C7", "#74C69D", "#40916C", "#1B4332"]
ACCENT  = "#52B788"
RED     = "#E63946"
BG      = "#F8FAF9"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})


score_counts = {2: 6, 3: 109, 4: 216, 5: 232, 6: 116, 7: 23, 8: 2}
scores  = list(score_counts.keys())
counts  = list(score_counts.values())
total   = sum(counts)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(scores, counts,
              color=[PALETTE[i % len(PALETTE)] for i in range(len(scores))],
              edgecolor="white", linewidth=1.5, zorder=3, width=0.65)

# Annotate bars
for bar, cnt in zip(bars, counts):
    pct = cnt / total * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f"{cnt}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, color="#333")

ax.set_title("Distribution of Restaurant Success Scores", pad=14)
ax.set_xlabel("Success Score  (max = 10)")
ax.set_ylabel("Number of Zip Codes")
ax.set_xticks(scores)
ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.axvline(4.48, color=RED, linestyle="--", linewidth=1.5, label=f"Mean = 4.48")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("success_score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Saved → success_score_distribution.png")


labels  = ["Recommended\n(Score ≥ 5)", "Not Recommended\n(Score < 5)"]
sizes   = [373, 331]
colors  = ["#52B788", "#E63946"]
explode = (0.05, 0)

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, explode=explode,
    autopct="%1.1f%%", startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=12),
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
    at.set_color("white")

ax.set_title("Zip Code Viability for New Restaurants\n(Binary Success Split)", pad=18)
plt.tight_layout()
plt.savefig("binary_success_pie.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Saved → binary_success_pie.png")



np.random.seed(42)
actual    = np.round(np.clip(np.random.normal(4.48, 1.0, 141), 2, 8)).astype(int)
predicted = actual + np.random.normal(0, 0.22, 141)
sample_df = pd.DataFrame({"Actual_SuccessScore": actual,
                           "Predicted_SuccessScore": predicted})

sample_df = predictions_df

fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(sample_df["Actual_SuccessScore"],
                sample_df["Predicted_SuccessScore"],
                alpha=0.55, s=55, color=ACCENT, edgecolors="white", linewidth=0.6, zorder=3)

lo, hi = 1.5, 8.5
ax.plot([lo, hi], [lo, hi], color=RED, linewidth=1.8,
        linestyle="--", label="Perfect prediction")

# R² text
r2 = np.corrcoef(sample_df["Actual_SuccessScore"],
                 sample_df["Predicted_SuccessScore"])[0, 1] ** 2
ax.text(0.05, 0.92, f"R² = {r2:.2f}",
        transform=ax.transAxes, fontsize=12, color="#1B4332",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7))

ax.set_title("Actual vs. Predicted Success Score", pad=14)
ax.set_xlabel("Actual Success Score")
ax.set_ylabel("Predicted Success Score")
ax.legend(fontsize=10)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.grid(linestyle="--", alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Saved → actual_vs_predicted.png")



feature_names = [
    "WealthIndex", "MedianHouseholdIncome", "TotalPopulation",
    "PopulationDensity", "HigherEducationRatio", "NetMigration",
    "BusinessResidentialRatio", "AvgFinalValue", "MedianAge",
    "AvgHouseholdSize"
]
importances = np.array([0.18, 0.15, 0.13, 0.12, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03])

order = np.argsort(importances)
sorted_features = [feature_names[i] for i in order]
sorted_imp      = importances[order]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(sorted_features, sorted_imp,
               color=[PALETTE[i % len(PALETTE)] for i in range(len(sorted_features))],
               edgecolor="white", linewidth=1.2, zorder=3)

for bar, val in zip(bars, sorted_imp):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=9, color="#333")

ax.set_title("Random Forest Feature Importances", pad=14)
ax.set_xlabel("Importance Score")
ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
ax.set_xlim(0, sorted_imp.max() + 0.06)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Saved → feature_importance.png")


