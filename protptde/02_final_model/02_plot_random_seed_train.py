import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
import plotly.express as px
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["font.size"] = 18
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 8
plt.rcParams["axes.prop_cycle"] = cycler("color", ["#74add1", "#ff7f0e", "#2ca02c", "#d62728", "#956cb4"])

data = pd.DataFrame()
for i in range(100):
    tmp = pd.read_csv(f"results/{i}/loss.csv", index_col=0)
    data.loc[i, "random_seed"] = i
    data.loc[i, "best_train_epoch_ratio"] = tmp["train_loss"].argmin() / len(tmp) * 100
    data.loc[i, "best_test_epoch_ratio"] = tmp["test_corr"].argmax() / len(tmp) * 100
    data.loc[i, "train_loss"] = tmp.iloc[tmp["test_corr"].argmax(), 0]
    data.loc[i, "test_corr"] = tmp["test_corr"].max()

result = pd.DataFrame()
result["train"] = data["best_train_epoch_ratio"]
result["test"] = data["best_test_epoch_ratio"]
result["index"] = np.arange(len(result))
result = result.melt(id_vars=["index"], var_name="dataset", value_name="epoch ratio")

fig, ax = plt.subplots()
sns.histplot(data=result, x="epoch ratio", hue="dataset", kde=True, ax=ax, bins=40)
ax.set(xlabel="Best epoch ratio (%)", ylabel="Count")
plt.savefig("Hist_best_train_test_epoch_ratio.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
sns.scatterplot(data=data, x="best_train_epoch_ratio", y="best_test_epoch_ratio", hue="test_corr", palette="coolwarm", ax=ax)
ax.set(xlabel="Best train epoch ratio (%)", ylabel="Best test epoch ratio (%)")
plt.savefig("Scatter_best_train_test_epoch_ratio.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
g = sns.jointplot(data=data, x="best_train_epoch_ratio", y="best_test_epoch_ratio", kind="scatter")
g.ax_joint.set_xlabel("Best train epoch ratio (%)", fontsize=12)
g.ax_joint.set_ylabel("Best test epoch ratio (%)", fontsize=12)
g.ax_joint.set_xlim(0, 100)
g.ax_joint.set_ylim(0, 100)
g.ax_joint.tick_params(axis="x", labelsize=10)
g.ax_joint.tick_params(axis="y", labelsize=10)
plt.savefig("Hist_Scatter_best_train_test_epoch_ratio.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
sns.histplot(data["test_corr"], bins=25, kde=True, ax=ax, legend=False)
ax.set(xlabel="Test dataset Spearman " + r"$\rho$", ylabel="Count", xlim=(0, 1))
plt.savefig("Hist_test_corr.pdf", bbox_inches="tight")

fig = px.scatter(data, x="best_train_epoch_ratio", y="best_test_epoch_ratio", color="test_corr", hover_name="random_seed")
fig.update_layout(
    xaxis_title="Best train epoch ratio (%)",
    yaxis_title="Best test epoch ratio (%)",
    title="Best train/test epoch ratio",
)
fig.write_html("Scatter_best_train_test_epoch_ratio.html")

counts, bins = np.histogram(data["test_corr"], bins=25)
max_bin_index = np.argmax(counts)
bin_start = bins[max_bin_index]
bin_end = bins[max_bin_index + 1]
peak_models = data[(data["test_corr"] >= bin_start) & (data["test_corr"] < bin_end)]
peak_median = peak_models["test_corr"].median()
selected_index = (peak_models["test_corr"] - peak_median).abs().idxmin()
selected_index = int(selected_index)
print("Selected Model Index:", selected_index, "Test Corr:", data.loc[selected_index, "test_corr"])
