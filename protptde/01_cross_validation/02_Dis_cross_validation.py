import os
import pandas as pd
import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["font.size"] = 18
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 2
plt.rcParams["axes.prop_cycle"] = cycler("color", ["#74add1", "#ff7f0e", "#2ca02c", "#d62728", "#956cb4"])


data = pd.read_csv("best_score_results/random_seed_0.csv")
data = data[["value", "params_model_combination", "params_max_lr", "params_num_layer"]]

for i in range(1, 50):
    tmp = pd.read_csv(f"best_score_results/random_seed_{i}.csv")
    tmp = tmp[["value", "params_model_combination", "params_max_lr", "params_num_layer"]]
    data = pd.concat([data, tmp])
data.reset_index(drop=True, inplace=True)
data.columns = ["Value", "Model combination", "Max learning rate", "Num layer"]

fig, ax = plt.subplots()

fig = sns.displot(data, x="Value", hue="Model combination", col="Num layer", row="Max learning rate", facet_kws=dict(margin_titles=True), kind="kde")

fig.set_xlabels(fontsize=28)
fig.set_ylabels(fontsize=28)
fig._legend.set_title("Model combination")
plt.setp(fig._legend.get_texts(), fontsize=26)
plt.setp(fig._legend.get_title(), fontsize=28)
fig._legend.set_bbox_to_anchor((1.02, 0.5))

plt.savefig(os.path.basename(__file__).split(".")[0].split("_", 1)[-1] + ".pdf", bbox_inches="tight")
