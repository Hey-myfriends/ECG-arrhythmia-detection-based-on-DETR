import torch, os, pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_logs(log, fields=("loss", "loss_ce", "loss_bbox", "loss_giou"), log_name="log.txt", ewm_col=1):
    func_name = "plot_utils.py::plot_logs"
    # pdb.set_trace()
    assert os.path.isdir(log), f"{func_name} - log must be a dir..."
    if not os.path.exists(log):
        raise ValueError(f"{func_name} - logs not exist...")

    df = pd.read_json(os.path.join(log, log_name), lines=True)
    fig, axs = plt.subplots(1, len(fields), figsize=(16, 5))

    for df, color in zip([df], sns.color_palette(n_colors=1)):
        for j, field in enumerate(fields):
            df.interpolate().ewm(com=ewm_col).mean().plot(
                y=[f"train_{field}", f"test_{field}"],
                ax=axs[j],
                color=[color] * 2,
                style=["-", "--"]
            )
    for ax, field in zip(axs, fields):
        ax.legend([f"train_{field}", f"test_{field}"])
        ax.set_title(field)
        ax.set_xlabel("Epoch")
    log_name = log_name.split(".")[0]
    plt.savefig(os.path.join(log, f"{log_name}.png"), dpi=300)
    print(f"log figure is saved in ", os.path.join(log, f"{log_name}.png"), "...")

if __name__ == "__main__":
    log = "D:\\Desktop\\ECG分类研究\\code\\outputs"
    plot_logs(log, log_name="log01.txt")
