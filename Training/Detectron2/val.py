import pandas as pd
import matplotlib.pyplot as plt


metrics_df = pd.read_json("output/metrics.json", orient="records", lines=True)
mdf = metrics_df.sort_values("iteration")


# 1. Loss curve
def plot_graph(mdf):
    fig, ax = plt.subplots()
    mdf1 = mdf[~mdf["total_loss"].isna()]
    ax.plot(mdf1["iteration"], mdf1["total_loss"], c="C0", label="train")
    if "validation_loss" in mdf.columns:
        mdf2 = mdf[~mdf["validation_loss"].isna()]
        ax.plot(mdf2["iteration"], mdf2["validation_loss"], c="C1", label="validation")
    # ax.set_ylim([0, 0.5])
    ax.legend()
    ax.set_title("Loss curve")
    plt.show()
    # plt.savefig(outdir/"loss.png")


def plot_accuracy(mdf, mode):
    fig, ax = plt.subplots()
    mdf3 = mdf[~mdf[f"{mode}/AP50"].isna()]
    ax.plot(mdf3["iteration"], mdf3[f"{mode}/AP50"] / 100., c="C2", label="validation")
    best_value=mdf3[f"{mode}/AP50"].max()
    indx_best=mdf3[mdf3[f"{mode}/AP50"] == mdf3[f"{mode}/AP50"].max()].index.values
    iter_best=mdf3["iteration"][indx_best[0]]
    print(f"Best value is {best_value} one iteration is {iter_best} ")
    ax.legend()
    ax.set_title("AP50")
    plt.show()


def plot_accuracy_each(mdf, mode):
    selected_lbels=["Sidewalk",
                "Bikelane",
                "Road",
                "Crosswalk",
                "Obstacle",
                "Stairs",
                "Car",
                "Bike",
                "E-Scooter",
                "Person",
                "Traffic Light"]

    mdf3 = mdf[~mdf[f"{mode}/AP50"].isna()]
    fig, ax = plt.subplots()
    mdf_bbox_class = mdf3.iloc[-1][[f"{mode}/AP-{col}" for col in selected_lbels]]
    mdf_bbox_class.plot(kind="bar", ax=ax)
    _ = ax.set_title("AP by class")
    plt.show()


plot_accuracy_each(mdf, mode='segm')
plot_accuracy(mdf, mode='segm')
plot_graph(mdf)
