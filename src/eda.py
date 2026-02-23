import os
import matplotlib
matplotlib.use("Agg")  # Prevents GUI-related crashes
import matplotlib.pyplot as plt
import seaborn as sns


def create_output_dir():
    os.makedirs("reports/figures", exist_ok=True)


def plot_histogram(df):
    """
    Creates histogram of median income (MedInc).
    """
    plt.figure()
    df["MedInc"].hist(bins=30)
    plt.title("Histogram of Median Income")
    plt.xlabel("Median Income")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("reports/figures/histogram_medinc.png")
    plt.close()


def plot_scatter(df):
    """
    Scatter plot between MedInc and MedHouseVal.
    """
    plt.figure()
    plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.5)
    plt.title("Median Income vs Median House Value")
    plt.xlabel("Median Income")
    plt.ylabel("Median House Value")
    plt.tight_layout()
    plt.savefig("reports/figures/scatter_medinc_vs_value.png")
    plt.close()


def plot_correlation_heatmap(df):
    """
    Correlation heatmap for all features.
    """
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()


def run_eda(df):
    create_output_dir()
    plot_histogram(df)
    plot_scatter(df)
    plot_correlation_heatmap(df)

    print("EDA plots saved to reports/figures/")
