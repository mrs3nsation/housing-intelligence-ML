import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_bubble_chart(user_income, user_price, user_population):

    os.makedirs("static/plots", exist_ok=True)

    # Synthetic sample data
    np.random.seed(42)
    sample_income = np.random.uniform(1, 10, 80)
    sample_price = sample_income * 0.5 + np.random.normal(0, 0.5, 80)
    sample_population = np.random.uniform(500, 5000, 80)

    sizes = sample_population / 20

    plt.figure(figsize=(7, 6))

    # Background market data
    plt.scatter(
        sample_income,
        sample_price,
        s=sizes,
        alpha=0.4,
        label="Market Sample Data"
    )

    # User house
    plt.scatter(
        user_income,
        user_price,
        s=user_population / 10,
        color="red",
        edgecolors="black",
        linewidth=2,
        label="Your Predicted House"
    )

    plt.xlabel("Median Income")
    plt.ylabel("Median House Value (in $100k)")
    plt.title("Housing Market Positioning")

    # Add legend
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("static/plots/user_analysis.png")
    plt.close()

    return "plots/user_analysis.png"