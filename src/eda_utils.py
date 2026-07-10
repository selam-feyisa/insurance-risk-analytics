"""
eda_utils.py

Reusable exploratory data analysis helpers for the ACIS insurance dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def overall_loss_ratio(df: pd.DataFrame) -> float:
    """Compute the portfolio-wide loss ratio: sum(TotalClaims) / sum(TotalPremium)."""
    total_claims = df["TotalClaims"].sum()
    total_premium = df["TotalPremium"].sum()
    return total_claims / total_premium if total_premium else float("nan")


def loss_ratio_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute loss ratio, total premium, total claims, and policy count
    for each category in `group_col`.
    """
    grouped = df.groupby(group_col).agg(
        total_premium=("TotalPremium", "sum"),
        total_claims=("TotalClaims", "sum"),
        policy_count=("TotalPremium", "size"),
    )
    grouped["loss_ratio"] = grouped["total_claims"] / grouped["total_premium"]
    return grouped.sort_values("loss_ratio", ascending=False)


def plot_loss_ratio_by_group(df: pd.DataFrame, group_col: str, top_n: int = 15,
                              title: str = None, ax=None):
    """Bar plot of loss ratio by group, sorted descending."""
    summary = loss_ratio_by_group(df, group_col).head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x=summary["loss_ratio"], y=summary.index, ax=ax, palette="rocket")
    ax.set_xlabel("Loss Ratio (TotalClaims / TotalPremium)")
    ax.set_ylabel(group_col)
    ax.set_title(title or f"Loss Ratio by {group_col}")
    plt.tight_layout()
    return ax


def plot_numeric_distribution(df: pd.DataFrame, column: str, bins: int = 50,
                               clip_quantile: float = 0.99, ax=None):
    """
    Histogram of a numeric column, clipped at a given upper quantile
    to keep extreme outliers from flattening the plot.
    """
    data = df[column].dropna()
    upper = data.quantile(clip_quantile)
    clipped = data[data <= upper]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(clipped, bins=bins, kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {column} (clipped at {int(clip_quantile*100)}th pct)")
    ax.set_xlabel(column)
    plt.tight_layout()
    return ax


def plot_categorical_counts(df: pd.DataFrame, column: str, top_n: int = 15, ax=None):
    """Bar plot of value counts for a categorical column."""
    counts = df[column].value_counts().head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="mako")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    ax.set_title(f"Top {top_n} categories in {column}")
    plt.tight_layout()
    return ax


def plot_outliers_boxplot(df: pd.DataFrame, columns: list[str], clip_quantile: float = 0.99):
    """Side-by-side boxplots for a list of numeric columns, clipped to reduce
    extreme scale distortion while still showing outlier spread."""
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        data = df[col].dropna()
        upper = data.quantile(clip_quantile)
        clipped = data[data <= upper]
        sns.boxplot(y=clipped, ax=ax, color="salmon")
        ax.set_title(f"{col} (clipped at {int(clip_quantile*100)}th pct)")

    plt.tight_layout()
    return fig


def monthly_trend(df: pd.DataFrame, date_col: str = "TransactionMonth") -> pd.DataFrame:
    """
    Aggregate claim frequency and severity by calendar month to
    reveal temporal trends over the 18-month observation window.
    """
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp[date_col]).dt.to_period("M")

    monthly = tmp.groupby("month").agg(
        total_premium=("TotalPremium", "sum"),
        total_claims=("TotalClaims", "sum"),
        claim_count=("TotalClaims", lambda x: (x > 0).sum()),
        policy_count=("TotalPremium", "size"),
    )
    monthly["claim_frequency"] = monthly["claim_count"] / monthly["policy_count"]
    monthly["loss_ratio"] = monthly["total_claims"] / monthly["total_premium"]
    return monthly


def plot_monthly_trend(df: pd.DataFrame, date_col: str = "TransactionMonth"):
    """Line plot of claim frequency and loss ratio over time."""
    monthly = monthly_trend(df, date_col)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    monthly["claim_frequency"].plot(ax=axes[0], marker="o", color="darkorange")
    axes[0].set_title("Claim Frequency Over Time")
    axes[0].set_ylabel("Claim Frequency")

    monthly["loss_ratio"].plot(ax=axes[1], marker="o", color="crimson")
    axes[1].set_title("Loss Ratio Over Time")
    axes[1].set_ylabel("Loss Ratio")
    axes[1].set_xlabel("Month")

    plt.tight_layout()
    return fig


def top_bottom_by_claims(df: pd.DataFrame, group_col: str = "make", n: int = 10) -> tuple:
    """
    Return the top-n and bottom-n groups (e.g. vehicle makes) by average
    claim amount among policies that had at least one claim.
    """
    claimed = df[df["TotalClaims"] > 0]
    avg_claims = claimed.groupby(group_col)["TotalClaims"].mean().sort_values()

    bottom = avg_claims.head(n)
    top = avg_claims.tail(n).sort_values(ascending=False)
    return top, bottom


def correlation_matrix(df: pd.DataFrame, columns: list[str], ax=None):
    """Heatmap of the correlation matrix for a given set of numeric columns."""
    corr = df[columns].corr()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return ax
