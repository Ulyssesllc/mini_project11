import os
import pandas as pd
import matplotlib.pyplot as plt


def analyze_and_visualize(csv_path, outdir=None):
    df = pd.read_csv(csv_path)
    # Only keep valid results
    valid_df = df[df["Valid"] == True].copy()

    # Parse test sizes
    def parse_size(row):
        with open(
            os.path.join(os.path.dirname(csv_path), "../test_cases", row["TestCase"]),
            "r",
        ) as f:
            f.readline()  # Skip header line (e.g., 'N,M,K')
            second_line = f.readline().strip()
            # Accept both comma and space separated
            if "," in second_line:
                N, M, K = map(int, second_line.split(","))
            else:
                N, M, K = map(int, second_line.split())
        return pd.Series({"N": N, "M": M, "K": K})

    sizes = valid_df.apply(parse_size, axis=1)
    valid_df = pd.concat([valid_df, sizes], axis=1)
    # Add a column for N+M (total requests)
    valid_df["NplusM"] = valid_df["N"] + valid_df["M"]

    # Display the CSV results as a table for clear comparison
    print("\n=== Full CSV Results Table ===")
    print(valid_df.to_string(index=False))

    # Plot the full CSV results table as a subplot
    fig_csv, ax_csv = plt.subplots(
        figsize=(
            min(20, 2 + 2 * len(valid_df.columns)),
            min(1 + 0.5 * len(valid_df), 20),
        )
    )
    ax_csv.axis("off")
    table_csv = ax_csv.table(
        cellText=valid_df.values,
        colLabels=valid_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table_csv.auto_set_font_size(False)
    table_csv.set_fontsize(10)
    table_csv.auto_set_column_width(col=list(range(len(valid_df.columns))))
    plt.title("Full CSV Results Table", fontsize=14, pad=20)
    plt.tight_layout()
    if outdir:
        plt.savefig(
            os.path.join(outdir, "full_csv_results_table.png"), bbox_inches="tight"
        )

    # Line plots for clear comparison (one line per algorithm)
    # 1. MaxRouteLength vs N+M
    plt.figure(figsize=(12, 6))
    for algo in valid_df["Algorithm"].unique():
        sub = valid_df[valid_df["Algorithm"] == algo]
        # Sort by N+M for line plot
        sub = sub.sort_values("NplusM")
        plt.plot(
            sub["NplusM"],
            sub["MaxRouteLength"],
            marker="o",
            label=algo,
            linestyle="-",
            linewidth=2,
        )
    plt.title("Max Route Length by Algorithm and Problem Size (N+M)")
    plt.ylabel("Max Route Length")
    plt.xlabel("Number of Requests (N+M)")
    plt.legend(title="Algorithm")
    plt.grid(True, linestyle="--", alpha=0.5)
    if outdir:
        plt.savefig(
            os.path.join(outdir, "max_route_length_by_algorithm_line.png"),
            bbox_inches="tight",
        )

    # 2. TotalCost vs N+M
    plt.figure(figsize=(12, 6))
    for algo in valid_df["Algorithm"].unique():
        sub = valid_df[valid_df["Algorithm"] == algo]
        sub = sub.sort_values("NplusM")
        plt.plot(
            sub["NplusM"],
            sub["TotalCost"],
            marker="o",
            label=algo,
            linestyle="-",
            linewidth=2,
        )
    plt.title("Total Cost by Algorithm and Problem Size (N+M)")
    plt.ylabel("Total Cost")
    plt.xlabel("Number of Requests (N+M)")
    plt.legend(title="Algorithm")
    plt.grid(True, linestyle="--", alpha=0.5)
    if outdir:
        plt.savefig(
            os.path.join(outdir, "total_cost_by_algorithm_line.png"),
            bbox_inches="tight",
        )

    # 3. Running Time (ms) vs N+M
    plt.figure(figsize=(12, 6))
    valid_df["TimeMs"] = pd.to_numeric(valid_df["TimeMs"], errors="coerce")
    for algo in valid_df["Algorithm"].unique():
        sub = valid_df[valid_df["Algorithm"] == algo]
        sub = sub.sort_values("NplusM")
        plt.plot(
            sub["NplusM"],
            sub["TimeMs"],
            marker="o",
            label=algo,
            linestyle="-",
            linewidth=2,
        )
    plt.title("Running Time (ms) by Algorithm and Problem Size (N+M)")
    plt.ylabel("Running Time (ms)")
    plt.xlabel("Number of Requests (N+M)")
    plt.legend(title="Algorithm")
    plt.grid(True, linestyle="--", alpha=0.5)
    if outdir:
        plt.savefig(
            os.path.join(outdir, "runtime_by_algorithm_line.png"), bbox_inches="tight"
        )

    # Show all figures at once
    plt.show()

    # Table: Summary statistics
    summary = (
        valid_df.groupby(["Algorithm", "N", "M", "K"])
        .agg(
            avg_max_route=("MaxRouteLength", "mean"),
            std_max_route=("MaxRouteLength", "std"),
            avg_total_cost=("TotalCost", "mean"),
            std_total_cost=("TotalCost", "std"),
            avg_time_ms=("TimeMs", "mean"),
            std_time_ms=("TimeMs", "std"),
            count=("TestCase", "count"),
        )
        .reset_index()
    )
    print("\n=== Summary Statistics by Algorithm and Problem Size (N, M, K) ===")
    print(summary.to_string(index=False))
    if outdir:
        summary.to_csv(os.path.join(outdir, "summary_statistics.csv"), index=False)


if __name__ == "__main__":
    # Example usage
    csv_path = os.path.join(os.path.dirname(__file__), "../results/compare_results.csv")
    outdir = os.path.join(os.path.dirname(__file__), "../results")
    analyze_and_visualize(csv_path, outdir)
