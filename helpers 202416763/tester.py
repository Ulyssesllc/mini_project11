import os
import csv
import sys
import subprocess
import pandas as pd
from tabulate import tabulate


def parse_problem_input_csv(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    N, M, K = map(int, rows[1])
    q = list(map(int, rows[2][1:]))
    Q = list(map(int, rows[3][1:]))
    d_start = 5 if rows[4][0] == "d_matrix" else 4
    total_points = 2 * N + 2 * M + 1
    d = [list(map(int, rows[d_start + i])) for i in range(total_points)]
    return N, M, K, q, Q, d


def validate_output(N, M, K, q, Q, d, output):
    # Returns (is_valid, max_route_length, error_message)
    lines = output.strip().split("\n")
    try:
        if int(lines[0]) != K:
            return False, None, "K mismatch in output"
        idx = 1
        all_routes = []
        for k in range(K):
            Lk = int(lines[idx])
            route = list(map(int, lines[idx + 1].split()))
            if len(route) != Lk:
                return False, None, f"Route length mismatch for taxi {k + 1}"
            if route[0] != 0 or route[-1] != 0:
                return False, None, f"Route for taxi {k + 1} must start and end at 0"
            all_routes.append(route)
            idx += 2
        # Check passenger directness and parcel capacity
        passenger_pickups = set(range(1, N + 1))
        passenger_dropoffs = set(i + N + M for i in range(1, N + 1))
        parcel_pickups = set(i + N for i in range(1, M + 1))
        parcel_dropoffs = set(i + 2 * N + M for i in range(1, M + 1))
        served_passengers = set()
        served_parcels = set()
        maxlen = 0
        for taxi, route in enumerate(all_routes):
            load = 0
            parcels_onboard = set()
            i = 0
            while i < len(route) - 1:
                pt = route[i]
                nxt = route[i + 1]
                # Passenger directness
                if pt in passenger_pickups:
                    if nxt != pt + N + M:
                        return (
                            False,
                            None,
                            f"Passenger {pt} not direct in taxi {taxi + 1}",
                        )
                    served_passengers.add(pt)
                # Parcel pickup
                if pt in parcel_pickups:
                    pid = pt - N
                    load += q[pid - 1]
                    parcels_onboard.add(pid)
                    if load > Q[taxi]:
                        return (
                            False,
                            None,
                            f"Taxi {taxi + 1} over capacity at point {pt}",
                        )
                    served_parcels.add(pid)
                # Parcel dropoff
                if pt in parcel_dropoffs:
                    pid = pt - 2 * N - M
                    if pid in parcels_onboard:
                        load -= q[pid - 1]
                        parcels_onboard.remove(pid)
                i += 1
            # Route length
            route_len = sum(d[route[j]][route[j + 1]] for j in range(len(route) - 1))
            maxlen = max(maxlen, route_len)
        # All passengers and parcels served?
        if served_passengers != passenger_pickups:
            return False, None, "Not all passengers served"
        if served_parcels != set(range(1, M + 1)):
            return False, None, "Not all parcels served"
        return True, maxlen, ""
    except Exception as e:
        return False, None, f"Output parse error: {e}"


def run_algorithm(algo_path, input_str):
    import time

    try:
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, algo_path],
            input=input_str.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        return result.stdout.decode(), result.stderr.decode(), elapsed_ms
    except subprocess.TimeoutExpired:
        return "", "Timeout", None


def main():
    # Test all .csv cases in the test_cases folder
    test_cases_dir = os.path.join(os.path.dirname(__file__), "../test_cases")
    test_files = [f for f in os.listdir(test_cases_dir) if f.endswith(".csv")]
    test_files.sort()
    algorithms = [
        ("dijkstra", "../algorithms 202416763/dijkstra.py"),
        ("greedy", "../algorithms 202416763/greedy.py"),
        ("hill_climbing", "../algorithms 202416763/hill_climbing.py"),
        ("metaheuristic", "../algorithms 202416763/metaheuristic.py"),
    ]
    all_results = []
    print(
        f"\n[tester.py] Found {len(test_files)} test case(s) in '{test_cases_dir}'.\n"
    )
    for i, test_file in enumerate(test_files, 1):
        print(f"[tester.py] Processing test case {i}/{len(test_files)}: {test_file}")
        csv_path = os.path.join(test_cases_dir, test_file)
        N, M, K, q, Q, d = parse_problem_input_csv(csv_path)
        # Reconstruct the input string for the algorithms
        test_input = (
            f"{N} {M} {K}\n"
            + " ".join(map(str, q))
            + " \n"
            + " ".join(map(str, Q))
            + " \n"
        )
        for row in d:
            test_input += " ".join(map(str, row)) + " \n"
        for j, (name, rel_path) in enumerate(algorithms, 1):
            print(
                f"    [tester.py]   Running algorithm {j}/{len(algorithms)}: {name} ...",
                end=" ",
            )
            algo_path = os.path.join(os.path.dirname(__file__), rel_path)
            out, err, elapsed_ms = run_algorithm(algo_path, test_input)
            valid, maxlen, msg = validate_output(N, M, K, q, Q, d, out)
            total_cost = None
            if valid:
                try:
                    lines = out.strip().split("\n")
                    idx = 1
                    cost = 0
                    for K in range(K):
                        route = list(map(int, lines[idx + 1].split()))
                        route_len = sum(
                            d[route[j]][route[j + 1]] for j in range(len(route) - 1)
                        )
                        cost += route_len
                        idx += 2
                    total_cost = cost
                except Exception:
                    total_cost = None
            if valid:
                print(f"OK (cost={total_cost}, max={maxlen}, t={elapsed_ms:.2f}ms)")
            else:
                print(f"FAILED: {msg if msg else err.strip()}")
            all_results.append(
                {
                    "TestCase": test_file,
                    "Algorithm": name,
                    "Valid": valid,
                    "MaxRouteLength": maxlen if valid else "",
                    "TotalCost": total_cost if valid else "",
                    "TimeMs": f"{elapsed_ms:.2f}"
                    if elapsed_ms is not None
                    else "Timeout",
                    "Error": msg if not valid else "",
                    "Stderr": err.strip(),
                }
            )
    # Write to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "../results/compare_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\n[tester.py] All results written to {csv_path}")

    # --- Custom Table Output (like the PNG) ---
    df = pd.DataFrame(all_results)

    # For each test case, show for each algorithm: running time (ms), total cost, max route length, error
    def cell(row):
        if row["Valid"]:
            return f"cost={row['TotalCost']}, max={row['MaxRouteLength']}, t={row['TimeMs']}ms"
        else:
            return f"F: {row['Error']}"

    df["Result"] = df.apply(cell, axis=1)
    pivot = df.pivot(index="TestCase", columns="Algorithm", values="Result")

    # Add input sizes columns (N, M, K) for each test case
    sizes = {}
    for test_file in df["TestCase"].unique():
        with open(os.path.join(test_cases_dir, test_file), "r") as f:
            f.readline()  # Skip header line (e.g., 'N,M,K')
            second_line = f.readline().strip()
            # Accept both comma and space separated
            if "," in second_line:
                N, M, K = map(int, second_line.split(","))
            else:
                N, M, K = map(int, second_line.split())
            sizes[test_file] = (N, M, K)
    pivot.insert(0, "N", [sizes[t][0] for t in pivot.index])
    pivot.insert(1, "M", [sizes[t][1] for t in pivot.index])
    pivot.insert(2, "K", [sizes[t][2] for t in pivot.index])

    # Print header similar to PNG
    print("\nRESULTS\n")
    print("F: Feasible Solution\nN/A: No Solution Found\n")
    print(tabulate(pivot, headers="keys", tablefmt="grid", showindex=True))

    # --- Save the results table as a PNG file ---
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(min(20, 2 + 2 * len(pivot.columns)), min(1 + 0.5 * len(pivot), 20))
        )
        ax.axis("off")
        table = ax.table(
            cellText=pivot.values,
            colLabels=pivot.columns,
            rowLabels=pivot.index,
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(pivot.columns))))
        plt.title("Results Table", fontsize=14, pad=20)
        plt.tight_layout()
        outdir = os.path.join(os.path.dirname(__file__), "../results")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "results_table.png")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"[tester.py] Results table saved as PNG to {outpath}")
    except Exception as e:
        print(f"[tester.py] Could not save results table as PNG: {e}")

    print(
        "\n[tester.py] All tests complete. Proceeding to analysis and visualization.\n"
    )


if __name__ == "__main__":
    main()
    # After running tests, automatically analyze and visualize results
    try:
        from analyze_results import analyze_and_visualize
    except ImportError:
        import importlib.util

        _analyze_path = os.path.join(os.path.dirname(__file__), "analyze_results.py")
        spec = importlib.util.spec_from_file_location("analyze_results", _analyze_path)
        analyze_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyze_mod)
        analyze_and_visualize = analyze_mod.analyze_and_visualize
    csv_path = os.path.join(os.path.dirname(__file__), "../results/compare_results.csv")
    outdir = os.path.join(os.path.dirname(__file__), "../results")
    print("\n--- Analysis and Visualization ---")
    analyze_and_visualize(csv_path, outdir)
