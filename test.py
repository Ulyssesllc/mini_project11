import os
import sys
import time
import subprocess
import glob
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import numpy as np


class MultiSolutionFeasibilityRunner:
    def __init__(self, solution_files):
        self.solution_files = solution_files
        self.results = {}  # Will store results for each solution
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"multi_feasibility_results_{self.timestamp}.txt"
        self.csv_file = f"multi_feasibility_results_{self.timestamp}.csv"
        self.comparison_csv = f"algorithm_feasibility_comparison_{self.timestamp}.csv"

        # Initialize results dictionary
        for solution in solution_files:
            self.results[solution] = []

    def find_test_cases(self):
        """Find all test case files in the current directory"""
        test_files = glob.glob("test_case_*.txt")
        if not test_files:
            print("No test case files found!")
            print("Run the test case generator first to create test cases.")
            return []

        test_files.sort()
        return test_files

    def calculate_route_cost(self, route, distance_matrix):
        """Calculate the total cost (distance) of a route"""
        if len(route) < 2:
            return 0

        total_cost = 0
        for i in range(len(route) - 1):
            from_point = route[i]
            to_point = route[i + 1]
            if from_point < len(distance_matrix) and to_point < len(
                distance_matrix[from_point]
            ):
                total_cost += distance_matrix[from_point][to_point]
            else:
                raise Exception(f"Invalid route point: {from_point} -> {to_point}")

        return total_cost

    def parse_distance_matrix(self, test_input):
        """Parse distance matrix from test input"""
        lines = test_input.strip().split("\n")
        n, m, k = map(int, lines[0].split())

        # Distance matrix starts from line 3 (0-indexed)
        matrix_start = 3
        total_points = 2 * n + 2 * m + 1

        distance_matrix = []
        for i in range(total_points):
            if matrix_start + i < len(lines):
                row = list(map(int, lines[matrix_start + i].split()))
                distance_matrix.append(row)
            else:
                raise Exception("Incomplete distance matrix")

        return distance_matrix

    def parse_customer_locations(self, test_input):
        """Parse customer pickup and delivery locations"""
        lines = test_input.strip().split("\n")
        n, m, k = map(int, lines[0].split())

        pickup_locations = list(map(int, lines[1].split()))
        delivery_locations = list(map(int, lines[2].split()))

        return pickup_locations, delivery_locations

    def check_feasibility(
        self, routes, pickup_locations, delivery_locations, distance_matrix
    ):
        """Check if the solution is feasible"""
        feasibility_issues = []

        # Track which customers are served
        customers_served = set()

        for route_idx, route in enumerate(routes):
            route_issues = []

            # Check if route starts and ends at depot
            if not route or route[0] != 0:
                route_issues.append("Route doesn't start at depot (0)")
            if not route or route[-1] != 0:
                route_issues.append("Route doesn't end at depot (0)")

            # Check route feasibility (pickup before delivery)
            route_customers = {}  # customer_id -> (pickup_visited, delivery_visited)

            for point in route[1:-1]:  # Exclude depot at start and end
                # Check if this is a pickup location
                for customer_id, pickup_loc in enumerate(pickup_locations):
                    if point == pickup_loc:
                        if customer_id not in route_customers:
                            route_customers[customer_id] = [False, False]
                        route_customers[customer_id][0] = True
                        customers_served.add(customer_id)

                # Check if this is a delivery location
                for customer_id, delivery_loc in enumerate(delivery_locations):
                    if point == delivery_loc:
                        if customer_id not in route_customers:
                            route_customers[customer_id] = [False, False]

                        # Check if pickup was visited before delivery
                        if not route_customers[customer_id][0]:
                            route_issues.append(
                                f"Customer {customer_id} delivery before pickup in route {route_idx}"
                            )

                        route_customers[customer_id][1] = True

            # Check if all customers in this route have both pickup and delivery
            for customer_id, (
                pickup_visited,
                delivery_visited,
            ) in route_customers.items():
                if pickup_visited and not delivery_visited:
                    route_issues.append(
                        f"Customer {customer_id} pickup without delivery in route {route_idx}"
                    )
                elif delivery_visited and not pickup_visited:
                    route_issues.append(
                        f"Customer {customer_id} delivery without pickup in route {route_idx}"
                    )

            if route_issues:
                feasibility_issues.extend(
                    [f"Route {route_idx}: {issue}" for issue in route_issues]
                )

        # Check if all customers are served
        total_customers = len(pickup_locations)
        unserved_customers = set(range(total_customers)) - customers_served
        if unserved_customers:
            feasibility_issues.append(f"Unserved customers: {list(unserved_customers)}")

        # Check for duplicate customer service
        all_served_customers = []
        for route in routes:
            for point in route[1:-1]:
                for customer_id, pickup_loc in enumerate(pickup_locations):
                    if point == pickup_loc:
                        all_served_customers.append(f"pickup_{customer_id}")
                for customer_id, delivery_loc in enumerate(delivery_locations):
                    if point == delivery_loc:
                        all_served_customers.append(f"delivery_{customer_id}")

        # Check for duplicates
        seen = set()
        duplicates = set()
        for service in all_served_customers:
            if service in seen:
                duplicates.add(service)
            seen.add(service)

        if duplicates:
            feasibility_issues.append(f"Duplicate services: {list(duplicates)}")

        return len(feasibility_issues) == 0, feasibility_issues

    def run_single_solution_test(
        self, solution_file, test_file, test_input, distance_matrix, n, m, k
    ):
        """Run a single test case for one solution"""
        try:
            pickup_locations, delivery_locations = self.parse_customer_locations(
                test_input
            )

            # Run the solution
            start_time = time.time()

            # Method 1: Try to import and run as module
            try:
                result = self.run_as_module(solution_file, test_input)
                execution_time = time.time() - start_time
                return self.process_result(
                    test_file,
                    result,
                    execution_time,
                    test_input,
                    distance_matrix,
                    pickup_locations,
                    delivery_locations,
                    n,
                    m,
                    k,
                    solution_file,
                )
            except Exception as e:
                # Method 2: Try to run as subprocess
                try:
                    result = self.run_as_subprocess(solution_file, test_input)
                    execution_time = time.time() - start_time
                    return self.process_result(
                        test_file,
                        result,
                        execution_time,
                        test_input,
                        distance_matrix,
                        pickup_locations,
                        delivery_locations,
                        n,
                        m,
                        k,
                        solution_file,
                    )
                except Exception as e2:
                    return {
                        "test_case": test_file,
                        "solution": solution_file,
                        "status": "ERROR",
                        "feasible": False,
                        "error": f"Import error: {str(e)}, Subprocess error: {str(e2)}",
                        "execution_time": 0,
                        "n": n,
                        "m": m,
                        "k": k,
                        "total_cost": 0,
                        "max_route_cost": 0,
                        "avg_route_cost": 0,
                        "num_routes": 0,
                        "max_route_length": 0,
                        "feasibility_issues": [],
                    }

        except Exception as e:
            return {
                "test_case": test_file,
                "solution": solution_file,
                "status": "ERROR",
                "feasible": False,
                "error": f"Execution error: {str(e)}",
                "execution_time": 0,
                "n": n,
                "m": m,
                "k": k,
                "total_cost": 0,
                "max_route_cost": 0,
                "avg_route_cost": 0,
                "num_routes": 0,
                "max_route_length": 0,
                "feasibility_issues": [],
            }

    def run_as_module(self, solution_file, test_input):
        """Try to run solution as imported module"""
        # Remove .py extension for import
        module_name = solution_file.replace(".py", "")

        # Import the solution module
        if module_name in sys.modules:
            del sys.modules[module_name]

        solution_module = __import__(module_name)

        # Try different common function names
        possible_functions = ["solve", "main", "taxi_routing", "solve_taxi_routing"]

        for func_name in possible_functions:
            if hasattr(solution_module, func_name):
                func = getattr(solution_module, func_name)
                # Try to call with input
                try:
                    return func(test_input)
                except:
                    # Try calling without parameters (reads from stdin)
                    import io

                    old_stdin = sys.stdin
                    sys.stdin = io.StringIO(test_input)
                    try:
                        result = func()
                        return result
                    finally:
                        sys.stdin = old_stdin

        # If no specific function found, try to execute the module
        import io

        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO(test_input)
        sys.stdout = io.StringIO()

        try:
            exec(compile(open(solution_file).read(), solution_file, "exec"))
            result = sys.stdout.getvalue()
            return result
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout

    def run_as_subprocess(self, solution_file, test_input):
        """Run solution as subprocess"""
        process = subprocess.Popen(
            [sys.executable, solution_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(
            input=test_input, timeout=60
        )  # Increased timeout for multiple solutions

        if process.returncode != 0:
            raise Exception(
                f"Process failed with return code {process.returncode}: {stderr}"
            )

        return stdout

    def process_result(
        self,
        test_file,
        result,
        execution_time,
        test_input,
        distance_matrix,
        pickup_locations,
        delivery_locations,
        n,
        m,
        k,
        solution_file,
    ):
        """Process and validate the result for feasibility"""
        try:
            # Parse output
            output_lines = result.strip().split("\n")
            if not output_lines or not output_lines[0].strip():
                return {
                    "test_case": test_file,
                    "solution": solution_file,
                    "status": "ERROR",
                    "feasible": False,
                    "error": "Empty output",
                    "execution_time": execution_time,
                    "n": n,
                    "m": m,
                    "k": k,
                    "total_cost": 0,
                    "max_route_cost": 0,
                    "avg_route_cost": 0,
                    "num_routes": 0,
                    "max_route_length": 0,
                    "feasibility_issues": ["Empty output"],
                }

            # Try to parse the number of routes (more flexible)
            try:
                output_k = int(output_lines[0])
            except ValueError:
                # If first line isn't a number, try to parse routes anyway
                output_k = k  # Assume expected number of routes

            # Parse routes with flexible format handling
            routes = []
            route_costs = []
            max_route_length = 0
            parsing_errors = []

            # Try different parsing strategies
            if len(output_lines) >= 2 * output_k + 1:
                # Standard format: K, then pairs of (route_length, route)
                for i in range(1, len(output_lines), 2):
                    if i + 1 >= len(output_lines):
                        break
                    try:
                        route_length = int(output_lines[i])
                        route = list(map(int, output_lines[i + 1].split()))
                        routes.append(route)
                    except (ValueError, IndexError) as e:
                        parsing_errors.append(
                            f"Route parsing error at line {i}: {str(e)}"
                        )
            else:
                # Alternative format: try to parse as just routes
                for i in range(1, min(len(output_lines), k + 1)):
                    try:
                        route = list(map(int, output_lines[i].split()))
                        if route:  # Only add non-empty routes
                            routes.append(route)
                    except (ValueError, IndexError) as e:
                        parsing_errors.append(
                            f"Route parsing error at line {i}: {str(e)}"
                        )

            # Calculate route costs and statistics
            valid_routes = []
            for route in routes:
                try:
                    if len(route) >= 2:  # At least depot -> depot
                        route_cost = self.calculate_route_cost(route, distance_matrix)
                        route_costs.append(route_cost)
                        valid_routes.append(route)
                        max_route_length = max(max_route_length, len(route) - 1)
                except Exception as e:
                    parsing_errors.append(f"Cost calculation error: {str(e)}")

            # Check feasibility
            is_feasible, feasibility_issues = self.check_feasibility(
                valid_routes, pickup_locations, delivery_locations, distance_matrix
            )

            # Combine parsing errors with feasibility issues
            all_issues = parsing_errors + feasibility_issues

            # Calculate final statistics
            total_cost = sum(route_costs) if route_costs else 0
            max_route_cost = max(route_costs) if route_costs else 0
            avg_route_cost = total_cost / len(route_costs) if route_costs else 0
            num_routes = len(valid_routes)

            # Determine overall status
            if parsing_errors:
                status = "ERROR"
            elif is_feasible:
                status = "FEASIBLE"
            else:
                status = "INFEASIBLE"

            return {
                "test_case": test_file,
                "solution": solution_file,
                "status": status,
                "feasible": is_feasible and not parsing_errors,
                "execution_time": execution_time,
                "n": n,
                "m": m,
                "k": k,
                "total_cost": total_cost,
                "max_route_cost": max_route_cost,
                "avg_route_cost": avg_route_cost,
                "num_routes": num_routes,
                "max_route_length": max_route_length,
                "feasibility_issues": all_issues,
                "error": "; ".join(all_issues) if all_issues else "",
            }

        except Exception as e:
            return {
                "test_case": test_file,
                "solution": solution_file,
                "status": "ERROR",
                "feasible": False,
                "error": f"Processing error: {str(e)}",
                "execution_time": execution_time,
                "n": n,
                "m": m,
                "k": k,
                "total_cost": 0,
                "max_route_cost": 0,
                "avg_route_cost": 0,
                "num_routes": 0,
                "max_route_length": 0,
                "feasibility_issues": [f"Processing error: {str(e)}"],
                "output": result[:500] + "..." if len(result) > 500 else result,
            }

    def run_all_tests(self):
        """Run all test cases for all solutions"""
        test_files = self.find_test_cases()
        if not test_files:
            return

        print(f"Found {len(test_files)} test cases")
        print(
            f"Testing {len(self.solution_files)} solutions: {', '.join(self.solution_files)}"
        )
        print("=" * 100)

        total_tests = len(test_files) * len(self.solution_files)
        current_test = 0

        for test_file in test_files:
            print(f"\n📁 Testing {test_file}:")

            # Read test case once
            try:
                with open(test_file, "r") as f:
                    test_input = f.read().strip()

                lines = test_input.strip().split("\n")
                n, m, k = map(int, lines[0].split())
                distance_matrix = self.parse_distance_matrix(test_input)

            except Exception as e:
                print(f"❌ Error reading test case: {e}")
                # Add error results for all solutions
                for solution in self.solution_files:
                    error_result = {
                        "test_case": test_file,
                        "solution": solution,
                        "status": "ERROR",
                        "feasible": False,
                        "error": f"Test case read error: {str(e)}",
                        "execution_time": 0,
                        "n": 0,
                        "m": 0,
                        "k": 0,
                        "total_cost": 0,
                        "max_route_cost": 0,
                        "avg_route_cost": 0,
                        "num_routes": 0,
                        "max_route_length": 0,
                        "feasibility_issues": [],
                    }
                    self.results[solution].append(error_result)
                continue

            # Test each solution on this test case
            test_results = {}
            for solution in self.solution_files:
                current_test += 1
                print(f"  [{current_test}/{total_tests}] {solution}...", end=" ")

                result = self.run_single_solution_test(
                    solution, test_file, test_input, distance_matrix, n, m, k
                )
                self.results[solution].append(result)
                test_results[solution] = result

                # Print immediate feedback
                if result["feasible"]:
                    print(
                        f"✅ FEASIBLE ({result['execution_time']:.3f}s, Max Route: {result['max_route_cost']})"
                    )
                elif result["status"] in ["INFEASIBLE", "ERROR"]:
                    issues_text = (
                        f"{len(result['feasibility_issues'])} issues"
                        if result["feasibility_issues"]
                        else "parsing/execution error"
                    )
                    print(f"❌ {result['status']} - {issues_text}")
                else:
                    print(f"⚠️ {result['status']} - {result['error']}")

            # Show comparison for this test case
            self.show_test_comparison(test_file, test_results)

        self.generate_reports()

    def show_test_comparison(self, test_file, test_results):
        """Show quick comparison for a single test case"""
        feasible_results = {
            sol: res for sol, res in test_results.items() if res["feasible"]
        }

        if len(feasible_results) > 1:
            print("  🏆 Feasible Solutions Comparison:")
            # Sort by max route cost (optimization objective)
            sorted_results = sorted(
                feasible_results.items(), key=lambda x: x[1]["max_route_cost"]
            )

            for i, (solution, result) in enumerate(sorted_results):
                rank = (
                    "🥇"
                    if i == 0
                    else "🥈"
                    if i == 1
                    else "🥉"
                    if i == 2
                    else f"{i + 1}."
                )
                print(
                    f"    {rank} {solution}: Max Route = {result['max_route_cost']}, Time = {result['execution_time']:.3f}s"
                )
        elif len(feasible_results) == 1:
            solution, result = next(iter(feasible_results.items()))
            print(f"  ✅ Only {solution} produced a feasible solution")
        else:
            print("  ❌ No feasible solutions found for this test case")

    def generate_reports(self):
        """Generate all reports"""
        self.generate_detailed_csv()
        self.generate_comparison_csv()
        self.generate_text_report()
        self.generate_results_visualization()

    def generate_detailed_csv(self):
        """Generate detailed CSV with all results"""
        csv_headers = [
            "Test Case",
            "Solution",
            "Status",
            "Feasible",
            "N",
            "M",
            "K",
            "Execution Time (s)",
            "Total Cost",
            "Max Route Cost",
            "Avg Route Cost",
            "Number of Routes",
            "Max Route Length",
            "Input Size",
            "Feasibility Issues Count",
            "Error Message",
            "Feasibility Issues Details",
        ]

        with open(self.csv_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)

            for solution in self.solution_files:
                for result in self.results[solution]:
                    row = [
                        result["test_case"],
                        result["solution"],
                        result["status"],
                        result["feasible"],
                        result["n"],
                        result["m"],
                        result["k"],
                        f"{result['execution_time']:.4f}",
                        result["total_cost"],
                        result["max_route_cost"],
                        f"{result['avg_route_cost']:.2f}",
                        result["num_routes"],
                        result["max_route_length"],
                        result["n"] + result["m"],
                        len(result.get("feasibility_issues", [])),
                        result.get("error", ""),
                        "; ".join(result.get("feasibility_issues", [])),
                    ]
                    writer.writerow(row)

        print(f"\n📊 Detailed CSV saved to: {self.csv_file}")

    def generate_comparison_csv(self):
        """Generate comparison CSV showing algorithms side by side"""
        test_files = list(
            set(
                result["test_case"]
                for solution_results in self.results.values()
                for result in solution_results
            )
        )
        test_files.sort()

        # Create headers
        headers = ["Test Case", "N", "M", "K", "Input Size"]
        for solution in self.solution_files:
            sol_name = solution.replace(".py", "").title()
            headers.extend(
                [
                    f"{sol_name} Feasible",
                    f"{sol_name} Status",
                    f"{sol_name} Time (s)",
                    f"{sol_name} Max Route Cost",
                    f"{sol_name} Total Cost",
                    f"{sol_name} Issues Count",
                ]
            )
        headers.append("Best Feasible Algorithm")
        headers.append("Feasible Count")

        with open(self.comparison_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for test_file in test_files:
                # Get results for this test case from all solutions
                test_results = {}
                for solution in self.solution_files:
                    for result in self.results[solution]:
                        if result["test_case"] == test_file:
                            test_results[solution] = result
                            break

                if not test_results:
                    continue

                # Get basic info from first available result
                first_result = next(iter(test_results.values()))
                row = [
                    test_file,
                    first_result["n"],
                    first_result["m"],
                    first_result["k"],
                    first_result["n"] + first_result["m"],
                ]

                # Add results for each solution
                feasible_results = {}
                for solution in self.solution_files:
                    result = test_results.get(solution, {})
                    row.extend(
                        [
                            result.get("feasible", False),
                            result.get("status", "N/A"),
                            f"{result.get('execution_time', 0):.4f}",
                            result.get("max_route_cost", "N/A"),
                            result.get("total_cost", "N/A"),
                            len(result.get("feasibility_issues", [])),
                        ]
                    )

                    if result.get("feasible", False):
                        feasible_results[solution] = result

                # Determine best feasible algorithm
                if feasible_results:
                    best_solution = min(
                        feasible_results.items(), key=lambda x: x[1]["max_route_cost"]
                    )
                    row.append(best_solution[0].replace(".py", "").title())
                    row.append(len(feasible_results))
                else:
                    row.extend(["None", 0])

                writer.writerow(row)

        print(f"📈 Comparison CSV saved to: {self.comparison_csv}")

    def generate_results_visualization(self):
        """Generate visualization of the results"""
        try:
            # Set style
            plt.style.use("default")
            sns.set_palette("husl")

            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))

            # Prepare data
            all_data = []
            for solution in self.solution_files:
                for result in self.results[solution]:
                    all_data.append(
                        {
                            "Solution": solution.replace(".py", "").title(),
                            "Test_Case": result["test_case"],
                            "Status": result["status"],
                            "Feasible": result["feasible"],
                            "Execution_Time": result["execution_time"],
                            "Max_Route_Cost": result["max_route_cost"]
                            if result["feasible"]
                            else None,
                            "Total_Cost": result["total_cost"]
                            if result["feasible"]
                            else None,
                            "N": result["n"],
                            "M": result["m"],
                            "Input_Size": result["n"] + result["m"],
                            "Issues_Count": len(result.get("feasibility_issues", [])),
                        }
                    )

            df = pd.DataFrame(all_data)

            # 1. Feasibility Rate by Algorithm
            ax1 = plt.subplot(2, 3, 1)
            feasibility_rates = df.groupby("Solution")["Feasible"].agg(["sum", "count"])
            feasibility_rates["rate"] = (
                feasibility_rates["sum"] / feasibility_rates["count"]
            ) * 100

            bars = ax1.bar(
                feasibility_rates.index,
                feasibility_rates["rate"],
                color=["#2ecc71", "#e74c3c", "#3498db", "#f39c12"][
                    : len(feasibility_rates)
                ],
            )
            ax1.set_title(
                "Feasibility Rate by Algorithm", fontsize=14, fontweight="bold"
            )
            ax1.set_ylabel("Feasibility Rate (%)")
            ax1.set_ylim(0, 100)

            # Add value labels on bars
            for bar, rate in zip(bars, feasibility_rates["rate"]):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plt.xticks(rotation=45)

            # 2. Status Distribution Heatmap
            ax2 = plt.subplot(2, 3, 2)
            status_pivot = df.pivot_table(
                index="Solution",
                columns="Status",
                values="Test_Case",
                aggfunc="count",
                fill_value=0,
            )

            sns.heatmap(status_pivot, annot=True, fmt="d", cmap="RdYlBu_r", ax=ax2)
            ax2.set_title("Test Results Distribution", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Status")
            ax2.set_ylabel("Algorithm")

            # 3. Execution Time Comparison (for feasible solutions only)
            ax3 = plt.subplot(2, 3, 3)
            feasible_df = df[df["Feasible"] == True]
            if not feasible_df.empty:
                box_data = [
                    feasible_df[feasible_df["Solution"] == sol]["Execution_Time"].values
                    for sol in feasible_df["Solution"].unique()
                ]
                labels = feasible_df["Solution"].unique()

                bp = ax3.boxplot(box_data, labels=labels, patch_artist=True)
                colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
                for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax3.set_title(
                    "Execution Time Distribution\n(Feasible Solutions Only)",
                    fontsize=14,
                    fontweight="bold",
                )
                ax3.set_ylabel("Execution Time (seconds)")
                plt.xticks(rotation=45)
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No Feasible Solutions",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=12,
                )
                ax3.set_title(
                    "Execution Time Distribution", fontsize=14, fontweight="bold"
                )

            # 4. Solution Quality vs Problem Size
            ax4 = plt.subplot(2, 3, 4)
            if not feasible_df.empty:
                for solution in feasible_df["Solution"].unique():
                    sol_data = feasible_df[feasible_df["Solution"] == solution]
                    ax4.scatter(
                        sol_data["Input_Size"],
                        sol_data["Max_Route_Cost"],
                        label=solution,
                        alpha=0.7,
                        s=50,
                    )

                ax4.set_title(
                    "Solution Quality vs Problem Size", fontsize=14, fontweight="bold"
                )
                ax4.set_xlabel("Problem Size (N + M)")
                ax4.set_ylabel("Max Route Cost")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No Feasible Solutions",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                    fontsize=12,
                )
                ax4.set_title(
                    "Solution Quality vs Problem Size", fontsize=14, fontweight="bold"
                )

            # 5. Test Case Success Rate
            ax5 = plt.subplot(2, 3, 5)
            test_success = df.groupby("Test_Case")["Feasible"].sum().sort_values()

            bars = ax5.barh(
                range(len(test_success)),
                test_success.values,
                color=plt.cm.RdYlGn(test_success.values / len(self.solution_files)),
            )
            ax5.set_title(
                "Algorithms with Feasible Solutions per Test Case",
                fontsize=14,
                fontweight="bold",
            )
            ax5.set_xlabel("Number of Feasible Solutions")
            ax5.set_yticks(range(len(test_success)))
            ax5.set_yticklabels(
                [
                    tc.replace("test_case_", "").replace(".txt", "")
                    for tc in test_success.index
                ],
                fontsize=8,
            )

            # Add value labels
            for i, v in enumerate(test_success.values):
                ax5.text(v + 0.05, i, str(v), va="center", fontweight="bold")

            # 6. Summary Statistics Table
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis("off")

            # Calculate summary statistics
            summary_data = []
            for solution in self.solution_files:
                sol_results = [r for r in self.results[solution]]
                feasible_count = sum(1 for r in sol_results if r["feasible"])
                total_tests = len(sol_results)
                feasible_rate = (
                    (feasible_count / total_tests) * 100 if total_tests > 0 else 0
                )

                feasible_results = [r for r in sol_results if r["feasible"]]
                avg_time = (
                    np.mean([r["execution_time"] for r in feasible_results])
                    if feasible_results
                    else 0
                )
                avg_cost = (
                    np.mean([r["max_route_cost"] for r in feasible_results])
                    if feasible_results
                    else 0
                )

                summary_data.append(
                    [
                        solution.replace(".py", "").title(),
                        f"{feasible_count}/{total_tests}",
                        f"{feasible_rate:.1f}%",
                        f"{avg_time:.3f}s" if avg_time > 0 else "N/A",
                        f"{avg_cost:.1f}" if avg_cost > 0 else "N/A",
                    ]
                )

            # Create table
            table = ax6.table(
                cellText=summary_data,
                colLabels=["Algorithm", "Feasible", "Rate", "Avg Time", "Avg Max Cost"],
                cellLoc="center",
                loc="center",
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # Style the table
            for i in range(len(summary_data) + 1):
                for j in range(5):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor("#3498db")
                        cell.set_text_props(weight="bold", color="white")
                    else:
                        cell.set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

            ax6.set_title("Summary Statistics", fontsize=14, fontweight="bold", pad=20)

            # Overall title and layout
            fig.suptitle(
                "Multi-Algorithm Taxi Routing Feasibility Analysis",
                fontsize=18,
                fontweight="bold",
                y=0.98,
            )

            plt.tight_layout()
            plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)

            # Save the visualization
            image_file = f"feasibility_analysis_{self.timestamp}.png"
            plt.savefig(image_file, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            print(f"📊 Results visualization saved to: {image_file}")

        except Exception as e:
            print(f"⚠️ Could not generate visualization: {str(e)}")
            print("   Make sure matplotlib, pandas, and seaborn are installed:")
            print("   pip install matplotlib pandas seaborn")

    def generate_text_report(self):
        """Generate comprehensive text report"""
        print("\n" + "=" * 100)
        print("MULTI-ALGORITHM FEASIBILITY ANALYSIS REPORT")
        print("=" * 100)

        # Overall statistics
        for solution in self.solution_files:
            results = self.results[solution]
            feasible = sum(1 for r in results if r["feasible"])
            infeasible = sum(1 for r in results if r["status"] == "INFEASIBLE")
            parsing_errors = sum(1 for r in results if r["status"] == "PARSING_ERROR")
            other_errors = sum(
                1
                for r in results
                if r["status"] not in ["FEASIBLE", "INFEASIBLE", "PARSING_ERROR"]
            )

            print(f"\n🔧 {solution.upper()}:")
            print(
                f"  Tests: {len(results)} | Feasible: {feasible} | Infeasible: {infeasible} | Parsing Errors: {parsing_errors} | Other Errors: {other_errors}"
            )
            print(f"  Feasibility Rate: {feasible / len(results) * 100:.1f}%")

            feasible_tests = [r for r in results if r["feasible"]]
            if feasible_tests:
                avg_time = sum(r["execution_time"] for r in feasible_tests) / len(
                    feasible_tests
                )
                avg_max_cost = sum(r["max_route_cost"] for r in feasible_tests) / len(
                    feasible_tests
                )
                print(
                    f"  Avg Time (feasible): {avg_time:.4f}s | Avg Max Route Cost: {avg_max_cost:.1f}"
                )

        # Algorithm ranking
        print(f"\n🏆 ALGORITHM RANKING BY FEASIBILITY:")

        # Collect feasibility statistics
        feasibility_stats = []
        for solution in self.solution_files:
            results = self.results[solution]
            feasible_count = sum(1 for r in results if r["feasible"])
            feasible_rate = feasible_count / len(results) * 100 if results else 0

            feasible_tests = [r for r in results if r["feasible"]]
            avg_max_cost = (
                sum(r["max_route_cost"] for r in feasible_tests) / len(feasible_tests)
                if feasible_tests
                else float("inf")
            )
            avg_time = (
                sum(r["execution_time"] for r in feasible_tests) / len(feasible_tests)
                if feasible_tests
                else 0
            )

            feasibility_stats.append(
                (solution, feasible_rate, feasible_count, avg_max_cost, avg_time)
            )

        # Rank by feasibility rate first, then by solution quality
        feasibility_stats.sort(key=lambda x: (-x[1], x[3]))

        print("  By Feasibility Rate and Solution Quality:")
        for i, (solution, rate, count, avg_cost, avg_time) in enumerate(
            feasibility_stats, 1
        ):
            emoji = (
                "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"  {i}."
            )
            cost_str = f"{avg_cost:.1f}" if avg_cost != float("inf") else "N/A"
            print(
                f"    {emoji} {solution}: {rate:.1f}% feasible ({count} solutions), Avg Max Cost: {cost_str}"
            )

        print(f"\nFiles generated:")
        print(f"  📊 {self.csv_file} - Detailed feasibility results for all algorithms")
        print(f"  📈 {self.comparison_csv} - Side-by-side feasibility comparison table")


def main():
    print("=== Multi-Algorithm Taxi Routing Feasibility Test Runner ===")
    print("Evaluate feasibility of multiple solution algorithms!")

    # Define the solution files
    solution_files = [
        "dijkstra.py",
        "hill_climbing.py",
        "metaheuristic.py",
        "greedy.py",
    ]

    # Check which files exist
    existing_files = []
    missing_files = []

    for file in solution_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            missing_files.append(file)

    if not existing_files:
        print("❌ No solution files found!")
        print(
            "Expected files: dijkstra.py, hill_climbing.py, metaheuristic.py, greedy.py"
        )
        return

    if missing_files:
        print(f"⚠️ Missing files: {', '.join(missing_files)}")
        print(f"✅ Found files: {', '.join(existing_files)}")

        proceed = input("Continue with available files? (y/n): ").strip().lower()
        if proceed != "y":
            return
    else:
        print(f"✅ All solution files found: {', '.join(existing_files)}")

    # Create and run tests
    print(f"\nRunning feasibility tests for {len(existing_files)} algorithms...")
    runner = MultiSolutionFeasibilityRunner(existing_files)
    runner.run_all_tests()

    print("\n🎉 Multi-algorithm feasibility test run completed!")


if __name__ == "__main__":
    main()
