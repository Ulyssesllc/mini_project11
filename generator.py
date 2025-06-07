import random
import os
import math


def generate_distance_matrix(num_points):
    """Generate a symmetric distance matrix using Euclidean distances"""
    # Generate random coordinates for each point
    coordinates = []
    for i in range(num_points):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        coordinates.append((x, y))

    # Create distance matrix
    distance_matrix = []
    for i in range(num_points):
        row = []
        for j in range(num_points):
            if i == j:
                row.append(0)
            else:
                # Euclidean distance, rounded to integer
                dist = math.sqrt(
                    (coordinates[i][0] - coordinates[j][0]) ** 2
                    + (coordinates[i][1] - coordinates[j][1]) ** 2
                )
                row.append(int(dist))
        distance_matrix.append(row)

    return distance_matrix


def generate_test_case(n, m, k, case_name):
    """Generate a single test case"""
    total_points = 2 * n + 2 * m + 1  # depot + pickup/dropoff points

    # Generate parcel quantities (1 to 100)
    parcel_quantities = [random.randint(1, 100) for _ in range(m)]

    # Generate taxi capacities (ensuring they can handle at least some parcels)
    min_capacity = max(parcel_quantities) if m > 0 else 50
    max_capacity = min(200, min_capacity + 100)
    taxi_capacities = [random.randint(min_capacity, max_capacity) for _ in range(k)]

    # Generate distance matrix
    distance_matrix = generate_distance_matrix(total_points)

    # Create the test case content
    content = []
    content.append(f"{n} {m} {k}")

    if m > 0:
        content.append(" ".join(map(str, parcel_quantities)))
    else:
        content.append("")

    content.append(" ".join(map(str, taxi_capacities)))

    # Add distance matrix
    for row in distance_matrix:
        content.append(" ".join(map(str, row)))

    # Write to file
    filename = f"test_case_{case_name}.txt"
    with open(filename, "w") as f:
        f.write("\n".join(content))

    print(f"Generated {filename}")
    return filename


def generate_all_test_cases():
    """Generate multiple test cases with different complexities"""
    test_cases = [
        # Small test cases
        (2, 2, 1, "test1"),
        (3, 3, 2, "test2"),
        (5, 4, 2, "test3"),
        # Medium test cases
        (10, 8, 3, "test4"),
        (15, 12, 4, "test5"),
        (20, 15, 5, "test6"),
        # Large test cases
        (50, 40, 10, "test7"),
        (100, 80, 15, "test8"),
        (200, 150, 20, "test9"),
        # Edge cases
        (1, 1, 1, "test10"),
        (500, 500, 100, "test11"),
        (100, 0, 5, "test12"),
        (0, 100, 5, "test13"),
        # Unbalanced cases
        (50, 5, 3, "test14"),
        (5, 50, 3, "test15"),
        (20, 20, 1, "test16"),
        (10, 10, 20, "test17"),
    ]

    generated_files = []

    for n, m, k, name in test_cases:
        try:
            filename = generate_test_case(n, m, k, name)
            generated_files.append(filename)
        except Exception as e:
            print(f"Error generating {name}: {e}")

    return generated_files


def generate_custom_test_case():
    """Generate a custom test case with user input"""
    print("\n=== Custom Test Case Generator ===")
    try:
        n = int(input("Enter number of passengers (N): "))
        m = int(input("Enter number of parcels (M): "))
        k = int(input("Enter number of taxis (K): "))
        name = input("Enter test case name: ")

        if n < 0 or m < 0 or k < 1:
            print("Invalid input! N, M must be >= 0, K must be >= 1")
            return None

        if n > 500 or m > 500 or k > 100:
            print("Warning: Large values may take time to generate")

        return generate_test_case(n, m, k, name)
    except ValueError:
        print("Invalid input! Please enter integers only.")
        return None


def main():
    """Main function to run the test case generator"""
    print("=== Taxi Routing Problem Test Case Generator ===")
    print("This script will generate test cases and save them as .txt files")
    print("in the current directory.\n")

    while True:
        print("Options:")
        print("1. Generate predefined test cases (18 cases)")
        print("2. Generate custom test case")
        print("3. Generate single random test case")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            print("\nGenerating predefined test cases...")
            files = generate_all_test_cases()
            print(f"\nGenerated {len(files)} test cases:")
            for f in files:
                print(f"  - {f}")

        elif choice == "2":
            file = generate_custom_test_case()
            if file:
                print(f"Generated custom test case: {file}")

        elif choice == "3":
            n = random.randint(1, 50)
            m = random.randint(1, 50)
            k = random.randint(1, 10)
            file = generate_test_case(n, m, k, "random")
            print(f"Generated random test case: {file}")
            print(f"  N={n}, M={m}, K={k}")

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice! Please enter 1, 2, 3, or 4.")

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
