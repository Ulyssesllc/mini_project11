import random
import os


def generate_case(
    N, M, K, q_range=(1, 100), Q_range=(1, 200), d_range=(1, 20), seed=None
):
    if seed is not None:
        random.seed(seed)
    q = [random.randint(*q_range) for _ in range(M)]
    Q = [random.randint(*Q_range) for _ in range(K)]
    total_points = 2 * N + 2 * M + 1
    d = [[0] * total_points for _ in range(total_points)]
    for i in range(total_points):
        for j in range(total_points):
            if i == j:
                d[i][j] = 0
            else:
                d[i][j] = random.randint(*d_range)
    # Optionally, make d symmetric (undirected distances)
    for i in range(total_points):
        for j in range(i + 1, total_points):
            d[j][i] = d[i][j]
    return N, M, K, q, Q, d


def write_case(filename, N, M, K, q, Q, d):
    import csv

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "M", "K"])
        writer.writerow([N, M, K])
        writer.writerow(["q"] + q)
        writer.writerow(["Q"] + Q)
        writer.writerow(["d_matrix"])
        for row in d:
            writer.writerow(row)


def generate_and_save(folder, N, M, K, idx=1, **kwargs):
    os.makedirs(folder, exist_ok=True)
    N, M, K, q, Q, d = generate_case(N, M, K, **kwargs)
    filename = os.path.join(folder, f"test_{N}_{M}_{K}_{idx}.csv")
    write_case(filename, N, M, K, q, Q, d)
    print(f"Generated: {filename}")


if __name__ == "__main__":
    # Generate 10 large test cases with increasing size
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_cases_folder = os.path.join(base_dir, "test_cases")
    test_params = [
        (100, 100, 10),
        (120, 120, 12),
        (150, 130, 15),
        (180, 160, 18),
        (200, 200, 20),
        (250, 220, 25),
        (300, 250, 30),
        (350, 300, 35),
        (400, 350, 40),
        (500, 500, 50),
    ]
    for idx, (N, M, K) in enumerate(test_params, 1):
        generate_and_save(test_cases_folder, N, M, K, idx=idx, seed=100 + idx)
