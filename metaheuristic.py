# PYTHON
import sys
import math
import random


def read_input():
    try:
        data = sys.stdin.read().strip().split()
        it = iter(data)
        N = int(next(it))
        M = int(next(it))
        K = int(next(it))
        q = [0] + [int(next(it)) for _ in range(M)]  # Parcel quantities (1-based)
        Q = [0] + [int(next(it)) for _ in range(K)]  # Taxi capacities (1-based)
        total_points = 2 * N + 2 * M + 1
        d = [[0] * total_points for _ in range(total_points)]
        for i in range(total_points):
            for j in range(total_points):
                d[i][j] = int(next(it))
        return N, M, K, q, Q, d
    except Exception as e:
        print(f"Input error: {e}", file=sys.stderr)
        raise


def calculate_route_distance(route, d):
    if len(route) < 2:
        return 0
    return sum(d[route[i]][route[i + 1]] for i in range(len(route) - 1))


def is_valid_route(route, N, M, q, Q, taxi_id):
    if route[0] != 0 or route[-1] != 0:
        return False

    for i in range(len(route) - 1):
        if 1 <= route[i] <= N:
            if route[i + 1] != route[i] + N + M:
                return False

    load = 0
    carried = set()
    for node in route:
        if N + 1 <= node <= N + M:  
            parcel_idx = node - N
            load += q[parcel_idx]
            carried.add(parcel_idx)
            if load > Q[taxi_id]:
                return False
        elif 2 * N + M + 1 <= node <= 2 * N + 2 * M:  
            parcel_idx = node - (2 * N + M)
            if parcel_idx not in carried:
                return False
            load -= q[parcel_idx]
            carried.remove(parcel_idx)

    pickups = set(node for node in route if 1 <= node <= N + M)
    dropoffs = set(node for node in route if N + M + 1 <= node <= 2 * N + 2 * M)
    for pu in pickups:
        dp = pu + N + M if pu <= N else pu + N + M
        if dp not in dropoffs:
            return False
    return True


def initial_solution(N, M, K, q, Q, d):
    
    taxis_route = [[] for _ in range(K + 1)]
    taxis_passengers = [[] for _ in range(K + 1)]

    for i in range(1, N + 1):
        k = (i - 1) % K + 1
        taxis_passengers[k].append(i)

    for k in range(1, K + 1):
        route = [0]
        unvisited = set(taxis_passengers[k])
        current = 0
        while unvisited:
            next_i = min(unvisited, key=lambda i: d[current][i])
            route.append(next_i)
            route.append(next_i + N + M)
            current = next_i + N + M
            unvisited.remove(next_i)
        route.append(0)
        taxis_route[k] = route

    parcels = [(q[i], i) for i in range(1, M + 1)]
    parcels.sort(reverse=True)  
    for _, parcel_idx in parcels:
        a = parcel_idx + N
        b = parcel_idx + 2 * N + M
        best_k = None
        best_route = None
        best_cost = float("inf")
        for k in range(1, K + 1):
            if q[parcel_idx] > Q[k]:
                continue
            route = taxis_route[k]
            L = len(route)
            for i in range(1, L):
                if 1 <= route[i - 1] <= N and route[i] == route[i - 1] + N + M:
                    continue
                for j in range(i, L):
                    if 1 <= route[j - 1] <= N and route[j] == route[j - 1] + N + M:
                        continue
                    new_route = route[:i] + [a] + route[i:j] + [b] + route[j:]
                    if is_valid_route(new_route, N, M, q, Q, k):
                        cost = calculate_route_distance(new_route, d)
                        if cost < best_cost:
                            best_cost = cost
                            best_k = k
                            best_route = new_route
        if best_k is not None:
            taxis_route[best_k] = best_route
        else:
            k = min(
                range(1, K + 1),
                key=lambda k: calculate_route_distance(taxis_route[k], d)
                if q[parcel_idx] <= Q[k]
                else float("inf"),
            )
            if q[parcel_idx] <= Q[k]:
                route = taxis_route[k]
                new_route = route[:1] + [a, b] + route[1:]
                if is_valid_route(new_route, N, M, q, Q, k):
                    taxis_route[k] = new_route

    return taxis_route


def get_max_route_length(routes, d):
    
    return max(calculate_route_distance(route, d) for route in routes[1:])


def neighbor_solution(routes, N, M, q, Q, d):
   
    routes = [r[:] for r in routes] 
    K = len(routes) - 1
    move_type = random.choice(["swap_passenger", "swap_parcel", "reorder"])

    if move_type == "swap_passenger" and N > 0:
        k1, k2 = random.sample(range(1, K + 1), 2)
        if taxis_passengers[k1] and taxis_passengers[k2]:
            p1 = random.choice(taxis_passengers[k1])
            p2 = random.choice(taxis_passengers[k2])
            route1 = routes[k1]
            route2 = routes[k2]
            new_route1 = [x for x in route1 if x not in [p1, p1 + N + M]]
            new_route2 = [x for x in route2 if x not in [p2, p2 + N + M]]
            i1 = random.randint(1, len(new_route1))
            i2 = random.randint(1, len(new_route2))
            new_route1 = new_route1[:i1] + [p2, p2 + N + M] + new_route1[i1:]
            new_route2 = new_route2[:i2] + [p1, p1 + N + M] + new_route2[i2:]
            if is_valid_route(new_route1, N, M, q, Q, k1) and is_valid_route(
                new_route2, N, M, q, Q, k2
            ):
                routes[k1] = new_route1
                routes[k2] = new_route2

    elif move_type == "swap_parcel" and M > 0:
        k1, k2 = random.sample(range(1, K + 1), 2)
        parcels1 = [x - N for x in routes[k1] if N + 1 <= x <= N + M]
        parcels2 = [x - N for x in routes[k2] if N + 1 <= x <= N + M]
        if parcels1 and parcels2:
            p1 = random.choice(parcels1)
            p2 = random.choice(parcels2)
            if q[p1] <= Q[k2] and q[p2] <= Q[k1]:
                route1 = routes[k1]
                route2 = routes[k2]
                new_route1 = [x for x in route1 if x not in [p1 + N, p1 + 2 * N + M]]
                new_route2 = [x for x in route2 if x not in [p2 + N, p2 + 2 * N + M]]
                i1 = random.randint(1, len(new_route1))
                j1 = random.randint(i1, len(new_route1))
                i2 = random.randint(1, len(new_route2))
                j2 = random.randint(i2, len(new_route2))
                new_route1 = (
                    new_route1[:i1]
                    + [p2 + N]
                    + new_route1[i1:j1]
                    + [p2 + 2 * N + M]
                    + new_route1[j1:]
                )
                new_route2 = (
                    new_route2[:i2]
                    + [p1 + N]
                    + new_route2[i2:j2]
                    + [p1 + 2 * N + M]
                    + new_route2[j2:]
                )
                if is_valid_route(new_route1, N, M, q, Q, k1) and is_valid_route(
                    new_route2, N, M, q, Q, k2
                ):
                    routes[k1] = new_route1
                    routes[k2] = new_route2

    elif move_type == "reorder":
        k = random.randint(1, K)
        route = routes[k]
        if len(route) > 3: 
            i, j = sorted(random.sample(range(1, len(route) - 1), 2))
            if all(
                not (1 <= route[x] <= N and route[x + 1] == route[x] + N + M)
                for x in range(i, j)
            ):
                new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                if is_valid_route(new_route, N, M, q, Q, k):
                    routes[k] = new_route

    return routes


def simulated_annealing(N, M, K, q, Q, d):

    routes = initial_solution(N, M, K, q, Q, d)
    current_cost = get_max_route_length(routes, d)
    best_routes = [r[:] for r in routes]
    best_cost = current_cost

    T = 1000.0  
    T_min = 0.1
    alpha = 0.995  
    max_iterations = 10000

    for _ in range(max_iterations):
        if T < T_min:
            break
        new_routes = neighbor_solution(routes, N, M, q, Q, d)
        new_cost = get_max_route_length(new_routes, d)
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < math.exp(-delta / T):
            routes = new_routes
            current_cost = new_cost
            if current_cost < best_cost:
                best_routes = [r[:] for r in routes]
                best_cost = current_cost
        T *= alpha

    return best_routes


def main():
    try:
        N, M, K, q, Q, d = read_input()
        global taxis_passengers  
        taxis_passengers = [[] for _ in range(K + 1)]
        for i in range(1, N + 1):
            k = (i - 1) % K + 1
            taxis_passengers[k].append(i)

        routes = simulated_annealing(N, M, K, q, Q, d)

        print(K)
        for k in range(1, K + 1):
            route = routes[k]
            print(len(route))
            print(" ".join(map(str, route)))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
