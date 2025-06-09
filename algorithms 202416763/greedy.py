# PYTHON
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        return

    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    q = [int(next(it)) for _ in range(M)]
    Q = [int(next(it)) for _ in range(K)]

    total_points = 2 * N + 2 * M + 1
    d = []
    for i in range(total_points):
        row = []
        for j in range(total_points):
            val = int(next(it))
            row.append(val)
        d.append(row)

    point_info = [None] * total_points
    for i in range(1, N + 1):
        point_info[i] = ("passenger", "pickup", i)
        drop_index = i + N + M
        point_info[drop_index] = ("passenger", "dropoff", i)
    for i in range(1, M + 1):
        pickup_index = i + N
        drop_index = i + 2 * N + M
        point_info[pickup_index] = ("parcel", "pickup", i, q[i - 1])
        point_info[drop_index] = ("parcel", "dropoff", i, q[i - 1])

    requests = []
    for i in range(1, N + 1):
        requests.append(("passenger", i))
    for i in range(1, M + 1):
        requests.append(("parcel", i, q[i - 1]))

    sorted_requests = []
    for req in requests:
        if req[0] == "passenger":
            sorted_requests.append(req)
        else:
            sorted_requests.append(req)
    sorted_requests.sort(
        key=lambda x: (
            0 if x[0] == "passenger" else 1,
            -x[2] if x[0] == "parcel" else 0,
        )
    )

    routes = [[0, 0] for _ in range(K)]
    dists = [0] * K
    load_profiles = [[0, 0] for _ in range(K)]
    assigned_requests = [set() for _ in range(K)]
    beam_size = 5

    def simulate_load(route, taxi_index, new_request_id):
        load = 0
        new_load_profile = []
        assigned_set = assigned_requests[taxi_index] | {new_request_id}
        for point in route:
            info = point_info[point]
            if info is not None:
                if info[0] == "parcel" and info[2] in assigned_set:
                    if info[1] == "pickup":
                        load += info[3]
                    elif info[1] == "dropoff":
                        load -= info[3]
            new_load_profile.append(load)
        return new_load_profile

    for req in sorted_requests:
        current_max_route = max(dists) if dists else 0
        best_new_max_route = float("inf")
        best_taxi = None
        best_new_route = None
        best_new_dist = None
        best_new_load = None
        best_new_request_id = None

        if req[0] == "passenger":
            request_id = req[1]
            pickup = request_id
            dropoff = request_id + N + M
            for k in range(K):
                current_route = routes[k]
                current_dist = dists[k]
                current_load = load_profiles[k]
                n = len(current_route)
                for i in range(n - 1):
                    a = current_route[i]
                    b = current_route[i + 1]
                    added_dist = (
                        d[a][pickup] + d[pickup][dropoff] + d[dropoff][b] - d[a][b]
                    )
                    new_dist = current_dist + added_dist
                    new_route = (
                        current_route[: i + 1]
                        + [pickup, dropoff]
                        + current_route[i + 1 :]
                    )
                    new_load = (
                        current_load[: i + 1]
                        + [current_load[i], current_load[i]]
                        + current_load[i + 1 :]
                    )
                    new_max_route = max(current_max_route, new_dist)
                    if new_max_route < best_new_max_route:
                        best_new_max_route = new_max_route
                        best_taxi = k
                        best_new_route = new_route
                        best_new_dist = new_dist
                        best_new_load = new_load
                        best_new_request_id = request_id
        else:
            request_id = req[1]
            quantity = req[2]
            pickup = request_id + N
            dropoff = request_id + 2 * N + M
            for k in range(K):
                if Q[k] < quantity:
                    continue
                current_route = routes[k]
                current_dist = dists[k]
                current_load = load_profiles[k]
                n = len(current_route)
                if n < 2:
                    continue
                candidates_p = []
                for i in range(n - 1):
                    a = current_route[i]
                    b = current_route[i + 1]
                    cost_p = d[a][pickup] + d[pickup][b] - d[a][b]
                    new_route_p = (
                        current_route[: i + 1] + [pickup] + current_route[i + 1 :]
                    )
                    new_dist_p = current_dist + cost_p
                    new_load_p = current_load[: i + 1] + [current_load[i] + quantity]
                    for j in range(i + 1, n):
                        new_load_p.append(current_load[j] + quantity)
                    candidates_p.append(
                        (cost_p, i, new_route_p, new_dist_p, new_load_p)
                    )

                candidates_p.sort(key=lambda x: x[3])
                candidates_p = candidates_p[:beam_size]

                for cost_p, pos_p, new_route_p, new_dist_p, new_load_p in candidates_p:
                    candidates_d = []
                    n_p = len(new_route_p)
                    for j in range(pos_p + 1, n_p - 1):
                        a = new_route_p[j]
                        b = new_route_p[j + 1]
                        cost_d = d[a][dropoff] + d[dropoff][b] - d[a][b]
                        new_route = (
                            new_route_p[: j + 1] + [dropoff] + new_route_p[j + 1 :]
                        )
                        new_dist = new_dist_p + cost_d
                        new_load = new_load_p[: j + 1] + [new_load_p[j] - quantity]
                        for idx in range(j + 1, len(new_load_p)):
                            new_load.append(new_load_p[idx] - quantity)
                        max_load = max(new_load) if new_load else 0
                        if max_load <= Q[k]:
                            candidates_d.append(
                                (cost_d, j, new_route, new_dist, new_load)
                            )
                    candidates_d.sort(key=lambda x: x[3])
                    candidates_d = candidates_d[:beam_size]
                    for cost_d, j, new_route, new_dist, new_load in candidates_d:
                        new_max_route = max(current_max_route, new_dist)
                        if new_max_route < best_new_max_route:
                            best_new_max_route = new_max_route
                            best_taxi = k
                            best_new_route = new_route
                            best_new_dist = new_dist
                            best_new_load = new_load
                            best_new_request_id = request_id

        if best_taxi is None:
            fallback_success = False
            for k in range(K):
                if req[0] == "passenger":
                    request_id = req[1]
                    pickup = request_id
                    dropoff = request_id + N + M
                    current_route = routes[k]
                    current_dist = dists[k]
                    current_load = load_profiles[k]
                    n = len(current_route)
                    for i in range(n - 1):
                        a = current_route[i]
                        b = current_route[i + 1]
                        added_dist = (
                            d[a][pickup] + d[pickup][dropoff] + d[dropoff][b] - d[a][b]
                        )
                        new_dist = current_dist + added_dist
                        new_route = (
                            current_route[: i + 1]
                            + [pickup, dropoff]
                            + current_route[i + 1 :]
                        )
                        new_load = (
                            current_load[: i + 1]
                            + [current_load[i], current_load[i]]
                            + current_load[i + 1 :]
                        )
                        new_max_route = max(current_max_route, new_dist)
                        if not fallback_success or new_max_route < best_new_max_route:
                            fallback_success = True
                            best_new_max_route = new_max_route
                            best_taxi = k
                            best_new_route = new_route
                            best_new_dist = new_dist
                            best_new_load = new_load
                            best_new_request_id = request_id
                else:
                    request_id = req[1]
                    quantity = req[2]
                    pickup = request_id + N
                    dropoff = request_id + 2 * N + M
                    if Q[k] < quantity:
                        continue
                    current_route = routes[k]
                    current_dist = dists[k]
                    current_load = load_profiles[k]
                    n = len(current_route)
                    for i in range(n - 1):
                        a = current_route[i]
                        b = current_route[i + 1]
                        cost_p = d[a][pickup] + d[pickup][b] - d[a][b]
                        new_route_p = (
                            current_route[: i + 1] + [pickup] + current_route[i + 1 :]
                        )
                        new_dist_p = current_dist + cost_p
                        new_load_p = current_load[: i + 1] + [
                            current_load[i] + quantity
                        ]
                        for j in range(i + 1, n):
                            new_load_p.append(current_load[j] + quantity)
                        n_p = len(new_route_p)
                        for j in range(i + 1, n_p - 1):
                            a2 = new_route_p[j]
                            b2 = new_route_p[j + 1]
                            cost_d = d[a2][dropoff] + d[dropoff][b2] - d[a2][b2]
                            new_route = (
                                new_route_p[: j + 1] + [dropoff] + new_route_p[j + 1 :]
                            )
                            new_dist = new_dist_p + cost_d
                            new_load = new_load_p[: j + 1] + [new_load_p[j] - quantity]
                            for idx in range(j + 1, len(new_load_p)):
                                new_load.append(new_load_p[idx] - quantity)
                            max_load = max(new_load) if new_load else 0
                            if max_load <= Q[k]:
                                new_max_route = max(current_max_route, new_dist)
                                if (
                                    not fallback_success
                                    or new_max_route < best_new_max_route
                                ):
                                    fallback_success = True
                                    best_new_max_route = new_max_route
                                    best_taxi = k
                                    best_new_route = new_route
                                    best_new_dist = new_dist
                                    best_new_load = new_load
                                    best_new_request_id = request_id
            if not fallback_success:
                print("Failed to assign request: ", req)
                return

        k_assign = best_taxi
        routes[k_assign] = best_new_route
        dists[k_assign] = best_new_dist
        load_profiles[k_assign] = best_new_load
        assigned_requests[k_assign].add(best_new_request_id)

    print(K)
    for k in range(K):
        route = routes[k]
        print(len(route))
        print(" ".join(map(str, route)))


if __name__ == "__main__":
    main()
