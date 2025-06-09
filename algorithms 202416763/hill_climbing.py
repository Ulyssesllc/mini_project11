# PYTHON
import random
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

    n_points = 2 * N + 2 * M + 1
    d = []
    for i in range(n_points):
        row = []
        for j in range(n_points):
            row.append(int(next(it)))
        d.append(row)

    passengers = []
    for i in range(1, N + 1):
        pickup = i
        dropoff = i + N + M
        passengers.append(("passenger", i, pickup, dropoff))

    parcels = []
    for j in range(1, M + 1):
        parcel_id = j
        pickup = j + N
        delivery = j + 2 * N + M
        parcels.append(("parcel", parcel_id, q[j - 1], pickup, delivery))

    pass_assign = [random.randint(0, K - 1) for _ in range(N)]
    parc_assign = [random.randint(0, K - 1) for _ in range(M)]

    def greedy_scheduler(events, capacity, dist_mat):
        if not events:
            return 0, [0, 0]

        unscheduled_set = set()
        for ev in events:
            unscheduled_set.add(ev)

        current_point = 0
        current_load = 0
        picked_parcels = set()
        total_distance = 0
        points = [0]

        available = []
        for ev in unscheduled_set:
            if ev[0] == "passenger":
                available.append(ev)
            elif ev[0] == "pickup":
                if current_load + ev[3] <= capacity:
                    available.append(ev)
            elif ev[0] == "delivery":
                if ev[1] in picked_parcels:
                    available.append(ev)

        while available:
            min_d = 10**18
            next_ev = None
            for ev in available:
                if ev[0] == "passenger":
                    d = dist_mat[current_point][ev[2]]
                else:
                    d = dist_mat[current_point][ev[2]]
                if d < min_d:
                    min_d = d
                    next_ev = ev

            if next_ev is None:
                break

            unscheduled_set.remove(next_ev)
            if next_ev[0] == "passenger":
                total_distance += dist_mat[current_point][next_ev[2]]
                total_distance += dist_mat[next_ev[2]][next_ev[3]]
                points.append(next_ev[2])
                points.append(next_ev[3])
                current_point = next_ev[3]
            elif next_ev[0] == "pickup":
                total_distance += dist_mat[current_point][next_ev[2]]
                points.append(next_ev[2])
                current_point = next_ev[2]
                current_load += next_ev[3]
                picked_parcels.add(next_ev[1])
            elif next_ev[0] == "delivery":
                total_distance += dist_mat[current_point][next_ev[2]]
                points.append(next_ev[2])
                current_point = next_ev[2]
                current_load += next_ev[3]
                picked_parcels.discard(next_ev[1])

            available = []
            for ev in unscheduled_set:
                if ev[0] == "passenger":
                    available.append(ev)
                elif ev[0] == "pickup":
                    if current_load + ev[3] <= capacity:
                        available.append(ev)
                elif ev[0] == "delivery":
                    if ev[1] in picked_parcels:
                        available.append(ev)

        total_distance += dist_mat[current_point][0]
        points.append(0)
        return total_distance, points

    def compute_routes(pass_assign, parc_assign):
        events_per_taxi = [[] for _ in range(K)]
        for i in range(N):
            taxi = pass_assign[i]
            ev = passengers[i]
            events_per_taxi[taxi].append(("passenger", ev[1], ev[2], ev[3]))
        for j in range(M):
            taxi = parc_assign[j]
            pcl = parcels[j]
            events_per_taxi[taxi].append(("pickup", pcl[1], pcl[3], pcl[2]))
            events_per_taxi[taxi].append(("delivery", pcl[1], pcl[4], -pcl[2]))

        total_dists = [0] * K
        routes = [None] * K
        for taxi in range(K):
            dist_val, points_route = greedy_scheduler(events_per_taxi[taxi], Q[taxi], d)
            total_dists[taxi] = dist_val
            routes[taxi] = points_route
        return total_dists, routes, events_per_taxi

    total_dists, routes, events_per_taxi = compute_routes(pass_assign, parc_assign)
    current_objective = max(total_dists) if total_dists else 0
    current_routes = routes
    current_pass_assign = pass_assign
    current_parc_assign = parc_assign
    current_events_per_taxi = events_per_taxi

    n = 100
    for i in range(n):
        move_type = "reassign_task"
        if random.random() < 0.5:
            task_idx = random.randint(0, N - 1)
            old_taxi = current_pass_assign[task_idx]
            new_taxi = random.randint(0, K - 1)
            while new_taxi == old_taxi and K > 1:
                new_taxi = random.randint(0, K - 1)
            new_pass_assign = current_pass_assign[:]
            new_parc_assign = current_parc_assign[:]
            new_pass_assign[task_idx] = new_taxi
        else:
            task_idx = random.randint(0, M - 1)
            old_taxi = current_parc_assign[task_idx]
            new_taxi = random.randint(0, K - 1)
            while new_taxi == old_taxi and K > 1:
                new_taxi = random.randint(0, K - 1)
            new_pass_assign = current_pass_assign[:]
            new_parc_assign = current_parc_assign[:]
            new_parc_assign[task_idx] = new_taxi

        new_total_dists, new_routes, new_events_per_taxi = compute_routes(
            new_pass_assign, new_parc_assign
        )
        new_objective = max(new_total_dists) if new_total_dists else 0

        if new_objective < current_objective:
            current_objective = new_objective
            current_pass_assign = new_pass_assign
            current_parc_assign = new_parc_assign
            current_events_per_taxi = new_events_per_taxi
            current_routes = new_routes
            current_total_dists = new_total_dists

    print(K)
    for taxi in range(K):
        route = current_routes[taxi]
        print(len(route))
        print(" ".join(str(x) for x in route))


if __name__ == "__main__":
    main()
