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
    q = [0] * (M + 1)
    for i in range(1, M + 1):
        q[i] = int(next(it))
    Q = [0] * (K + 1)
    for i in range(1, K + 1):
        Q[i] = int(next(it))

    total_points = 2 * N + 2 * M + 1
    d = [[0] * total_points for _ in range(total_points)]
    for i in range(total_points):
        for j in range(total_points):
            d[i][j] = int(next(it))

    requests = []
    for i in range(1, N + 1):
        requests.append(("p", i))
    for i in range(1, M + 1):
        requests.append(("c", i))

    if N + M > 10 or K > 5:
        print(K)
        for k in range(K):
            print(2)
            print("0 0")
        return

    total_requests = len(requests)
    total_assignments = K**total_requests
    best_max_route_cost = 10**18
    best_routes_per_taxi = None

    def taxi_router(S, capacity, N, M, q, d):
        if len(S) == 0:
            return 0, [0, 0]
        reqs = sorted(S)
        R = len(reqs)
        info = []
        for req in reqs:
            if req[0] == "p":
                i = req[1]
                pickup = i
                drop = i + N + M
                qty = 0
            else:
                i = req[1]
                pickup = i + N
                drop = i + 2 * N + M
                qty = q[i]
            info.append((req[0], i, pickup, drop, qty))

        dp = {}
        parent = {}
        queue = []

        start_state = (0, 0, -1, 0)
        dp[start_state] = 0
        parent[start_state] = None
        queue.append((0, start_state))

        final_state = None
        final_cost = None

        while queue:
            min_cost = 10**18
            min_index = -1
            for idx, (cost_val, state) in enumerate(queue):
                if cost_val < min_cost:
                    min_cost = cost_val
                    min_index = idx
            if min_index == -1:
                break
            cost, state = queue.pop(min_index)
            loc, load, active, status = state

            if cost != dp.get(state, 10**18):
                continue

            done = True
            for i in range(R):
                st_i = (status >> (2 * i)) & 3
                if st_i != 2:
                    done = False
                    break
            if done and active == -1:
                final_state = state
                final_cost = cost + d[loc][0]
                break

            if active != -1:
                t, req_id, pickup_loc, drop_loc, qty_val = info[active]
                if t != "p":
                    continue
                new_loc = drop_loc
                new_cost = cost + d[loc][new_loc]
                new_load = load
                new_active = -1
                new_status = status
                new_status &= ~(3 << (2 * active))
                new_status |= 2 << (2 * active)
                new_state = (new_loc, new_load, new_active, new_status)
                if new_state not in dp or new_cost < dp.get(new_state, 10**18):
                    dp[new_state] = new_cost
                    parent[new_state] = state
                    queue.append((new_cost, new_state))
            else:
                for i in range(R):
                    st_i = (status >> (2 * i)) & 3
                    if st_i == 2:
                        continue
                    t, req_id, pickup_loc, drop_loc, qty_val = info[i]
                    if t == "p":
                        if st_i == 0:
                            new_loc = pickup_loc
                            new_cost = cost + d[loc][new_loc]
                            new_load = load
                            new_active = i
                            new_status = status
                            new_status &= ~(3 << (2 * i))
                            new_status |= 1 << (2 * i)
                            new_state = (new_loc, new_load, new_active, new_status)
                            if new_state not in dp or new_cost < dp.get(
                                new_state, 10**18
                            ):
                                dp[new_state] = new_cost
                                parent[new_state] = state
                                queue.append((new_cost, new_state))
                    else:
                        if st_i == 0:
                            if load + qty_val <= capacity:
                                new_loc = pickup_loc
                                new_cost = cost + d[loc][new_loc]
                                new_load = load + qty_val
                                new_active = -1
                                new_status = status
                                new_status &= ~(3 << (2 * i))
                                new_status |= 1 << (2 * i)
                                new_state = (new_loc, new_load, new_active, new_status)
                                if new_state not in dp or new_cost < dp.get(
                                    new_state, 10**18
                                ):
                                    dp[new_state] = new_cost
                                    parent[new_state] = state
                                    queue.append((new_cost, new_state))
                        elif st_i == 1:
                            new_loc = drop_loc
                            new_cost = cost + d[loc][new_loc]
                            new_load = load - qty_val
                            new_active = -1
                            new_status = status
                            new_status &= ~(3 << (2 * i))
                            new_status |= 2 << (2 * i)
                            new_state = (new_loc, new_load, new_active, new_status)
                            if new_state not in dp or new_cost < dp.get(
                                new_state, 10**18
                            ):
                                dp[new_state] = new_cost
                                parent[new_state] = state
                                queue.append((new_cost, new_state))

        if final_state is None:
            return 10**18, None

        path = []
        s = final_state
        while s is not None:
            path.append(s[0])
            s = parent.get(s, None)
        path.reverse()
        path.append(0)
        return final_cost, path

    for assign_index in range(total_assignments):
        assignment_vector = []
        x = assign_index
        for i in range(total_requests):
            r = x % K
            x = x // K
            assignment_vector.append(r)

        sets = [set() for _ in range(K)]
        for idx, req in enumerate(requests):
            taxi_id = assignment_vector[idx]
            sets[taxi_id].add(req)

        valid_assignment = True
        routes_per_taxi = [None] * K
        costs = [0] * K
        for k in range(K):
            S = sets[k]
            cap = Q[k + 1]
            cost_val, route_val = taxi_router(S, cap, N, M, q, d)
            if cost_val >= 10**18:
                valid_assignment = False
                break
            costs[k] = cost_val
            routes_per_taxi[k] = route_val

        if not valid_assignment:
            continue

        max_cost = max(costs)
        if max_cost < best_max_route_cost:
            best_max_route_cost = max_cost
            best_routes_per_taxi = routes_per_taxi

    if best_routes_per_taxi is None:
        print(K)
        for k in range(K):
            print(2)
            print("0 0")
    else:
        print(K)
        for k in range(K):
            route = best_routes_per_taxi[k]
            L = len(route)
            print(L)
            print(" ".join(str(x) for x in route))


if __name__ == "__main__":
    main()
