import time
import pulp
import numpy as np
from data import (
    N_PRODUCTS, N_DAYS, WINDOW, SLIDE,
    demand, max_inventory, initial_inventory,
    production_cost, inventory_cost,
)


def solve_window_pulp(start: int, inventory_init: np.ndarray) -> tuple[float, np.ndarray, float, float]:
    end = min(start + WINDOW, N_DAYS)
    n_days = end - start
    days = list(range(n_days))
    products = list(range(N_PRODUCTS))

    t_build_start = time.perf_counter()

    prob = pulp.LpProblem("production_planning", pulp.LpMinimize)

    prod = {
        (p, t): pulp.LpVariable(f"prod_{p}_{t}", lowBound=0)
        for p in products for t in days
    }
    inv = {
        (p, t): pulp.LpVariable(f"inv_{p}_{t}", lowBound=0, upBound=max_inventory[p])
        for p in products for t in days
    }

    prob += pulp.LpAffineExpression(
        [(prod[p, t], production_cost[p]) for p in products for t in days]
        + [(inv[p, t], inventory_cost[p]) for p in products for t in days]
    )

    for p in products:
        for t in days:
            prev_inv = inventory_init[p] if t == 0 else inv[p, t - 1]
            prob += inv[p, t] == prev_inv + prod[p, t] - demand[p, start + t]

    t_build_end = time.perf_counter()

    solver = pulp.HiGHS(msg=False)
    prob.solve(solver)

    t_solve_end = time.perf_counter()

    inv_end = np.array([pulp.value(inv[p, SLIDE - 1]) for p in products])
    build_time = t_build_end - t_build_start
    solve_time = t_solve_end - t_build_end
    return pulp.value(prob.objective), inv_end, build_time, solve_time


def run_rolling_horizon_pulp() -> dict:
    inventory = initial_inventory.copy()
    objectives = []
    build_times = []
    solve_times = []

    for start in range(0, N_DAYS - WINDOW + 1, SLIDE):
        obj, inventory, bt, st = solve_window_pulp(start, inventory)
        objectives.append(obj)
        build_times.append(bt)
        solve_times.append(st)

    return {
        "objectives": objectives,
        "build_times": build_times,
        "solve_times": solve_times,
    }


if __name__ == "__main__":
    t_start = time.perf_counter()
    result = run_rolling_horizon_pulp()
    t_end = time.perf_counter()
    bt = result["build_times"]
    st = result["solve_times"]
    n = len(result["objectives"])
    print(f"PuLP (HiGHS): {n} windows, total {t_end - t_start:.2f}s")
    print(f"  build avg: {np.mean(bt)*1000:.1f}ms  solve avg: {np.mean(st)*1000:.1f}ms")
