import time
import numpy as np
import xarray as xr
import linopy
from data import (
    N_PRODUCTS, N_DAYS, WINDOW, SLIDE,
    demand, max_inventory, initial_inventory,
    production_cost, inventory_cost,
)

products = np.arange(N_PRODUCTS)
days = np.arange(WINDOW)
inner_days = np.arange(1, WINDOW)

da_max_inv = xr.DataArray(max_inventory, dims=["product"])
da_prod_cost = xr.DataArray(production_cost, dims=["product"])
da_inv_cost = xr.DataArray(inventory_cost, dims=["product"])

SOLVER_OPTS = {"solver_name": "highs", "io_api": "direct", "output_flag": False, "log_to_console": False}


def build_model(inventory_init: np.ndarray, start: int) -> linopy.Model:
    m = linopy.Model()
    coords = {"product": products, "day": days}

    prod = m.add_variables(lower=0, coords=coords, name="prod")
    inv = m.add_variables(lower=0, upper=da_max_inv, coords=coords, name="inv")

    da_demand = xr.DataArray(demand[:, start:start + WINDOW], dims=["product", "day"])
    da_init = xr.DataArray(inventory_init, dims=["product"])

    # day=0: 初期在庫を使ったバランス制約
    m.add_constraints(
        inv.isel(day=0) - prod.isel(day=0) == da_init - da_demand.isel(day=0),
        name="balance_0",
    )
    # day=1..WINDOW-1: 前日在庫を使ったバランス制約（day次元を一括追加）
    inv_cur  = inv.isel(day=slice(1, None)).assign_coords(day=inner_days)
    inv_prev = inv.isel(day=slice(0, -1)).assign_coords(day=inner_days)
    prod_cur = prod.isel(day=slice(1, None)).assign_coords(day=inner_days)
    m.add_constraints(
        inv_cur - inv_prev - prod_cur == -da_demand.isel(day=slice(1, None)).assign_coords(day=inner_days),
        name="balance_inner",
    )

    m.add_objective((da_prod_cost * prod + da_inv_cost * inv).sum())
    return m


def update_model(m: linopy.Model, inventory_init: np.ndarray, start: int) -> None:
    da_demand = xr.DataArray(demand[:, start:start + WINDOW], dims=["product", "day"])
    da_init = xr.DataArray(inventory_init, dims=["product"])

    m.constraints["balance_0"].rhs = (da_init - da_demand.isel(day=0)).values
    m.constraints["balance_inner"].rhs = (-da_demand.isel(day=slice(1, None))).values


def run_rolling_horizon_linopy() -> dict:
    inventory = initial_inventory.copy()
    objectives = []
    build_times = []
    solve_times = []

    t0 = time.perf_counter()
    m = build_model(inventory, start=0)
    initial_build_time = time.perf_counter() - t0

    for i, start in enumerate(range(0, N_DAYS - WINDOW + 1, SLIDE)):
        t0 = time.perf_counter()
        if i > 0:
            update_model(m, inventory, start)
        t1 = time.perf_counter()

        m.solve(**SOLVER_OPTS, progress=False)
        t2 = time.perf_counter()

        inv_vals = m.solution["inv"].isel(day=SLIDE - 1).values
        inventory = np.maximum(inv_vals, 0)
        objectives.append(float(m.objective.value))
        build_times.append(initial_build_time if i == 0 else t1 - t0)
        solve_times.append(t2 - t1)

    return {
        "objectives": objectives,
        "build_times": build_times,
        "solve_times": solve_times,
    }


if __name__ == "__main__":
    t_start = time.perf_counter()
    result = run_rolling_horizon_linopy()
    t_end = time.perf_counter()
    bt = result["build_times"]
    st = result["solve_times"]
    n = len(result["objectives"])
    print(f"linopy (HiGHS direct): {n} windows, total {t_end - t_start:.2f}s")
    print(f"  build/update avg: {np.mean(bt)*1000:.1f}ms  solve avg: {np.mean(st)*1000:.1f}ms")
    print(f"  (first build: {bt[0]*1000:.1f}ms, subsequent update avg: {np.mean(bt[1:])*1000:.1f}ms)")
