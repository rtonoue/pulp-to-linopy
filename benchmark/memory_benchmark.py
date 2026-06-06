import gc
import tracemalloc
import numpy as np
import xarray as xr
import pulp
import linopy
from data import (
    N_PRODUCTS, WINDOW, SLIDE,
    demand, max_inventory, initial_inventory,
    production_cost, inventory_cost,
)

products_np = np.arange(N_PRODUCTS)
days_np = np.arange(WINDOW)
inner_days = np.arange(1, WINDOW)
da_max_inv = xr.DataArray(max_inventory, dims=["product"])
da_prod_cost = xr.DataArray(production_cost, dims=["product"])
da_inv_cost = xr.DataArray(inventory_cost, dims=["product"])


def measure_peak_mb(func) -> float:
    gc.collect()
    tracemalloc.start()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def build_pulp(start: int = 0):
    products = list(range(N_PRODUCTS))
    days = list(range(WINDOW))
    inventory = initial_inventory.copy()

    prob = pulp.LpProblem("mem_test", pulp.LpMinimize)
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
            prev = inventory[p] if t == 0 else inv[p, t - 1]
            prob += inv[p, t] == prev + prod[p, t] - demand[p, start + t]
    return prob


def build_linopy(start: int = 0):
    inventory = initial_inventory.copy()
    coords = {"product": products_np, "day": days_np}
    da_demand = xr.DataArray(demand[:, start:start + WINDOW], dims=["product", "day"])
    da_init = xr.DataArray(inventory, dims=["product"])

    m = linopy.Model()
    prod = m.add_variables(lower=0, coords=coords, name="prod")
    inv = m.add_variables(lower=0, upper=da_max_inv, coords=coords, name="inv")
    m.add_constraints(
        inv.isel(day=0) - prod.isel(day=0) == da_init - da_demand.isel(day=0),
        name="balance_0",
    )
    inv_cur  = inv.isel(day=slice(1, None)).assign_coords(day=inner_days)
    inv_prev = inv.isel(day=slice(0, -1)).assign_coords(day=inner_days)
    prod_cur = prod.isel(day=slice(1, None)).assign_coords(day=inner_days)
    m.add_constraints(
        inv_cur - inv_prev - prod_cur
        == -da_demand.isel(day=slice(1, None)).assign_coords(day=inner_days),
        name="balance_inner",
    )
    m.add_objective((da_prod_cost * prod + da_inv_cost * inv).sum())
    return m


N_TRIALS = 5

print(f"=== Memory Benchmark (N_PRODUCTS={N_PRODUCTS}, WINDOW={WINDOW}) ===\n")
print(f"{'':30s} {'peak MB':>10s}")
print("-" * 42)

pulp_peaks = [measure_peak_mb(build_pulp) for _ in range(N_TRIALS)]
linopy_peaks = [measure_peak_mb(build_linopy) for _ in range(N_TRIALS)]

p_med = np.median(pulp_peaks)
l_med = np.median(linopy_peaks)

print(f"{'PuLP (model build)':30s} {p_med:>10.1f}")
print(f"{'linopy (model build)':30s} {l_med:>10.1f}")
print(f"{'ratio':30s} {p_med/l_med:>10.1f}x")
