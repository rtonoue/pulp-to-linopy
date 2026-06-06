"""
m.solve() の総時間と HiGHS 内部ログの "HiGHS run time" を比較して
インターフェースのオーバーヘッドを計測する。
"""
import sys, os, re, time, tempfile, statistics
import numpy as np
import xarray as xr
import linopy
import pulp

sys.path.insert(0, ".")
from data import (
    N_PRODUCTS, N_DAYS, WINDOW, SLIDE,
    demand, max_inventory, initial_inventory,
    production_cost, inventory_cost,
)

products = np.arange(N_PRODUCTS)
days     = np.arange(WINDOW)
inner    = np.arange(1, WINDOW)
da_max_inv   = xr.DataArray(max_inventory,   dims=["product"])
da_prod_cost = xr.DataArray(production_cost, dims=["product"])
da_inv_cost  = xr.DataArray(inventory_cost,  dims=["product"])


def parse_highs_run_time(log_path: str) -> float | None:
    """HiGHS のログから 'HiGHS run time : X.XX' を抽出して秒で返す。"""
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"HiGHS run time\s*:\s*([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


# ─────────────────── linopy ───────────────────
def build_linopy(inventory_init, start):
    m = linopy.Model()
    coords = {"product": products, "day": days}
    x = m.add_variables(lower=0, coords=coords, name="x")
    s = m.add_variables(lower=0, upper=da_max_inv, coords=coords, name="s")
    da_d    = xr.DataArray(demand[:, start:start + WINDOW], dims=["product", "day"])
    da_init = xr.DataArray(inventory_init, dims=["product"])
    m.add_constraints(
        s.isel(day=0) - x.isel(day=0) == da_init - da_d.isel(day=0), name="b0"
    )
    sc = s.isel(day=slice(1, None)).assign_coords(day=inner)
    sp = s.isel(day=slice(0, -1)).assign_coords(day=inner)
    xc = x.isel(day=slice(1, None)).assign_coords(day=inner)
    m.add_constraints(
        sc - sp - xc == -da_d.isel(day=slice(1, None)).assign_coords(day=inner),
        name="bi",
    )
    m.add_objective((da_prod_cost * x + da_inv_cost * s).sum())
    return m


N = 15
linopy_total_times = []
linopy_highs_times = []
pulp_total_times   = []
pulp_highs_times   = []

inventory = initial_inventory.copy()
m = build_linopy(inventory, 0)

for i, start in enumerate(range(0, N_DAYS - WINDOW + 1, SLIDE)):
    if i >= N:
        break

    # ── linopy ──
    if i > 0:
        da_d    = xr.DataArray(demand[:, start:start + WINDOW], dims=["product", "day"])
        da_init = xr.DataArray(inventory, dims=["product"])
        m.constraints["b0"].rhs = (da_init - da_d.isel(day=0)).values
        m.constraints["bi"].rhs = (-da_d.isel(day=slice(1, None))).values

    log_fd, log_path = tempfile.mkstemp(suffix=".log")
    os.close(log_fd)
    t0 = time.perf_counter()
    m.solve(
        solver_name="highs",
        io_api="direct",
        output_flag=True,
        log_to_console=False,
        log_file=log_path,
        progress=False,
    )
    total_l = time.perf_counter() - t0
    highs_t = parse_highs_run_time(log_path) or 0.0
    linopy_total_times.append(total_l)
    linopy_highs_times.append(highs_t)
    inventory = np.maximum(m.solution["s"].isel(day=SLIDE - 1).values, 0)

    # ── PuLP ──
    inv2 = initial_inventory.copy() if i == 0 else inventory
    prob = pulp.LpProblem("p", pulp.LpMinimize)
    xp = {(p, t): pulp.LpVariable(f"x{p}_{t}", lowBound=0)
          for p in range(N_PRODUCTS) for t in range(WINDOW)}
    sp2 = {(p, t): pulp.LpVariable(f"s{p}_{t}", lowBound=0, upBound=max_inventory[p])
           for p in range(N_PRODUCTS) for t in range(WINDOW)}
    prob += pulp.LpAffineExpression(
        [(xp[p, t], production_cost[p]) for p in range(N_PRODUCTS) for t in range(WINDOW)]
        + [(sp2[p, t], inventory_cost[p]) for p in range(N_PRODUCTS) for t in range(WINDOW)]
    )
    for p in range(N_PRODUCTS):
        for t in range(WINDOW):
            prev = inv2[p] if t == 0 else sp2[p, t - 1]
            prob += sp2[p, t] == prev + xp[p, t] - demand[p, start + t]

    log_fd, log_path = tempfile.mkstemp(suffix=".log")
    os.close(log_fd)
    solver = pulp.HiGHS(msg=True, logPath=log_path)
    t0 = time.perf_counter()
    prob.solve(solver)
    total_p = time.perf_counter() - t0
    highs_tp = parse_highs_run_time(log_path) or 0.0
    pulp_total_times.append(total_p)
    pulp_highs_times.append(highs_tp)

p_total = statistics.median(pulp_total_times)
p_highs = statistics.median(pulp_highs_times)
l_total = statistics.median(linopy_total_times)
l_highs = statistics.median(linopy_highs_times)

print(f"{'':35s} {'PuLP':>10s} {'linopy':>10s}")
print("-" * 57)
print(f"{'solve call total (ms)':35s} {p_total*1e3:>10.1f} {l_total*1e3:>10.1f}")
print(f"{'  HiGHS 内部時間 (ms)':35s} {p_highs*1e3:>10.1f} {l_highs*1e3:>10.1f}")
print(f"{'  Python/interface 側 (ms)':35s} {(p_total-p_highs)*1e3:>10.1f} {(l_total-l_highs)*1e3:>10.1f}")
