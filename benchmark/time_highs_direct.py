"""HiGHS の各フェーズ（matrices/transfer/solve）の実行時間を計測する。"""
import sys, time, statistics
import numpy as np
import xarray as xr
import highspy
import linopy

sys.path.insert(0, ".")
from data import (
    N_PRODUCTS, WINDOW,
    demand, max_inventory, initial_inventory,
    production_cost, inventory_cost,
)

products = np.arange(N_PRODUCTS)
days = np.arange(WINDOW)
inner = np.arange(1, WINDOW)
da_max_inv   = xr.DataArray(max_inventory,   dims=["product"])
da_prod_cost = xr.DataArray(production_cost, dims=["product"])
da_inv_cost  = xr.DataArray(inventory_cost,  dims=["product"])

# モデル構築
m = linopy.Model()
coords = {"product": products, "day": days}
x = m.add_variables(lower=0, coords=coords, name="x")
s = m.add_variables(lower=0, upper=da_max_inv, coords=coords, name="s")
da_d = xr.DataArray(demand[:, :WINDOW], dims=["product", "day"])
da_init = xr.DataArray(initial_inventory, dims=["product"])
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
m.solve(solver_name="highs", io_api="direct",
        output_flag=False, log_to_console=False, progress=False)

N = 8
matrices_times, transfer_times, run_times = [], [], []
rng = np.random.default_rng(1)

for _ in range(N):
    # キャッシュを無効化するため RHS を微小変化
    m.constraints["b0"].rhs = (
        (da_init - da_d.isel(day=0)).values + rng.uniform(-0.1, 0.1, N_PRODUCTS)
    )

    # (1) m.matrices 再計算
    t0 = time.perf_counter()
    mat = m.matrices
    matrices_times.append(time.perf_counter() - t0)

    # (2) matrices → HiGHS オブジェクトへロード
    A = mat.A.tocsc()
    lb = mat.lb.astype(np.float64)
    ub = mat.ub.astype(np.float64)
    c  = mat.c.astype(np.float64)
    row_lb = mat.b[0].astype(np.float64)
    row_ub = mat.b[1].astype(np.float64)
    indptr  = A.indptr.astype(np.int32)
    indices = A.indices.astype(np.int32)
    data    = A.data.astype(np.float64)
    n_vars = len(lb)
    n_rows = A.shape[0]

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.addVars(n_vars, lb, ub)
    for j in range(n_vars):
        h.changeColCost(j, float(c[j]))

    t0 = time.perf_counter()
    h.addRows(row_lb, row_ub, n_rows, indptr, indices, data)
    transfer_times.append(time.perf_counter() - t0)

    # (3) HiGHS 求解
    t0 = time.perf_counter()
    h.run()
    t_run = time.perf_counter() - t0
    run_times.append(t_run)

    highs_rt = h.getInfoValue("run_time")[1]
    iters    = h.getInfoValue("simplex_iteration_count")[1]
    print(f"  h.run()={t_run*1e3:.2f}ms  "
          f"HiGHS_run_time={highs_rt*1e3:.2f}ms  "
          f"simplex_iters={iters}")

print()
print(f"m.matrices recompute:    {statistics.median(matrices_times)*1e3:6.1f} ms")
print(f"model → HiGHS (addRows): {statistics.median(transfer_times)*1e3:6.1f} ms")
print(f"HiGHS h.run() solve:     {statistics.median(run_times)*1e3:6.1f} ms")
total = statistics.median(matrices_times) + statistics.median(transfer_times) + statistics.median(run_times)
print(f"sum:                     {total*1e3:6.1f} ms")
