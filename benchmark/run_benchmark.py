import time
import numpy as np
from solve_pulp import run_rolling_horizon_pulp
from solve_linopy import run_rolling_horizon_linopy

print("=== Rolling Horizon Benchmark (500 products, 9-day window, 51 windows) ===\n")

print("Running PuLP (HiGHS) ...")
t0 = time.perf_counter()
res_pulp = run_rolling_horizon_pulp()
t_pulp = time.perf_counter() - t0

print("Running linopy (HiGHS direct) ...")
t0 = time.perf_counter()
res_linopy = run_rolling_horizon_linopy()
t_linopy = time.perf_counter() - t0

bt_p = np.array(res_pulp["build_times"])
st_p = np.array(res_pulp["solve_times"])
bt_l = np.array(res_linopy["build_times"])
st_l = np.array(res_linopy["solve_times"])

print()
print(f"{'':30s} {'PuLP':>10s} {'linopy':>10s} {'speedup':>10s}")
print("-" * 62)
print(f"{'total wall time (s)':30s} {t_pulp:>10.2f} {t_linopy:>10.2f} {t_pulp/t_linopy:>9.1f}x")
print(f"{'build/update avg (ms)':30s} {np.mean(bt_p)*1000:>10.1f} {np.mean(bt_l)*1000:>10.1f} {np.mean(bt_p)/np.mean(bt_l):>9.1f}x")
print(f"{'  (excl. first build)':30s} {np.mean(bt_p)*1000:>10.1f} {np.mean(bt_l[1:])*1000:>10.1f} {np.mean(bt_p)/np.mean(bt_l[1:]):>9.1f}x")
print(f"{'solve avg (ms)':30s} {np.mean(st_p)*1000:>10.1f} {np.mean(st_l)*1000:>10.1f} {'':>10s}")
print(f"{'cumulative build time (s)':30s} {bt_p.sum():>10.2f} {bt_l.sum():>10.2f} {bt_p.sum()/bt_l.sum():>9.1f}x")
