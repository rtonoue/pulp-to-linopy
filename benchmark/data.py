import numpy as np

N_PRODUCTS = 500
N_DAYS = 365
WINDOW = 9
SLIDE = 7

RNG = np.random.default_rng(42)

demand = RNG.integers(10, 100, size=(N_PRODUCTS, N_DAYS)).astype(float)
max_inventory = RNG.integers(200, 500, size=N_PRODUCTS).astype(float)
initial_inventory = (max_inventory * 0.3).astype(float)
production_cost = RNG.uniform(1.0, 5.0, size=N_PRODUCTS)
inventory_cost = RNG.uniform(0.1, 0.5, size=N_PRODUCTS)
