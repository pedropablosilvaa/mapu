import time
import pandas as pd
from mapu.stats import nestednodf
from mapu.vegdist import designdist

x = pd.read_csv(
    "/Users/psilva/Documents/projects/ecology/package/comparison/adv_comm.csv"
)

# Warm up
nestednodf(x)
designdist(x, method="(A+B-2*J)/(A+B)", terms="minimum")

t0 = time.perf_counter()
for _ in range(50):
    nestednodf(x)
t1 = time.perf_counter()
print(f"nestednodf: {(t1-t0)/50 * 1000:.2f} ms")

t0 = time.perf_counter()
for _ in range(50):
    designdist(x, method="(A+B-2*J)/(A+B)", terms="minimum")
t1 = time.perf_counter()
print(f"designdist: {(t1-t0)/50 * 1000:.2f} ms")
