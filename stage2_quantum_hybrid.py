
!pip install dimod dwave-samplers

import pandas as pd
from collections import defaultdict
from dwave.samplers import SimulatedAnnealingSampler
import dimod

df = pd.read_csv("customer_predictions.csv")

costs = df["campaign_cost"].tolist()
probabilities = df["conversion_probability"].tolist()

n = len(df)
BUDGET = 250
REQUIRED = 6
PENALTY = 25

Q = defaultdict(float)

# Maximize predicted conversions
for i in range(n):
    Q[(f"x{i}", f"x{i}")] -= probabilities[i]

def add_square_penalty(terms, target, penalty):
    for var, a in terms:
        Q[(var, var)] += penalty * (a*a - 2*target*a)

    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            v1, a = terms[i]
            v2, b = terms[j]
            Q[(v1, v2)] += 2*penalty*a*b

# Exactly six customers
add_square_penalty(
    [(f"x{i}", 1) for i in range(n)],
    REQUIRED, PENALTY
)

# Budget constraint using binary slack variables
budget_terms = [
    (f"x{i}", costs[i]) for i in range(n)
]

for k in range(9):
    budget_terms.append((f"s{k}", 2**k))

add_square_penalty(budget_terms, BUDGET, PENALTY)

sampler = SimulatedAnnealingSampler()

samples = sampler.sample_qubo(
    dict(Q), num_reads=1000, seed=42
)

# Select the best feasible solution
valid = []

for sample, energy in samples.data(
    fields=["sample", "energy"]
):
    selected = [
        i for i in range(n)
        if sample[f"x{i}"] == 1
    ]

    total_cost = sum(costs[i] for i in selected)

    if len(selected) == REQUIRED and total_cost <= BUDGET:
        value = sum(probabilities[i] for i in selected)
        valid.append((value, selected, total_cost))

if not valid:
    raise ValueError("No feasible solution found.")

value, selected, total_cost = max(valid)

selected_ids = df.iloc[selected]["customer_id"].tolist()

print("Selected customers:", selected_ids)
print("Total customers:", len(selected_ids))
print("Total cost:", total_cost)
print("Expected conversions:", value)

pd.DataFrame({
    "customer_id": selected_ids
}).to_csv("stage2_selected.csv", index=False)
