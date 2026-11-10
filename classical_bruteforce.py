
import pandas as pd
from itertools import combinations
import time

df = pd.read_csv("customer_predictions.csv")

costs = df["campaign_cost"].tolist()
probabilities = df["conversion_probability"].tolist()

BUDGET = 250
REQUIRED = 6

start = time.time()

feasible = []

for combo in combinations(range(len(df)), REQUIRED):
    cost = sum(costs[i] for i in combo)
    score = sum(probabilities[i] for i in combo)

    if cost <= BUDGET:
        feasible.append((score, combo, cost))

if not feasible:
    raise ValueError("No feasible combination found.")

best_score, best_combo, best_cost = max(feasible)

best_ids = df.iloc[list(best_combo)]["customer_id"].tolist()

print("Best customer IDs:", best_ids)
print("Total cost:", best_cost)
print("Expected conversions:", best_score)
print("Execution time:", time.time() - start)

pd.DataFrame({
    "customer_id": best_ids
}).to_csv("bruteforce_result.csv", index=False)
