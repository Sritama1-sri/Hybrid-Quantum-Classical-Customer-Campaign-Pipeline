
!pip install qiskit qiskit-aer

import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

df = pd.read_csv("customer_predictions.csv")

# Read exact six IDs from Stage 2
stage2 = pd.read_csv("stage2_selected.csv")
selected_ids = stage2["customer_id"].tolist()

assert len(selected_ids) == 6
assert len(set(selected_ids)) == 6

n = len(df)

qc = QuantumCircuit(n, n)

for i, customer in enumerate(df["customer_id"]):
    if customer in selected_ids:
        qc.x(i)

qc.measure(range(n), range(n))

backend = AerSimulator()

compiled = transpile(
    qc, backend,
    optimization_level=1,
    seed_transpiler=42
)

job = backend.run(
    compiled,
    shots=1024,
    seed_simulator=42
)

result = job.result()
counts = result.get_counts()

print("Selected IDs:", selected_ids)
print("Simulation counts:", counts)
print("Qubits:", compiled.num_qubits)
print("Circuit depth:", compiled.depth())
print("Gate counts:", compiled.count_ops())
print("SWAP count:", compiled.count_ops().get("swap", 0))

compiled.draw(
    output="mpl",
    filename="quantum_circuit.png"
)
