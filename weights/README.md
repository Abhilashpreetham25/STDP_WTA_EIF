# Trained networks

| folder | network | files |
|---|---|---|
| `LIF_400` | LIF baseline (E0), 3 epochs, seed 1 | `XeAe.npy`, `theta_A.npy`, `assignments.npy` |
| `EIF_homeo_s-1` | homeostatic EIF, s₀ = −1 mV (E3), 3 epochs, seed 35 | `XeAe.npy`, `theta_A.npy`, `s_A.npy`, `assignments_adaptive.npy` |
| `connections` | recurrent Ae→Ai and Ai→Ae weights used by every run | `AeAi.npy`, `AiAe.npy` |

`XeAe.npy` and `theta_A.npy` use the original Diehl & Cook format: (i, j, w) rows, and θ in volts. `s_A.npy` is each neuron's s in volts. `assignments*.npy` gives the digit of each neuron, taken from the 60,000-image label run, so `evaluate.py --assignments` gives paper-protocol accuracy from a test run alone. `assignments_adaptive.npy` comes from the label run with `--s-mode adaptive`, so use it with `--s-mode adaptive` tests. For frozen s, run a label pass first.

Checked: with these files, the test runs in `results/` give 89.40% (LIF) and 89.31% (EIF, adaptive s).
