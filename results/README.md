# Results

Each test run has its settings (`run_info.json`) and evaluation (`evaluation.json`, paper protocol) in `runs/`. HTML reports are in `reports/`; open `report.html` in a browser, or enable GitHub Pages for the repository to view them online. Accuracy below comes from `evaluation.json`.

| experiment | run | accuracy | full report | region report |
|---|---|---|---|---|
| E0 LIF baseline | [LIF_test400](runs/LIF_test400) | 89.40% | [report](reports/LIF_test400/report.html) · [training](reports/LIF_train400/report.html) | – |
| E1 fixed vd −28.5, Δ 0.5 | [run](runs/E1_fixed_vd-28.5_d0.5) | 88.68% | – | [regions](reports/regions_E1_vd-28.5_d0.5/report.html) |
| E1 fixed vd −30.5, Δ 1.0 | [run](runs/E1_fixed_vd-30.5_d1.0) | 87.97% | – | [regions](reports/regions_E1_vd-30.5_d1.0/report.html) |
| E1 fixed vd −31.5, Δ 0.5 | [run](runs/E1_fixed_vd-31.5_d0.5) | 84.57% | [report](reports/E1_fixed_vd-31.5_d0.5/report.html) | [regions](reports/regions_E1_vd-31.5_d0.5/report.html) |
| E1 fixed vd −33.5, Δ 1.0 | [run](runs/E1_fixed_vd-33.5_d1.0) | 84.11% | – | [regions](reports/regions_E1_vd-33.5_d1.0/report.html) |
| E1 fixed vd −38.87, Δ 0.5 | [run](runs/E1_fixed_vd-38.87_d0.5) | 62.44% | – | [regions](reports/regions_E1_vd-38.87_d0.5/report.html) |
| E2 EIF-trained, vd −32.87 | [run](runs/E2_EIFtrained_vd-32.87_d0.5) | 82.37% | [report](reports/E2_EIFtrained_vd-32.87_d0.5/report.html) | [regions](reports/regions_E2_EIFtrained/report.html) |
| E3 homeo s₀ = 0, frozen | [run](runs/E3_homeo_s0_frozen) | 88.41% | [report](reports/E3_homeo_s0_frozen/report.html) | [regions](reports/regions_E3_s0_frozen/report.html) |
| E3 homeo s₀ = 0, adaptive | [run](runs/E3_homeo_s0_adaptive) | 89.01% | [report](reports/E3_homeo_s0_adaptive/report.html) | [regions](reports/regions_E3_s0_adaptive/report.html) |
| E3 homeo s₀ = −1, frozen | [run](runs/E3_homeo_s-1_frozen) | 88.90% | [report](reports/E3_homeo_s-1_frozen/report.html) | [regions](reports/regions_E3_s-1_frozen/report.html) |
| E3 homeo s₀ = −1, adaptive | [run](runs/E3_homeo_s-1_adaptive) | 89.31% | [report](reports/E3_homeo_s-1_adaptive/report.html) | [regions](reports/regions_E3_s-1_adaptive/report.html) |

The full reports of the EIF runs were made without `--label-dir`, so their header accuracy uses labels from the test set itself, which is optimistic. The paths inside the reports and `run_info.json` are the original local paths (`runs/...`); the run folders were renamed here to say what each one is.

Not included: weights, spike counts and C++ build folders (too large), and three E1 full reports that were generated against the wrong run folder.
