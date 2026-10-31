# Experiments

All networks have 400 excitatory and 400 inhibitory neurons, dt = 0.5 ms, Euler integration, 350 ms image + 150 ms rest. Accuracy uses the paper protocol: each neuron gets the digit it fires most for over the 60,000 training images, then the 10,000 test images are classified by the mean response of each digit's neurons. Settings for every test run are in `results/runs/<run>/run_info.json`; the evaluation is in `evaluation.json` next to it.

## The model in one paragraph

Excitatory neuron (LIF in the original):

    τ dv/dt = (E_L − v) + g_e(−v) + g_i(E_inh − v)          spike when v > v_th + θ

θ rises 0.05 mV with every spike and decays very slowly. It is what stops one neuron from winning every image. The EIF version adds one term:

    τ dv/dt = (E_L − v) + Δ·exp((v − v_d)/Δ) + g_e(−v) + g_i(E_inh − v)

Above the runaway point Vx (set by v_d and Δ), v accelerates to a spike on its own. So each neuron is in one of three regions, depending on where its detector (v_th + θ, which grows with θ) sits relative to Vx:

| region | where the detector is | behaviour |
|---|---|---|
| A | below Vx | spike is detected before the runaway starts: LIF-like |
| B | 0–3 Δ above Vx | the runaway decides the spike |
| C | more than 3 Δ above Vx | θ no longer restrains the neuron |

`regions.py` puts every neuron in a region and relates the regions to accuracy and errors.

## E0. LIF baseline

Trained 3 epochs (seed 1), labelled and tested with seed 0. Report: `results/reports/LIF_test400`, training report `LIF_train400`.

| accuracy | spikes / image | Gini (test / label) | main confusions |
|---|---|---|---|
| **89.40%** | 17.9 | 0.301 / 0.251 | 4→9, 3→8, 7→9, 9→4, 2→8 |

The paper reports 87.0% for 400 neurons.

## E1. EIF at test time, one fixed vd

**Question.** If I keep the LIF-trained network (weights, θ and neuron labels from E0) and only switch the excitatory neurons to EIF, what does the exponential term do to competition and accuracy?

**What I ran.** Five test passes, each with one vd for all 400 neurons. I chose the vd values from the region map of the trained θ so they cover the whole range: almost every neuron in region A (vd −28.5 mV), a mix of all three regions (−31.5, −33.5), and every neuron in region C (−38.87). I ran Δ = 0.5 and 1.0 mV to see whether the onset sharpness matters on its own.

| vd (mV) | Δ (mV) | Vx (mV) | regions A / B / C | accuracy | spikes / image | vs LIF: right→wrong / wrong→right |
|---|---|---|---|---|---|---|
| −28.5 | 0.5 | −26.33 | 378 / 11 / 11 | 88.68% | 18.0 | 229 / 157 |
| −30.5 | 1.0 | −26.86 | 366 / 26 / 8 | 87.97% | 18.2 | 269 / 126 |
| −31.5 | 0.5 | −29.37 | 262 / 76 / 62 | 84.57% | 19.2 | 642 / 159 |
| −33.5 | 1.0 | −29.94 | 221 / 144 / 35 | 84.11% | 19.4 | 691 / 162 |
| −38.87 | 0.5 | −36.85 | 0 / 0 / 400 | 62.44% | 7,591 | 2877 / 181 |

| vd | mean θ of A / B / C neurons (mV) | C share of the winning digit's spikes (correct / wrong images) | images wrongly called "1" (LIF: 136) |
|---|---|---|---|
| −28.5 | 41.6 / 46.2 / 48.6 | 5.3% / 10.0% | 260 |
| −30.5 | 41.5 / 46.0 / 49.0 | 4.8% / 9.5% | 309 |
| −31.5 | 40.6 / 43.3 / 45.8 | 27.4% / 58.9% | 692 |
| −33.5 | 40.2 / 43.3 / 46.6 | 19.4% / 53.3% | 724 |

**What I concluded.**

1. **What matters is how many neurons are past Vx, not vd or Δ themselves.** Accuracy falls steadily with the share of neurons whose detector sits past Vx: 5.5% → 88.7%, 8.5% → 88.0%, 34.5% → 84.6%, 44.8% → 84.1%, 100% → 62.4%. The Δ = 0.5 and Δ = 1.0 runs fall on the same trend.
2. **No fixed vd beats LIF.** Even with 378 neurons in region A, 229 images flip from right to wrong and only 157 from wrong to right (McNemar p = 3·10⁻⁴).
3. **A single vd hits the high-θ neurons first.** θ differs by about 13 mV across neurons (37–50 mV). The neurons with the largest θ, which won most often during training, are the first to have their detector pushed past Vx, and once there they over-respond. The digit-1 neurons have the highest mean θ (45.9 mV), so "1" becomes an attractor: images wrongly classified as 1 rise from 136 (LIF) to 724.
4. **Region C neurons cause the errors.** On wrong images, C neurons give about twice the share of the winning digit's spikes that they give on correct images.
5. **With every neuron in region C, winner-take-all collapses.** The network fires 7,591 spikes per image, against 18 for LIF; inhibition can no longer stop the runaway.

So the onset point cannot be one number for the whole network. It has to be set relative to each neuron's own threshold. E2 then asks whether training with EIF fixes this on its own.

**Limitation.** The weights were learned with LIF neurons, so E1 shows the mismatch between EIF dynamics and a network tuned for LIF, not what an EIF network would learn.

Reports: `results/reports/regions_E1_*` for every run, and `E1_fixed_vd-31.5_d0.5` for the full report of one run.

## E2. Training with EIF and one fixed vd

**Question.** Does the network adapt if it learns with EIF neurons from the start?

Trained from scratch with EIF neurons. Labelled and tested at vd = −32.87 mV, Δ = 0.5 mV (Vx = −30.76 mV), seed 1.

| accuracy | spikes / image | regions A / B / C | θ range (mV) | vs LIF: right→wrong / wrong→right |
|---|---|---|---|---|
| 82.37% | 21.1 | 280 / 46 / 74 | 35.9 – 374.8 | 899 / 196 |

- 45 neurons ended with θ > 60 mV (23 above 100 mV). In every LIF-trained network θ stays at or below 50 mV.
- Those 45 neurons fire 29.8% of all test spikes.
- Region-C share of the winning digit's spikes: 40.4% on correct images, 71.2% on wrong ones.
- The 7 digit-1 neurons all ended in region C, with a mean θ of 255 mV; 940 test images were wrongly called "1" (LIF: 136). This is the same attractor as in E1, now created by learning.

**Result: homeostasis escape.** θ normally stops a busy neuron by raising its detector. Once the detector is past Vx, though, the exponential term fires the neuron before v gets near the detector. Raising θ then no longer slows the neuron down, so each spike raises θ further. A fixed vd cannot work with an adaptive threshold.

Reports: `results/reports/E2_EIFtrained_vd-32.87_d0.5`, `regions_E2_EIFtrained`.

## E3. Per-neuron homeostatic onset point

### Why this design

E1 and E2 gave three requirements:

1. **Vx must be defined relative to each neuron's own threshold.** A shared vd hits the high-θ neurons first (E1). When θ grows past a fixed Vx, θ stops working (E2). So each neuron stores s = threshold − Vx instead of vd. When θ rises, Vx rises with it, and a neuron can only cross into region C if s itself grows.
2. **The rule must be local and driven by the neuron's own spikes, like θ.** I rejected rules based on the population's mean threshold or mean rate. They need information a single neuron does not have, and they make every neuron move together.
3. **s must stay in a range where the EIF equation is valid.** If Vx gets closer than Δ to E_L, the neuron has no resting state. So s has a floor and a ceiling.

Before settling on this, I tried two other rules:

- **Explicit δ rules (`--vd-mode proposal`).** My first design: Vx moved in fixed steps depending on how far the threshold was from reset. Learning stalled, and accuracy fell to chance in short runs. Some steps asked for a Vx below E_L + Δ, which breaks requirement 3.
- **Tracking (`--vd-mode track`).** vd = threshold − k, the same k for every neuron. It meets requirement 1, but every neuron sits at the same distance from its runaway, so nothing adapts.

The homeostatic rule (`--vd-mode homeo`) meets all three requirements:

- every own spike: s −= a (a = 0.02 mV), so the neuron becomes less excitable;
- between spikes: s drifts up at a·r0 (r0 = 0.0735 Hz, i.e. 14.7 spikes per image spread over 400 neurons and 0.5 s), so the neuron becomes more excitable;
- s is kept in [−1.5, min(2, θ − 7 − Δ − 1)] mV.

A neuron firing above r0 drifts toward region A (LIF-like), and a quiet one drifts toward region B, where the runaway helps it fire. It is the same idea as θ, applied to the spike onset instead of the threshold.

**Settings tested.**
- **Two starting values.** s₀ = 0 puts every neuron on the A/B boundary. s₀ = −1 mV starts every neuron LIF-like, in region A. This shows whether the end state depends on the start.
- **Frozen or adaptive s at test.** *Frozen*: each neuron keeps the s it learned. *Adaptive*: s keeps following the rule during the label and test passes. The rule uses no labels, so the network stays unsupervised. Comparing the two separates the effect of the learned s values from the effect of ongoing homeostasis.

### Runs

Trained 3 epochs (180,000 images) with Δ = 0.5 mV, from s₀ = 0 (seed 1) and s₀ = −1 mV (seed 35). My first s₀ = −1 run, with seed 1, locked up after about 2,000 images: every neuron fired at its maximum rate (see Caveats), so I retrained it with seed 35. Each network was then labelled and tested twice, frozen and adaptive. Commands and full output: `notebooks/EIF_homeo_s0.ipynb`, `notebooks/EIF_homeo_s-1.ipynb`.

| network | accuracy | McNemar vs LIF (right→wrong / wrong→right, p) | Gini test / label | regions A / B at end of test | spikes / image |
|---|---|---|---|---|---|
| LIF (E0) | **89.40%** | – | 0.301 / 0.251 | – | 17.9 |
| s₀ = 0, frozen | 88.41% | 308 / 209, p = 1.5·10⁻⁵ | 0.314 / 0.267 | 398 / 2 | 18.1 |
| s₀ = 0, adaptive | 89.01% | 273 / 234, p = 0.09 | **0.241 / 0.155** | 307 / 93 | 18.4 |
| s₀ = −1, frozen | 88.90% | 283 / 233, p = 0.03 | 0.321 / 0.271 | 399 / 1 | 18.1 |
| s₀ = −1, adaptive | **89.31%** | 259 / 250, p = 0.72 | **0.243 / 0.154** | 294 / 106 | 18.3 |

McNemar: exact two-sided binomial test on the test images where exactly one of the two networks is right.

Training accuracy over the last 10,000 images: 90.9% (s₀ = 0), 90.8% (s₀ = −1). At the end of training 398 and 399 neurons were in region A, 16 of them at the floor.

### Outcomes

1. **The homeostasis escape is gone.** θ stays at 35–50 mV, as in LIF; no neuron reaches region C; and there is no digit attractor. Accuracy is back from 82.4% (E2) to the LIF level.
2. **With adaptive s, EIF matches LIF and shares the firing more evenly.** Accuracy is 89.31% and 89.01%, against 89.40%, and neither difference is significant. The Gini falls by about 0.06 on the test set and about 0.10 on the label set, so more neurons take part. Fewer images need a repeat presentation: 138–139 per 10,000 test images against 182 for LIF, and 883–894 per 60,000 label images against 1,185.
3. **Frozen s is slightly worse.** It costs 0.4–0.6 points of accuracy (significant for s₀ = 0) and gives LIF-level inequality. Most of the benefit comes from homeostasis continuing during testing, not from the learned s values.
4. **The starting value matters little.** Both starts end in nearly the same place: almost all neurons in region A, and very similar accuracy and Gini.
5. **The homeostasis is one-sided.** Training drove almost every neuron into region A. During adaptive testing, quiet neurons climbed into region B (93 and 106 neurons) while busy neurons sat at the floor. Lowering s further changes almost nothing, because a neuron deep in region A already behaves like LIF. So the rule can lift quiet neurons but cannot brake busy ones, which is why the Gini improves while accuracy does not. In these runs neurons fired about 0.09 Hz (18 spikes / 400 neurons / 0.5 s), above r0, which also pushes most of them down.
6. **The same mistakes as LIF.** All five networks share the top confusions 4→9, 3→8, 7→9 and 2→8. The remaining errors come from the learned templates and the spike-count readout, not from the neuron model.

Reports: `results/reports/E3_homeo_*` (receptive fields, confusions, activity) and `results/reports/regions_E3_*` (regions, how s moved, Gini, comparison with LIF).

## What I conclude

1. Swapping in a more realistic spike onset is not free. The onset point has to follow each neuron's adaptive threshold, otherwise homeostasis fails.
2. With a per-neuron homeostatic onset, EIF reaches LIF accuracy and spreads activity more evenly across neurons. It does not improve accuracy.
3. With a spike-count readout over 350 ms, what limits accuracy is the learned templates and the readout, not how a spike starts. The EIF difference is in spike timing, and a count readout ignores timing. That is the motivation for [next-steps.md](next-steps.md).

## Caveats

- One seed per configuration; E0 used a different seed (0) from E3 (1 / 35).
- In E3 the label and test runs used Δ = 1.0 mV, the default, while training used 0.5 mV. Both are recorded in `run_info.json`, and region counts use the Δ of the test run. To be rerun with Δ = 0.5.
- Euler at dt = 0.5 ms becomes unstable for large inhibitory conductance (g_i above about 400). This is what locked up the first s₀ = −1 run. A semi-implicit update of v would remove it.
- The visualize reports of the EIF runs were made without `--label-dir`, so the accuracy in their header uses labels from the test set itself, which is optimistic. The tables here come from `evaluate.py`, using the paper protocol.
- Three visualize reports from E1 (vd −28.5, −38.87 and −33.5) were generated against the wrong run folder, so they are not included. The region reports of those runs are correct and are included.
