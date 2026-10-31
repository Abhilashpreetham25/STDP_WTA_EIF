# Usage guide

This folder runs the Diehl & Cook unsupervised digit-recognition network as **one compiled
C++ program**. Python prepares the inputs, generates the C++ code, compiles it, and
reads the results back. The whole experiment loop runs inside the compiled program:
showing images, retrying silent examples with a louder input, normalising weights,
and STDP learning.

| File | What it is |
|---|---|
| `dc_standalone.py` | the simulation (LIF neurons). Every setting is a command-line option |
| `EIF.py` | the same simulation with EIF excitatory neurons: `--vd-mode fixed / track / homeo / proposal`, per-neuron adaptive s (`--homeo`, `--s-init`, `--s-mode`). See **[EIF neurons](#eif-neurons-eifpy)** at the end |
| `evaluate.py` | accuracy from a label run plus a test run (paper protocol), or a test run alone. For EIF runs also the neuron settings, s statistics and label/test mismatches |
| `visualize.py` | HTML report of a run: receptive fields, confusions, spike timing, training curves |
| `regions.py` | EIF only: each neuron's region (A/B/C) from its Vx, related to accuracy and errors; for homeo runs, how s moved |
| `config_example.json` | example parameter file |

---

## 1. Setup

You need Python 3, `brian2` (tested with 2.10.1), `numpy`, and **a C++ compiler**.

- **Linux / WSL:** `g++` (`sudo apt install build-essential`).
- **macOS:** Xcode command-line tools (`xcode-select --install`).
- **Windows:** "Microsoft C++ Build Tools" (the *Desktop development with C++* workload).
  If Brian2's Cython target already works on your machine, the compiler is installed.
  All runs in `results/` were done on Windows with MSVC.

Expected folder layout (the defaults; every path can be changed with an option):

```
your_project/
├── code/                    <- this repository's code folder
├── stdp-mnist/                       <- the original GitHub repository
│   ├── weights/XeAe.npy, theta_A.npy     pretrained 400-neuron weights
│   └── random/XeAe.npy, AeAi.npy, AiAe.npy   initial / recurrent weights (400 neurons)
└── brain_env/Lib/site-packages/mnist/   training.pickle + testing.pickle (or the idx files)
```

Run all commands from `your_project/`.

`EIF.py` defaults to `--mnist-dir ../brain_env/Lib/site-packages/mnist/` because I ran it from inside the code folder; pass `--mnist-dir` when running from elsewhere.

---

## 2. Typical workflows

**Test the pretrained 400-neuron network**
```
python code/dc_standalone.py --mode label --n-e 400 --out-dir runs/label400
python code/dc_standalone.py --mode test  --n-e 400 --out-dir runs/test400
python code/evaluate.py --label-dir runs/label400 --test-dir runs/test400
```

**Train your own network, then evaluate it**
```
python code/dc_standalone.py --mode train --n-e 400 --epochs 3 --out-dir runs/train400
python code/dc_standalone.py --mode label --n-e 400 --weights-dir runs/train400/weights ^
       --conn-dir runs/train400/connections --out-dir runs/train400_label
python code/dc_standalone.py --mode test  --n-e 400 --weights-dir runs/train400/weights ^
       --conn-dir runs/train400/connections --out-dir runs/train400_test
python code/evaluate.py --label-dir runs/train400_label --test-dir runs/train400_test
```
The label and test *simulations* do not depend on each other: both use the same frozen
weights and only record spike counts. The labelling happens afterwards in `evaluate.py`,
which assigns each neuron a digit from the label-run counts and applies those assignments
to the test-run counts. So on a machine with 2 or more cores you can start the label and
test runs in two terminals at the same time; only `evaluate.py` has to wait for both.

**A different network size.** Use `--n-e 100` (or any size). The files in `--conn-dir`
only fit 400 neurons, so the script generates new matrices with Diehl's rules:
input weights `(rand + 0.01) * 0.3`, Ae→Ai one-to-one at 10.4, and Ai→Ae all-to-all
except self at 17.0. It saves them in `<out-dir>/connections/`. Pass that folder as
`--conn-dir` when you label/test the trained network, so it uses the same recurrent
weights.

**Use a parameter file.** Put any options in a JSON file (keys use underscores):
```
python code/dc_standalone.py --config my_experiment.json --seed 3
```
Options typed on the command line override the file, and the file overrides the
defaults. Every run writes all the values it used to `<out-dir>/run_info.json`.

---

## 3. Parameters

Run `python dc_standalone.py --help` for the full list with defaults. The main ones:

### Run
| Option | Default | Meaning |
|---|---|---|
| `--mode` | `test` | `train` (STDP on, training images), `label` (frozen, training images), `test` (frozen, test images) |
| `--n-examples` | 10000 / 60000 / 60000×epochs | examples to process (test / label / train) |
| `--epochs` | 3 | train mode only |
| `--first-example` | 0 | start at image k; wraps around the data set. Use it to split a pass into shards |
| `--seed` | 0 | Poisson input, synaptic delays, generated weights |
| `--n-e` | 400 | excitatory neurons (inhibitory = same number) |

### Timing
| Option | Default | Meaning |
|---|---|---|
| `--stim-ms` | 350 | image presentation |
| `--rest-ms` | 150 | silence after every presentation |
| `--dt-ms` | 0.5 | time step (stim and rest must be whole multiples of it). 0.05 is closer to the exact dynamics but much slower |
| `--method` | `euler` | neuron integration method. `exponential_euler` is **not** better at dt 0.5 (it overestimates excitation) |
| `--max-delay-ms` | 10 | input→excitatory delays, uniform in [0, max) |

### Input and retry rule
| Option | Default | Meaning |
|---|---|---|
| `--start-intensity` | 2 | rate (Hz) = pixel / `rate-divisor` × intensity, so at most 63.75 Hz |
| `--rate-divisor` | 8 | |
| `--min-spikes` | 5 | if Ae fires fewer spikes than this, the example is shown again |
| `--intensity-step` | 1 | intensity added on each retry |
| `--max-retries` | 10 (test/label), 100 (train) | cap per example, as in the earlier Brian2 port. `-1` = unlimited (Brian1 behaviour). After the cap the example is accepted anyway |
| `--count-window` | `cycle` | `cycle`: count from the end of the previous stimulus, so spikes in the preceding rest are included (this is what the original code does). `stim`: count only during the stimulus |

### Neurons and synapses
`--tau-e-ms 100`, `--v-rest-e-mv -65`, `--v-reset-e-mv -65`, `--v-thresh-e-mv -52`,
`--theta-offset-mv 20` (threshold = v_thresh + θ − offset), `--refrac-e-ms 5`,
`--e-inh-e-mv -100`; inhibitory: `--tau-i-ms 10`, `--v-rest-i-mv -60`,
`--v-reset-i-mv -45`, `--v-thresh-i-mv -40`, `--refrac-i-ms 2`, `--e-inh-i-mv -85`;
conductances `--tau-ge-ms 1`, `--tau-gi-ms 2`; generated weights `--w-ei 10.4`,
`--w-ie 17`, `--w-input-init 0.3`.

### Learning (train mode)
| Option | Default | Meaning |
|---|---|---|
| `--nu-pre`, `--nu-post` | 0.0001, 0.01 | depression on pre spikes, potentiation on post spikes |
| `--tc-pre-ms`, `--tc-post1-ms`, `--tc-post2-ms` | 20, 20, 40 | trace time constants |
| `--wmax` | 1.0 | weights are clipped to [0, wmax] |
| `--w-total` | 78 | each neuron's input weights are rescaled to sum to this before every image |
| `--theta-plus-mv`, `--tc-theta-ms`, `--theta-init-mv` | 0.05, 1e7, 20 | adaptive threshold |
| `--init-from-weights` | off | continue training from `--weights-dir` instead of random weights |

### EIF neurons (`EIF.py` only; full explanation in [EIF neurons](#eif-neurons-eifpy))
| Option | Default | Meaning |
|---|---|---|
| `--vd-mode` | `fixed` | `fixed`: one vd for all · `track`: vd = vth − k · `homeo`: per-neuron s = vth − Vx, adapted homeostatically · `proposal`: original δ1/δ2/δ3 rules (reproduction only) |
| `--delta-e-mv` | 1.0 | Δ, sharpness of spike onset (mV). Runs so far used 0.5 |
| `--vd-e-mv` | 100 | fixed mode: vd (mV). 100 = exponential off (identical to LIF) |
| `--vd-k-mv` | 2.0 | track mode: k = vth − vd (mV). 2.09 at Δ = 0.5 puts Vx at the threshold |
| `--homeo` | `a=0.02,r0=0.0735,smin=-1.5,smax=2.0,margin=1` | homeo mode: `a` = change of s per own spike (mV); `r0` = target rate (Hz), s drifts up at a·r0; s kept in [smin, min(smax, δ1 − Δ − margin)] mV |
| `--s-init` | train: `0` (homeo) · label/test: `trained` | starting s for every neuron: a number in mV (`0`, `-1`, `-1.5`, `2`), or `trained` = per-neuron values from `<weights-dir>/s_A.npy` |
| `--s-mode` | train: `adaptive` · label/test: `frozen` | `adaptive`: s keeps changing during the run · `frozen`: s stays at its start value |
| `--proposal` | see `--help` | proposal mode rule parameters |

### Outputs and performance
| Option | Default | Meaning |
|---|---|---|
| `--out-dir` | `runs/<mode>_ne<N>` | results folder |
| `--build-dir` | `<out-dir>/build` | generated C++ code. Reusing it skips recompiling unchanged files |
| `--report-every` | 1000 | progress line with ETA every N examples, printed and also written to `<out-dir>/progress.txt` (IDE consoles such as Jupyter or Spyder often don't show the compiled program's output) |
| `--snapshot-every` | 10000 | train: save weights/θ about every N examples (0 = off) |
| `--perf-block` | 10000 | train: accuracy per block, as the original printed during training |
| `--save-train-activity` | off | train: also save per-example spike counts |
| `--record-inh-spikes` | off | keep every inhibitory spike (otherwise only the count) |
| `--threads` | 0 | OpenMP threads. Measured slower than 1 thread for 400 neurons |

---

## 4. How the code is organised

`dc_standalone.py` has six numbered sections:

| Section | Functions | What happens |
|---|---|---|
| 1 PARAMETERS | `build_parser`, `get_args` | options, JSON config, defaults that depend on the mode |
| 2 DATA | `load_mnist` | reads the pickles or idx files → images `[N,784]`, labels `[N]` |
| 3 WEIGHTS | `load_matrix`, `save_matrix`, `get_weights` | reads `(i, j, w)` weight files, generates missing ones |
| 4 NETWORK | `build_network`, `add_synapses`, `add_monitors` | all Brian2 objects (4a–4h, see below) |
| 5 RUN | `run` | standalone device, `Network.run`, compile and execute |
| 6 RESULTS | `spike_counts_per_example`, `assign_and_predict`, `save_results` | per-example counts, labels, weights, `run_info.json` |

**What runs where.** Sections 1–3 and 6 are ordinary Python. Section 4 only *describes* the
network; nothing is simulated until `device.build()` in section 5 turns it into C++ (in
`--build-dir`) and runs it. The consequence: **you cannot put Python code inside the
simulation loop.** Anything that must happen during the run has to be written as Brian2
model code: equations, `on_pre`/`on_post`, `run_regularly`, or a function with a C++
implementation. That is why the experiment loop lives in a "controller" (4b).

### Section 4 in detail

**4a `stop_network`**: a 3-line C++ function. When the last example's rest period ends,
it sets Brian's internal stop flag, so the run ends exactly there. `network.run()` is
given an upper bound, `n_examples × (1 + max_retries)` presentations, which is never
reached normally.

**4a `report_progress`**: prints the `[k/N examples] elapsed … ETA …` lines.

**4b Controller**: a `NeuronGroup` with **one** element that holds the experiment state
(`idx` = current example, `retry`, `intensity`, `total` = spike count, `cs` = step within
the presentation). Its `run_regularly` code executes at the start of every time step,
before anything else. Every presentation, retry or not, has the same shape:

```
step cs:  0 ............ STIM-1 | STIM ............. CYCLE-1 | 0 ...
          image on (stim = 1)   | rest (stim = 0)            | next presentation
                                ^ decision step (end_stim = 1)
```

At the decision step:
- `fail = total < MIN_SPIKES and retry < MAX_RETRIES`: keep the same `idx`, add
  `INTENSITY_STEP` to the intensity, increase `retry`.
- otherwise `accept`: record the example (the `accepted` event, monitored in 4h), move to
  `idx + 1`, reset the intensity and retry counter.

`total` is reset at the decision step (`--count-window cycle`) or at step 0 (`stim`). At
the last rest step after the final example, it calls `stop_network`. The code uses
`int(condition)` arithmetic instead of `if`, because each line is one vectorised
statement.

**4c Input layer `X`**: 784 Poisson neurons. Their threshold is
`rand() < stim_ * img(image, i) * intensity_ * dt`, which is what `PoissonGroup` does.
`img` is a `TimedArray` used as a lookup table, where row k holds image k's rates. The
variables ending in `_` are *linked variables*: read-only views of the controller's
`stim`, `idx` and `intensity`.

**4d Neurons `E`, `I`**: the conductance-based LIF equations of the paper. θ is fixed in
label/test and adaptive in train (decays with `tc_theta`, `+theta_plus` per spike).

**4e Input synapses `syn_input`**: train mode has the STDP rule (event-driven traces
`pre`, `post1`, `post2`, weights clipped to [0, wmax]). Label/test mode has a plain
`ge_post += w` and skips zero weights. With learning off the two give identical results;
the plain version is faster.

**4f Normalisation (train)**: three `run_regularly` steps every `CYCLE`, at the start of
the step: `colsum = 0`, then `colsum_post += w`, then `w *= w_total / colsum_post`.
Presentations always start at multiples of `CYCLE`, so this runs before every image,
retries included, like the Python loop did.

**4g Recurrent synapses**: `syn_ei` (Ae→Ai, raises **ge**) and `syn_ie` (Ai→Ae, raises
**gi**). The *source* population sets the sign. Getting this backwards silently
switches off the winner-take-all.

**4h Monitors**: `syn_counter` adds every Ae spike to the controller's `total`.
`mon_exc` keeps all Ae spikes. `mon_accepted` records each accepted example (index,
spike count, intensity, retries). In train mode, the `snap_*` monitors take weight and θ
snapshots.

**Section 6**: the per-example spike counts are rebuilt from `mon_exc` and the accepted
times. They are checked against the controller's own counter, and a warning prints if
they disagree.

---

## 5. Output files

| File | Content |
|---|---|
| `activity/resultPopVecs<N>.npy` | (label/test) spike counts `[N, n_e]`, same format as the original |
| `activity/inputNumbers<N>.npy` | (label/test) digit label of each example |
| `examples.npz` | per example: image index, label, retries, final intensity, spikes, stimulus end time |
| `weights/XeAe.npy`, `weights/theta_A.npy` | (train) learned weights `(i, j, w)` and θ in volts, original format |
| `weights/XeAe_ex<k>.npy`, `theta_A_ex<k>.npy` | (train) snapshots, `k` = examples finished at that moment |
| `connections/` | the recurrent (and, in train mode, initial input) matrices this run used |
| `run_info.json` | all parameters, weight sources, timings, presentations, retries, spike totals, train accuracy per block |
| `build/` | generated C++ project (safe to delete) |

`evaluate.py` writes `evaluation.json` into the test folder.

---

## 6. Changing the model: where to edit

| You want to… | Edit |
|---|---|
| change a constant | command-line option, no code change |
| change the neuron equations | `build_network` 4d (`E`/`I` model strings) |
| change the learning rule | `add_synapses` 4e (`syn_input` model, `on_pre`, `on_post`) |
| change the retry rule or the schedule | `build_network` 4b (controller `run_regularly` code) |
| record something extra | `add_monitors` (any Brian2 monitor), then read it in `save_results` |
| add a new option | `build_parser`, then use `args.<name>` |

Rules of thumb:
1. Anything evaluated *during* the run must be Brian2 code (strings) or a C++ function
   (see `stop_network` for the pattern). Python functions are only allowed outside the run.
2. Constants used inside model strings must be in the group's `namespace=dict(...)` or
   be literal numbers. The run is called with `namespace={}`, so Python variables are
   *not* picked up automatically.
3. After a change, run a 10k test pass and compare `run_info.json`
   (spikes/example, retries) and the accuracy with a previous run.

To debug the generated C++, look in `<build-dir>/code_objects/`. One file per Brian2
operation, named after the object (`controller_step`, `syn_input_pre`, …).

---

## 7. Differences from the earlier scripts, and limitations

- **θ decay in training** follows Brian1 (decays continuously); the earlier Brian2 port
  paused it during refractoriness. With τ = 10⁷ ms this makes no measurable difference.
- **Retry cap.** The default follows the Brian2 port (10 / 100). Use `--max-retries -1`
  for the original unlimited behaviour.
- **Results are written only when the run finishes.** If the process is killed, nothing
  from that run is saved. For long trainings, split them: train one epoch, then continue
  with `--init-from-weights --weights-dir <previous>/weights`. Each continuation starts
  with fresh membrane potentials and new random delays.
- **Snapshots are taken by simulated time**, so "every 10,000 examples" lands slightly
  earlier when there were retries. The file name says how many examples were finished.
- **Memory.** All training images are stored as 64-bit rates (about 380 MB for 60,000
  images), plus spike records (tens of MB). Plan for about 1 GB.
- **Threads.** `--threads 2` was 2.7× slower than 1 thread here: the network is too small
  for parallel overhead to pay off. To use more cores, run independent jobs in parallel
  (label + test, several seeds, parameter sweeps).
- **Sharding a pass** with `--first-example` is close to, but not exactly, one long run:
  each shard starts from rest instead of from the previous example's leftover membrane
  potential.

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| nothing printed after `Compiling and running …` | normal in Jupyter/Spyder/VS Code interactive: the compiled program's output goes elsewhere. Open `<out-dir>/progress.txt`, or check Task Manager for `main.exe` (running) vs `cl.exe` (still compiling). Don't stop the run; results are only written at the end |
| `error: command 'cl.exe' failed` / compiler not found | no C++ compiler; see Setup |
| `XeAe.npy / theta_A.npy missing or not for n_e=…` | `--weights-dir` holds weights for another network size |
| `WARNING: only k of N examples finished` | `--max-retries` very large with `-1`, or very low input; raise the intensity |
| 0 Ai spikes in `run_info.json`, or thousands of Ae spikes per example | inhibition not working; check any edits to 4g |
| accuracy far below ~89–91 % | wrong `--conn-dir` (recurrent weights for another size) or changed timing |

## Visualising results (`visualize.py`)

`visualize.py` turns run folders into one self-contained `report.html` (open it in any
browser) plus the individual PNGs. Every section appears only if its inputs exist.

```bat
:: pretrained or trained network: label + test runs
python code/visualize.py --label-dir runs/label400 --test-dir runs/test400 ^
       --weights-dir stdp-mnist/weights --out reports/ne400

:: a training run
python code/visualize.py --train-dir runs/train400 --out reports/train400

:: compare runs (e.g. LIF vs EIF) in one table
python code/visualize.py --label-dir runs/label_lif --test-dir runs/test_lif ^
       --compare runs/test_eif:runs/label_eif --out reports/lif_vs_eif
```

| Section | Shows | Needs |
|---|---|---|
| Network | every receptive field labelled by digit, mean field per digit, θ and weight histograms | weights |
| Selectivity | neurons per digit, neuron × digit response heatmap, selectivity index | label run |
| Test performance | confusion matrix, accuracy per digit, accuracy vs confidence, misclassified images | test run (+ MNIST for the images) |
| Activity | spikes per image, retries, retries by digit | test run |
| Spike timing | rasters, first-spike latency, gap to the second neuron (near-simultaneous winners), neurons firing per image | test run with `--save-spikes` |
| Training | block accuracy, retries and spikes over training, how evenly neurons share spikes, θ and receptive fields over time | train run (`--save-train-activity` for the sharing plots) |

Add `--save-spikes` (and `--record-inh-spikes`) to a run to keep every spike time in
`spikes.npz`; for a 10,000-image test run that is a few MB.

## EIF regions (`regions.py`)

For a test run with frozen theta and EIF parameters (vd, Δ), puts every excitatory neuron
in a region by where its spike detector (−72 + θ mV) sits relative to the EIF runaway
point Vx (the unstable fixed point, computed from vd, Δ and E_L):

| Region | m = (detector − Vx)/Δ | Meaning |
|---|---|---|
| A | m < 0 | detector below Vx: spikes like a LIF neuron |
| B | 0 ≤ m < 3 | detector just above Vx: the runaway decides the spike |
| C | m ≥ 3 (`--c-boundary`) | detector far above Vx: θ no longer restrains the neuron |

It then relates each digit's neurons (share in A/B/C) to that digit's accuracy, false
positives and confusions, and to the change against a baseline (LIF) run.

```bat
python code/regions.py --label-dir runs/eif_label --test-dir runs/eif_test ^
       --baseline-label-dir runs/lif_label --baseline-test-dir runs/lif_test ^
       --weights-dir stdp-mnist/weights --out reports/regions_vx32
```

Where Vx comes from depends on the EIF run's `--vd-mode` (read from its `run_info.json`):

| `--vd-mode` | Vx |
|---|---|
| `fixed` | one Vx for all neurons, from `--vd-e-mv` and `--delta-e-mv` |
| `track` | per neuron, from vd = vth − k |
| `homeo`, `proposal` | per neuron, Vx = vth − s, so m = s/Δ. s at the **start** of the test run (the trained `s_A.npy` for `--s-init trained`, or the constant given) or at the **end** (`s_end.npy`): `--s-at end` (default) or `--s-at start` |

`--vd-mv` / `--delta-mv` override the run's values (one fixed vd for all neurons), e.g. to
see where a LIF run's neurons would sit for a given vd. For runs with `--s-mode adaptive`
the report adds a section on how s moved during the test: s at start against s at end
per neuron, the change in s against spikes fired, region transitions (A → B, …) and the
mean change per digit.

Outputs `report.html`, `neurons.csv` (one row per neuron: digit, θ, detector, Vx, m,
region, s at start and end, selectivity, test spikes, times most active) and
`regions.json`. Use the same test images (same `--n-examples`, `--first-example`) for the
run and the baseline so the per-image comparison works. `regions.py` needs
`visualize.py` and `dc_standalone.py` in the same folder.

## EIF neurons (`EIF.py`)

`EIF.py` is `dc_standalone.py` with exponential integrate-and-fire excitatory neurons
(the inhibitory neurons stay LIF):

```
tau_e dv/dt = (E_L − v) + Δ·exp((v − vd)/Δ) + ge·(−v) + gi·(E_inh − v)
spike when v > vth = v_thresh + θ − offset   (= −72 + θ mV, Diehl & Cook threshold)
```

Δ = `--delta-e-mv`. The **runaway point Vx** is where the neuron starts to rise to a spike
on its own; vd is set so that the runaway happens there:
vd = Vx − Δ·ln((Vx − E_L)/Δ). Vx can never be closer to E_L than Δ.

### How vd is chosen: `--vd-mode`

| Mode | vd / Vx | Adapts? |
|---|---|---|
| `fixed` (default) | vd = `--vd-e-mv` for every neuron. `--vd-e-mv 100` switches the exponential off (= LIF) | no |
| `track` | vd = vth − k for each neuron, k = `--vd-k-mv`. vd follows the neuron's own θ | only through θ |
| `homeo` | each neuron has its own **s = vth − Vx**, adapted homeostatically (below) | yes, in training; option at test |
| `proposal` | the first δ1/δ2/δ3 rule set, kept for reproduction only | yes |

**s:** s < 0 → region A (Vx above vth: spike counted before the runaway, LIF-like);
s > 0 → region B (Vx below vth: once v passes Vx the spike is certain); larger s = more
excitable.

### The homeostatic rule (`--vd-mode homeo`)

Every 0.5 ms step, for each excitatory neuron:

1. s drifts up at a·r0 (slowly more excitable); paused while refractory.
2. On the neuron's own spike: s −= a (less excitable). θ += 0.05 mV as before.
3. s is clipped to [smin, min(smax, δ1 − Δ − margin)] mV, with δ1 = vth − V_reset = θ − 7 mV.
   The upper limit keeps Vx at least Δ + margin above E_L.

Over time each neuron fires close to r0: firing more often pushes s down (toward A),
firing less often lets it rise (toward B). Parameters, `--homeo` (comma-separated):

| Key | Default | Meaning |
|---|---|---|
| `a` | 0.02 | change of s per own spike (mV). Keep it small: large steps make many neurons excitable at once |
| `r0` | 0.0735 | target rate (Hz): 14.7 spikes per image ÷ 400 neurons ÷ 0.5 s |
| `smin`, `smax` | −1.5, 2.0 | band for s (mV); −3Δ … +4Δ at Δ = 0.5 |
| `margin` | 1 | extra distance kept from the physical limit (mV) |

### Starting value and adaptation: `--s-init`, `--s-mode`

| Option | Values | Default |
|---|---|---|
| `--s-init` | a number in mV (`0`, `-1`, `-1.5`, `2`, …) for every neuron, or `trained` = each neuron's value from `<weights-dir>/s_A.npy` | train: `0` (homeo), `-25` (proposal); label/test: `trained` |
| `--s-mode` | `adaptive` (s keeps changing during the run) or `frozen` | train: `adaptive`; label/test: `frozen` |

Starting values are clipped to the band (with a warning). What the starting value means
at Δ = 0.5 mV:

| `--s-init` | Start | Notes |
|---|---|---|
| `-1.5` | bottom of the band, deepest in A | most LIF-like; untested |
| `-1` | region A | used in `results/` (E3) |
| `0` | A/B boundary | used in `results/` (E3) |
| `2` | top of the band, all neurons in B | untested and risky: all neurons maximally excitable at once Try a short run first |

`--s-init` matters mostly early in training: moving s by 1 mV takes about 50 net spikes,
i.e. a few thousand images per neuron.

### Train, then test with frozen or adaptive s

```bat
:: train: s adapts, starting at -1 mV
python code/EIF.py --mode train --n-e 400 --epochs 1 --seed 1 ^
       --vd-mode homeo --delta-e-mv 0.5 --s-init -1 --out-dir C:\dc_runs\h_train

:: option 1: each neuron's trained s, frozen
python code/EIF.py --mode label --vd-mode homeo --delta-e-mv 0.5 ^
       --weights-dir C:\dc_runs\h_train\weights --conn-dir C:\dc_runs\h_train\connections ^
       --s-init trained --s-mode frozen --out-dir C:\dc_runs\h_label_frozen
python code/EIF.py --mode test  --vd-mode homeo --delta-e-mv 0.5 ^
       --weights-dir C:\dc_runs\h_train\weights --conn-dir C:\dc_runs\h_train\connections ^
       --s-init trained --s-mode frozen --out-dir C:\dc_runs\h_test_frozen

:: option 2: start from the trained s, keep adapting  (same flags, --s-mode adaptive)
:: control:  every neuron s = 0, frozen               (same flags, --s-init 0)
```

The label and test runs of one comparison must use the same `--vd-mode`, `--delta-e-mv`,
`--homeo`, `--s-init` and `--s-mode`; `evaluate.py` warns when they differ. Each run
starts from the trained s; in option 2 the label and test runs drift separately, and
results depend on image order (use 2 seeds and compare). Option 2 adapts without labels,
but it is a different model from option 1, so report which one a number comes from.

### What the homeo runs write

| File | When | Content |
|---|---|---|
| `weights/s_A.npy` | train | each neuron's s at the end (volts, like `theta_A.npy`) |
| `weights/s_A_exN.npy` | train, with snapshots | s at each snapshot |
| `s_end.npy` | label/test | s at the end of the run (= start when frozen) |
| `run_info.json` → `s_start`, `s_end` | every run | neurons in region A and B, at the floor and at the ceiling, s min/max/mean |

Every run also prints these counts at the start and end. Many neurons at the ceiling
during a run is the warning sign for the synchronous lock-up.

### Evaluation

`evaluate.py` works as before and also prints and saves (in `evaluation.json`) each
run's neuron settings, the s counts at start and end, spikes per image and retries,
the median confidence margin, and any setting that differs between the label and test
runs (`label_test_mismatch`).

### Notes

- Use `--method euler` (default) for EIF. `exponential_euler` treats the exponential
  term as constant over a step and spikes up to 1.6 ms late at dt = 0.5 ms.
- At dt = 0.5 ms Euler is unstable once the inhibitory conductance gi exceeds about 400
  (many inhibitory neurons firing in the same step). The network then locks up with
  every excitatory neuron firing at its maximum rate. Confirm timing results at
  `--dt-ms 0.05`.
