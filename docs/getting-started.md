# Getting started

## 1. Install

- Python 3.9 or newer, then `pip install -r requirements.txt` (brian2, numpy, matplotlib).
- A C++ compiler, because Brian2 standalone compiles the whole network:
  - Windows: Microsoft C++ Build Tools ("Desktop development with C++"). All runs in this repository were done this way.
  - Linux: `sudo apt install build-essential`.
  - macOS: `xcode-select --install`.

## 2. Get MNIST

The code reads the four standard MNIST files, gzipped or not:

```
mnist/train-images-idx3-ubyte.gz   mnist/train-labels-idx1-ubyte.gz
mnist/t10k-images-idx3-ubyte.gz    mnist/t10k-labels-idx1-ubyte.gz
```

They are on the [MNIST page](http://yann.lecun.com/exdb/mnist/) and on mirrors such as `https://ossci-datasets.s3.amazonaws.com/mnist/`. The `training.pickle` / `testing.pickle` files made by the original code also work. Pass the folder with `--mnist-dir mnist/`.

## 3. Initial weights

Training starts from random input weights and fixed recurrent weights:

| file | connection | rule |
|---|---|---|
| `XeAe.npy` | 784 input → excitatory | all-to-all, (rand + 0.01) × 0.3 |
| `AeAi.npy` | excitatory → inhibitory | one-to-one, 10.4 |
| `AiAe.npy` | inhibitory → excitatory | all-to-all except self, 17.0 |

There are three ways to get them:
- Do nothing. If `--conn-dir` has no file of the right size, the scripts generate it with these rules and save a copy in `<out-dir>/connections/`.
- Generate one fixed set for a network of any size: `python code/make_random_weights.py --n-e 400 --out random/ --seed 0`.
- Use the original `random/` folder from [stdp-mnist](https://github.com/peter-u-diehl/stdp-mnist). My training runs used it. The recurrent files are also in `weights/connections/`.

## 4. Reproduce my test results (minutes, no training)

`weights/` holds the trained LIF baseline and the homeostatic EIF network (s₀ = −1). A test pass takes about 4 min (LIF) or 7 min (EIF), plus 20–30 s to compile. The commands are in the main [README](../README.md#reproduce-the-test-results-without-training).

## 5. Train your own network

```bash
# LIF, 3 epochs (several hours)
python code/dc_standalone.py --mode train --n-e 400 --epochs 3 --conn-dir random/ \
       --mnist-dir mnist/ --out-dir runs/train400
# give every neuron a digit (60,000 training images, about 40 min)
python code/dc_standalone.py --mode label --weights-dir runs/train400/weights \
       --conn-dir runs/train400/connections --mnist-dir mnist/ --out-dir runs/label400
# test (10,000 images)
python code/dc_standalone.py --mode test --weights-dir runs/train400/weights \
       --conn-dir runs/train400/connections --mnist-dir mnist/ --out-dir runs/test400
python code/evaluate.py --label-dir runs/label400 --test-dir runs/test400
python code/visualize.py --label-dir runs/label400 --test-dir runs/test400 --out reports/test400
```

For EIF neurons, use `code/EIF.py` with the same options plus `--vd-mode homeo --delta-e-mv 0.5 --s-init -1`. Keep the same `--delta-e-mv` for training, labelling and testing. Before a long run, try `--n-examples 2000` to check that everything works.

## 6. Tips

- Results are written only when a run finishes. Progress is printed and also appended to `<out-dir>/progress.txt`, which helps in Jupyter, where the compiled program's output may not show.
- On Windows, keep run folders outside OneDrive. A locked `main.exe` makes the build fail with LNK1104.
- The label and test passes do not depend on each other, so they can run in two terminals at once.
- Every option is described in [usage.md](usage.md).
