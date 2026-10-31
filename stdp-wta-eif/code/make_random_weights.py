#!/usr/bin/env python3
"""Initial and recurrent weights for a network of any size, with Diehl & Cook's rules.

Python 3 version of Diehl&Cook_MNIST_random_conn_generator.py from
https://github.com/peter-u-diehl/stdp-mnist, writing the same (i, j, w) format:

    XeAe  input -> excitatory, all-to-all, (rand + 0.01) * 0.3
    AeAi  excitatory -> inhibitory, one-to-one, 10.4
    AiAe  inhibitory -> excitatory, all-to-all except self, 17.0

XeAi (input -> inhibitory) is not written: the network does not use it.

    python make_random_weights.py --n-e 400 --out random/ --seed 0

dc_standalone.py and EIF.py generate any missing file with the same rules, so this
script is only needed to keep one fixed set of initial weights across runs. To repeat
my training runs exactly, use the original random/ folder of the repository above.
"""
import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-e', type=int, default=400, help='excitatory (= inhibitory) neurons')
    p.add_argument('--n-input', type=int, default=784)
    p.add_argument('--out', default='random/')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--w-input', type=float, default=0.3)
    p.add_argument('--w-ei', type=float, default=10.4)
    p.add_argument('--w-ie', type=float, default=17.0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    n_in, n_e = a.n_input, a.n_e

    w = (rng.random((n_in, n_e)) + 0.01) * a.w_input
    np.save(os.path.join(a.out, 'XeAe.npy'),
            np.array([(i, j, w[i, j]) for j in range(n_e) for i in range(n_in)]))
    np.save(os.path.join(a.out, 'AeAi.npy'), np.array([(i, i, a.w_ei) for i in range(n_e)]))
    w = np.full((n_e, n_e), a.w_ie)
    np.fill_diagonal(w, 0)
    np.save(os.path.join(a.out, 'AiAe.npy'),
            np.array([(i, j, w[i, j]) for i in range(n_e) for j in range(n_e)]))
    print(f'Saved XeAe, AeAi, AiAe for {n_e} neurons to {a.out}')


if __name__ == '__main__':
    main()
