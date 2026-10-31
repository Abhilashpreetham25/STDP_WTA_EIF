#!/usr/bin/env python3
"""
Diehl & Cook (2015) unsupervised STDP network for MNIST
Brian2 C++ STANDALONE version, with every run setting exposed as a parameter.

The whole experiment (image schedule, the low-activity retry rule, weight
normalisation, STDP) runs inside one compiled C++ program. Python only
prepares the inputs, builds the network, and post-processes the results.

Quick start (see README.md for the full guide):
    python dc_standalone.py --mode test  --n-e 400 --weights-dir stdp-mnist/weights
    python dc_standalone.py --mode label --n-e 400 --weights-dir stdp-mnist/weights
    python dc_standalone.py --mode train --n-e 400 --epochs 3 --out-dir runs/train400
    python evaluate.py --label-dir runs/label --test-dir runs/test

Sections in this file
    1. PARAMETERS        command line / JSON config
    2. DATA              MNIST loading
    3. WEIGHTS           loading, generating and saving connection matrices
    4. NETWORK           neurons, synapses, the in-network controller
    5. RUN               compile + run the C++ program
    6. RESULTS           spike counts per example, labels, weights, run_info.json
"""
import argparse
import gzip
import json
import math
import os
import pickle
import sys
import time

import numpy as np
# Windows consoles often use cp1252, which cannot print characters such as '−'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass


# =============================================================================
# 1. PARAMETERS
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description='Diehl & Cook (2015) STDP network, Brian2 C++ standalone',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, default=None,
                   help='JSON file with parameter values (keys = option names with '
                        'underscores). Command-line options override it.')

    g = p.add_argument_group('run')
    g.add_argument('--mode', choices=['train', 'label', 'test'], default='test',
                   help='train: STDP on, training images. label: weights frozen, '
                        'training images (to assign neurons to digits). test: weights '
                        'frozen, test images.')
    g.add_argument('--n-examples', type=int, default=None,
                   help='examples to process. Default: test 10000, label 60000, '
                        'train 60000*epochs')
    g.add_argument('--epochs', type=int, default=3, help='train mode only')
    g.add_argument('--first-example', type=int, default=0,
                   help='index of the first image (lets you split a pass into shards)')
    g.add_argument('--seed', type=int, default=0,
                   help='random seed (Poisson input, delays, generated weights)')

    g = p.add_argument_group('paths')
    g.add_argument('--mnist-dir', type=str, default='brain_env/Lib/site-packages/mnist/',
                   help='folder with training.pickle/testing.pickle or the MNIST idx files')
    g.add_argument('--weights-dir', type=str, default='stdp-mnist/weights/',
                   help='trained XeAe.npy + theta_A.npy (used by label/test, and by train '
                        'if --init-from-weights is set)')
    g.add_argument('--conn-dir', type=str, default='stdp-mnist/random/',
                   help='initial/recurrent matrices XeAe.npy, AeAi.npy, AiAe.npy. Files '
                        'that are missing or have the wrong size are generated.')
    g.add_argument('--init-from-weights', action='store_true',
                   help='train mode: start from --weights-dir instead of random weights '
                        '(continue training)')
    g.add_argument('--out-dir', type=str, default=None,
                   help='where results go. Default: runs/<mode>_ne<N>')
    g.add_argument('--build-dir', type=str, default=None,
                   help='C++ build folder. Default: <out-dir>/build. Re-using it skips '
                        'recompiling unchanged files.')

    g = p.add_argument_group('network size')
    g.add_argument('--n-e', type=int, default=400,
                   help='excitatory neurons (= inhibitory neurons)')

    g = p.add_argument_group('timing')
    g.add_argument('--dt-ms', type=float, default=0.5, help='simulation time step')
    g.add_argument('--method', choices=['euler', 'exponential_euler'], default='euler',
                   help='integration method for the neuron equations. euler = original')
    g.add_argument('--stim-ms', type=float, default=350.0, help='image presentation time')
    g.add_argument('--rest-ms', type=float, default=150.0, help='rest time after each image')
    g.add_argument('--max-delay-ms', type=float, default=10.0,
                   help='input->excitatory delays are uniform in [0, max)')

    g = p.add_argument_group('input and retry rule')
    g.add_argument('--start-intensity', type=float, default=2.0,
                   help='rate = pixel / rate-divisor * intensity (Hz)')
    g.add_argument('--rate-divisor', type=float, default=8.0)
    g.add_argument('--intensity-step', type=float, default=1.0,
                   help='added to the intensity on every retry')
    g.add_argument('--min-spikes', type=int, default=5,
                   help='retry the example if Ae fires fewer spikes than this')
    g.add_argument('--max-retries', type=int, default=None,
                   help='max retries per example. Default: 10 (test/label), 100 (train), '
                        'as in the Brian2 port. -1 = unlimited (Brian1 behaviour)')
    g.add_argument('--count-window', choices=['cycle', 'stim'], default='cycle',
                   help="cycle: count spikes since the previous stimulus ended, i.e. "
                        "including the preceding rest (original code). stim: count only "
                        "during the stimulus")

    g = p.add_argument_group('excitatory neurons')
    g.add_argument('--tau-e-ms', type=float, default=100.0)
    g.add_argument('--v-rest-e-mv', type=float, default=-65.0)
    g.add_argument('--v-reset-e-mv', type=float, default=-65.0)
    g.add_argument('--v-thresh-e-mv', type=float, default=-52.0)
    g.add_argument('--theta-offset-mv', type=float, default=20.0,
                   help='threshold = v_thresh + theta - offset')
    g.add_argument('--refrac-e-ms', type=float, default=5.0)
    g.add_argument('--e-inh-e-mv', type=float, default=-100.0,
                   help='inhibitory reversal potential of excitatory neurons')

    g = p.add_argument_group('inhibitory neurons')
    g.add_argument('--tau-i-ms', type=float, default=10.0)
    g.add_argument('--v-rest-i-mv', type=float, default=-60.0)
    g.add_argument('--v-reset-i-mv', type=float, default=-45.0)
    g.add_argument('--v-thresh-i-mv', type=float, default=-40.0)
    g.add_argument('--refrac-i-ms', type=float, default=2.0)
    g.add_argument('--e-inh-i-mv', type=float, default=-85.0)

    g = p.add_argument_group('synapses')
    g.add_argument('--tau-ge-ms', type=float, default=1.0)
    g.add_argument('--tau-gi-ms', type=float, default=2.0)
    g.add_argument('--w-ei', type=float, default=10.4,
                   help='Ae->Ai weight (one-to-one), used when generating AeAi')
    g.add_argument('--w-ie', type=float, default=17.0,
                   help='Ai->Ae weight (all-to-all except self), used when generating AiAe')
    g.add_argument('--w-input-init', type=float, default=0.3,
                   help='generated XeAe = (rand + 0.01) * this')

    g = p.add_argument_group('learning (train mode)')
    g.add_argument('--nu-pre', type=float, default=0.0001)
    g.add_argument('--nu-post', type=float, default=0.01)
    g.add_argument('--tc-pre-ms', type=float, default=20.0)
    g.add_argument('--tc-post1-ms', type=float, default=20.0)
    g.add_argument('--tc-post2-ms', type=float, default=40.0)
    g.add_argument('--wmax', type=float, default=1.0)
    g.add_argument('--w-total', type=float, default=78.0,
                   help='each column of XeAe is normalised to this sum before every image')
    g.add_argument('--theta-plus-mv', type=float, default=0.05)
    g.add_argument('--tc-theta-ms', type=float, default=1e7)
    g.add_argument('--theta-init-mv', type=float, default=20.0)

    g = p.add_argument_group('outputs')
    g.add_argument('--snapshot-every', type=int, default=10000,
                   help='train mode: save weights/theta about every N examples (0 = off)')
    g.add_argument('--perf-block', type=int, default=10000,
                   help='train mode: report accuracy per block of N examples')
    g.add_argument('--save-train-activity', action='store_true',
                   help='train mode: also save the per-example spike counts')
    g.add_argument('--report-every', type=int, default=1000,
                   help='print progress every N finished examples (0 = off)')
    g.add_argument('--record-inh-spikes', action='store_true',
                   help='store every inhibitory spike (otherwise only counts)')
    g.add_argument('--save-spikes', action='store_true',
                   help='save every excitatory spike (neuron, time) to spikes.npz, for '
                        'rasters and first-spike latency in visualize.py')

    g = p.add_argument_group('performance')
    g.add_argument('--threads', type=int, default=0,
                   help='OpenMP threads (0 = off; usually slower for this network size)')
    return p


def get_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            cfg = json.load(f)
        known = {a.dest for a in parser._actions}
        unknown = set(cfg) - known
        if unknown:
            parser.error(f'unknown keys in {pre.config}: {sorted(unknown)}')
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)

    if args.n_examples is None:
        args.n_examples = {'test': 10000, 'label': 60000,
                           'train': 60000 * args.epochs}[args.mode]
    if args.max_retries is None:
        args.max_retries = 100 if args.mode == 'train' else 10
    if args.out_dir is None:
        args.out_dir = os.path.join('runs', f'{args.mode}_ne{args.n_e}')
    if args.build_dir is None:
        args.build_dir = os.path.join(args.out_dir, 'build')

    # the schedule is counted in time steps, so the durations must be whole steps
    for name in ['stim_ms', 'rest_ms']:
        steps = getattr(args, name) / args.dt_ms
        if abs(steps - round(steps)) > 1e-9:
            parser.error(f'--{name.replace("_", "-")} must be a multiple of --dt-ms')
    return args


# =============================================================================
# 2. DATA
# =============================================================================
def _read_idx(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        raw = f.read()
    ndim = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * k: 8 + 4 * k], 'big') for k in range(ndim)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(mnist_dir, train):
    """Returns (images uint8 [N,784], labels int [N]). Accepts the pickles made by the
    original script (training.pickle / testing.pickle) or the raw idx files."""
    name = 'training' if train else 'testing'
    pkl = os.path.join(mnist_dir, name + '.pickle')
    if os.path.isfile(pkl):
        with open(pkl, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        return d['x'].reshape(len(d['x']), -1), np.asarray(d['y']).reshape(-1).astype(int)
    stem = 'train' if train else 't10k'
    for img, lab in [(f'{stem}-images-idx3-ubyte', f'{stem}-labels-idx1-ubyte'),
                     (f'{stem}-images.idx3-ubyte', f'{stem}-labels.idx1-ubyte')]:
        for ext in ['', '.gz']:
            ip, lp = os.path.join(mnist_dir, img + ext), os.path.join(mnist_dir, lab + ext)
            if os.path.isfile(ip) and os.path.isfile(lp):
                x = _read_idx(ip)
                return x.reshape(len(x), -1), _read_idx(lp).astype(int)
    sys.exit(f'MNIST {name} set not found in {mnist_dir}')


# =============================================================================
# 3. WEIGHTS
# =============================================================================
def load_matrix(path, n_src, n_tgt):
    """Original format: array of rows (i, j, w). Returns a dense [n_src, n_tgt] matrix,
    or None if the file is missing or does not fit the requested size."""
    if not os.path.isfile(path):
        return None
    r = np.load(path, allow_pickle=True, encoding='latin1')
    if r.size == 0:
        return np.zeros((n_src, n_tgt))
    i, j = r[:, 0].astype(int), r[:, 1].astype(int)
    # The file must be for exactly this size. A smaller network's file (e.g. the
    # 400-neuron AiAe.npy used with --n-e 6400) would otherwise load into one
    # corner of the matrix and leave most neurons without lateral inhibition.
    if i.max() != n_src - 1 or j.max() != n_tgt - 1:
        return None
    if path.endswith('XeAe.npy') and len(r) != n_src * n_tgt:
        return None
    m = np.zeros((n_src, n_tgt))
    m[i, j] = r[:, 2]
    return m


def save_matrix(path, m):
    """Same (i, j, w) list format as the original save_connections()."""
    i, j = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing='ij')
    np.save(path, np.column_stack([i.ravel(), j.ravel(), m.ravel()]))


def get_weights(args, n_input, rng):
    n_e = args.n_e
    info = {}

    def from_conn_dir(name, n_src, n_tgt, generator):
        m = load_matrix(os.path.join(args.conn_dir, name + '.npy'), n_src, n_tgt)
        if m is None:
            m = generator()
            info[name] = 'generated'
        else:
            info[name] = os.path.join(args.conn_dir, name + '.npy')
        return m

    # Ae -> Ai: one-to-one;  Ai -> Ae: all-to-all except self (Diehl's generator)
    w_ei = from_conn_dir('AeAi', n_e, n_e, lambda: np.eye(n_e) * args.w_ei)
    w_ie = from_conn_dir('AiAe', n_e, n_e,
                         lambda: (np.ones((n_e, n_e)) - np.eye(n_e)) * args.w_ie)

    use_trained = args.mode in ('label', 'test') or args.init_from_weights
    if use_trained:
        w_in = load_matrix(os.path.join(args.weights_dir, 'XeAe.npy'), n_input, n_e)
        theta_file = os.path.join(args.weights_dir, 'theta_A.npy')
        if w_in is None or not os.path.isfile(theta_file):
            sys.exit(f'{args.weights_dir}: XeAe.npy / theta_A.npy missing or not for '
                     f'n_e={n_e}')
        theta = np.load(theta_file)          # volts, as saved by the original code
        if theta.shape != (n_e,):
            sys.exit(f'theta_A.npy has {theta.shape[0]} neurons, expected {n_e}')
        info['XeAe'] = os.path.join(args.weights_dir, 'XeAe.npy')
        info['theta'] = theta_file
    else:
        w_in = from_conn_dir('XeAe', n_input, n_e,
                             lambda: (rng.random((n_input, n_e)) + 0.01) * args.w_input_init)
        theta = np.ones(n_e) * args.theta_init_mv * 1e-3
        info['theta'] = f'constant {args.theta_init_mv} mV'
    return w_in, w_ei, w_ie, theta, info


# =============================================================================
# 4. NETWORK
# =============================================================================
def build_network(args, images, b2):
    """images: float array [n_images, 784] with the rate of each pixel in Hz at
    intensity 1 (= pixel / rate_divisor). Returns a dict of Brian2 objects."""
    from brian2 import (NeuronGroup, Synapses, SpikeMonitor, EventMonitor, StateMonitor,
                        TimedArray, linked_var, implementation, check_units,
                        defaultclock, ms, mV, Hz, volt)

    train = args.mode == 'train'
    n_e, n_input = args.n_e, images.shape[1]
    dt = args.dt_ms * ms
    defaultclock.dt = dt
    STIM = int(round(args.stim_ms / args.dt_ms))
    REST = int(round(args.rest_ms / args.dt_ms))
    CYCLE = STIM + REST

    # ---- 4a. stop function: ends the C++ run loop after the last example ----------
    @implementation('cpp', '''
    double stop_network(double flag) {
        if (flag > 0.5) Network::_globally_stopped = true;
        return 0.0;
    }''', headers=['"network.h"'])
    @check_units(flag=1, result=1)
    def stop_network(flag):
        raise NotImplementedError('only available in C++ standalone mode')

    # progress line every REPORT_EVERY finished examples. Printed by the C++ program AND
    # appended to <out-dir>/progress.txt (IDE consoles often do not show the program output)
    progress_file = os.path.abspath(os.path.join(args.out_dir, 'progress.txt')).replace('\\', '/')
    with open(progress_file, 'w') as f:
        f.write('compiling...\n')
    @implementation('cpp', '''
    double report_progress(double done, double total, double every, double pres, double tsim) {
        static std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        static bool started = false;
        long d = (long)done;
        char line[400];
        line[0] = 0;
        if (!started) {
            started = true;
    #if defined(_M_X64) || defined(_M_IX86) || defined(__SSE__)
            // flush-to-zero + denormals-are-zero: values below ~1e-308 become 0.
            // Avoids very slow denormal arithmetic (notably with MSVC on Windows).
            _mm_setcsr(_mm_getcsr() | 0x8040);
    #endif
            std::snprintf(line, sizeof(line), "simulation started (compilation finished)\\n");
        } else if (every > 0.5 && d > 0 && d % (long)every == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::snprintf(line, sizeof(line),
                          "  [%ld/%ld examples] elapsed %.1f min, ETA %.1f min | presentations %ld "
                          "(retries %ld), simulated %.0f s, %.2f wall-s per simulated-s\\n",
                          d, (long)total, el / 60.0, el / d * (total - d) / 60.0, (long)pres,
                          (long)pres - d, tsim, el / (tsim > 0 ? tsim : 1));
        }
        if (line[0]) {
            std::printf("%s", line);
            std::fflush(stdout);
            FILE* f = std::fopen("PROGRESS_FILE", "a");
            if (f) { std::fputs(line, f); std::fclose(f); }
        }
        return 0.0;
    }'''.replace('PROGRESS_FILE', progress_file), headers=['<chrono>', '<cstdio>', '<xmmintrin.h>'])
    @check_units(done=1, total=1, every=1, pres=1, tsim=1, result=1)
    def report_progress(done, total, every, pres, tsim):
        raise NotImplementedError('only available in C++ standalone mode')

    # ---- 4b. controller: one "neuron" that runs the experiment schedule ---------
    # Every presentation is STIM steps of input followed by REST steps of silence.
    # At the first rest step the controller decides: accept the example, or retry it
    # with a higher intensity (fewer than MIN_SPIKES spikes and retries left).
    ctrl = NeuronGroup(1, '''
        cs : integer              # step inside the current presentation (0..CYCLE-1)
        idx : integer             # current example (0..N-1); N when all are done
        retry : integer           # retries used for the current example
        intensity : 1             # current input intensity
        total : integer           # Ae spikes counted for the current example
        stim : integer            # 1 while an image is shown
        accept : integer          # 1 on the step an example is accepted
        acc_idx : integer         # values recorded when an example is accepted
        acc_count : integer
        acc_intensity : 1
        acc_retries : integer
        presentations : integer   # images shown so far, retries included
        dummy : 1
        ''', events={'accepted': 'accept == 1'}, name='controller',
        namespace=dict(STIM=STIM, CYCLE=CYCLE, MIN_SPIKES=args.min_spikes,
                       MAX_RETRIES=args.max_retries if args.max_retries >= 0 else 2**30,
                       INTENSITY_STEP=args.intensity_step,
                       START_INTENSITY=args.start_intensity,
                       N_EXAMPLES=args.n_examples, REPORT_EVERY=args.report_every,
                       stop_network=stop_network, report_progress=report_progress))
    ctrl.intensity = args.start_intensity
    reset_total = 'end_stim' if args.count_window == 'cycle' else 'int(cs == 0)'
    ctrl.run_regularly(f'''
        end_stim = int(cs == STIM)
        fail = end_stim * int(total < MIN_SPIKES) * int(retry < MAX_RETRIES)
        accept = end_stim - fail
        acc_idx = idx
        acc_count = total
        acc_intensity = intensity
        acc_retries = retry
        idx = idx + accept
        retry = (retry + fail) * (1 - accept)
        intensity = intensity + fail * INTENSITY_STEP + accept * (START_INTENSITY - intensity)
        total = total * (1 - {reset_total})
        stim = int(cs < STIM) * int(idx < N_EXAMPLES)
        presentations = presentations + int(cs == 0) * int(idx < N_EXAMPLES)
        dummy = report_progress(idx * accept, N_EXAMPLES, REPORT_EVERY, presentations, t / second)
        dummy = stop_network(int(cs == CYCLE - 1) * int(idx >= N_EXAMPLES))
        cs = (cs + 1) % CYCLE
        ''', dt=dt, when='start', order=0, name='controller_step')

    def link(group, n, var):
        setattr(group, var + '_', linked_var(ctrl, var, index=np.zeros(n, dtype=int)))

    # ---- 4c. input layer: Poisson neurons, rate = image pixel * intensity --------
    img = TimedArray(images * Hz, dt=1 * ms)      # row k = image k (looked up by "time")
    X = NeuronGroup(n_input, '''
        stim_ : integer (linked)
        idx_ : integer (linked)
        intensity_ : 1 (linked)
        ''', threshold='rand() < stim_ * img(((FIRST + idx_) % N_IMAGES) * ms, i) '
                       '* intensity_ * dt',
        namespace=dict(img=img, FIRST=args.first_example, N_IMAGES=len(images)),
        name='input')
    for v in ['stim', 'idx', 'intensity']:
        link(X, n_input, v)

    # ---- 4d. excitatory and inhibitory neurons ----------------------------------
    p = dict(v_rest_e=args.v_rest_e_mv * mV, v_reset_e=args.v_reset_e_mv * mV,
             v_thresh_e=args.v_thresh_e_mv * mV, offset=args.theta_offset_mv * mV,
             tau_e=args.tau_e_ms * ms, E_inh_e=args.e_inh_e_mv * mV,
             v_rest_i=args.v_rest_i_mv * mV, v_reset_i=args.v_reset_i_mv * mV,
             v_thresh_i=args.v_thresh_i_mv * mV, tau_i=args.tau_i_ms * ms,
             E_inh_i=args.e_inh_i_mv * mV, tau_ge=args.tau_ge_ms * ms,
             tau_gi=args.tau_gi_ms * ms, tc_theta=args.tc_theta_ms * ms,
             theta_plus=args.theta_plus_mv * mV)
    theta_eq = 'dtheta/dt = -theta / tc_theta : volt' if train else 'theta : volt'
    E = NeuronGroup(n_e, f'''
        dv/dt = ((v_rest_e - v) + ge * (-v) + gi * (E_inh_e - v)) / tau_e : volt (unless refractory)
        dge/dt = -ge / tau_ge : 1
        dgi/dt = -gi / tau_gi : 1
        {theta_eq}
        colsum : 1                # column sum of input weights (normalisation)
        ''', threshold='v > theta - offset + v_thresh_e',
        reset='v = v_reset_e' + ('; theta += theta_plus' if train else ''),
        refractory=args.refrac_e_ms * ms, method=args.method, namespace=p, name='exc')
    I = NeuronGroup(n_e, '''
        dv/dt = ((v_rest_i - v) + ge * (-v) + gi * (E_inh_i - v)) / tau_i : volt (unless refractory)
        dge/dt = -ge / tau_ge : 1
        dgi/dt = -gi / tau_gi : 1
        ''', threshold='v > v_thresh_i', reset='v = v_reset_i',
        refractory=args.refrac_i_ms * ms, method=args.method, namespace=p, name='inh')
    E.v = args.v_rest_e_mv * mV - 40 * mV          # as in the original
    I.v = args.v_rest_i_mv * mV - 40 * mV
    return dict(ctrl=ctrl, X=X, E=E, I=I, STIM=STIM, REST=REST, CYCLE=CYCLE, p=p)


def add_synapses(args, net, w_in, w_ei, w_ie, theta):
    from brian2 import Synapses, ms, volt
    train = args.mode == 'train'
    ctrl, X, E, I, CYCLE = net['ctrl'], net['X'], net['E'], net['I'], net['CYCLE']
    E.theta = theta * volt

    # ---- 4e. input -> excitatory (STDP in train mode, static otherwise) ---------
    if train:
        S_in = Synapses(X, E, '''
            w : 1
            dpre/dt = -pre / tc_pre : 1 (event-driven)
            dpost1/dt = -post1 / tc_post1 : 1 (event-driven)
            dpost2/dt = -post2 / tc_post2 : 1 (event-driven)
            ''', on_pre='''
            ge_post += w
            pre = 1.
            w = clip(w - nu_pre * post1, 0, wmax)
            ''', on_post='''
            w = clip(w + nu_post * pre * post2, 0, wmax)
            post1 = 1.
            post2 = 1.
            ''', namespace=dict(tc_pre=args.tc_pre_ms * ms, tc_post1=args.tc_post1_ms * ms,
                                tc_post2=args.tc_post2_ms * ms, nu_pre=args.nu_pre,
                                nu_post=args.nu_post, wmax=args.wmax), name='syn_input')
        S_in.connect()                                   # all-to-all, i-major order
        S_in.w = w_in.ravel()
    else:
        S_in = Synapses(X, E, 'w : 1', on_pre='ge_post += w', name='syn_input')
        ii, jj = np.nonzero(w_in)                        # zero weights only add 0.0
        S_in.connect(i=ii, j=jj)
        S_in.w = w_in[ii, jj]
    S_in.delay = f'rand() * {args.max_delay_ms!r} * ms'

    # ---- 4f. normalisation before every image (train mode) ----------------------
    # Every presentation (retries included) starts at t = k * CYCLE, so running these
    # three steps every CYCLE at the start of the step normalises before each image.
    if train:
        dt_cycle = CYCLE * net['E'].clock.dt
        E.run_regularly('colsum = 0', dt=dt_cycle, when='start', order=1, name='norm_reset')
        S_in.run_regularly('colsum_post += w', dt=dt_cycle, when='start', order=2,
                           name='norm_sum')
        S_in.namespace['W_TOTAL'] = args.w_total
        S_in.run_regularly('w = w * W_TOTAL / (colsum_post + int(colsum_post == 0))',
                           dt=dt_cycle, when='start', order=3, name='norm_scale')

    # ---- 4g. recurrent connections ----------------------------------------------
    # The SOURCE population decides the sign: excitatory spikes raise ge, inhibitory gi.
    a, b = np.nonzero(w_ei)
    S_ei = Synapses(E, I, 'w : 1', on_pre='ge_post += w', name='syn_ei')
    S_ei.connect(i=a, j=b)
    S_ei.w = w_ei[a, b]
    a, b = np.nonzero(w_ie)
    S_ie = Synapses(I, E, 'w : 1', on_pre='gi_post += w', name='syn_ie')
    S_ie.connect(i=a, j=b)
    S_ie.w = w_ie[a, b]

    # ---- 4h. spike counter for the retry rule -----------------------------------
    S_cnt = Synapses(E, ctrl, on_pre='total_post += 1', name='syn_counter')
    S_cnt.connect(i=np.arange(args.n_e), j=0)
    net.update(S_in=S_in, S_ei=S_ei, S_ie=S_ie, S_cnt=S_cnt)


def add_monitors(args, net):
    from brian2 import SpikeMonitor, EventMonitor, StateMonitor
    net['mon_e'] = SpikeMonitor(net['E'], name='mon_exc')
    net['mon_i'] = SpikeMonitor(net['I'], record=args.record_inh_spikes, name='mon_inh')
    net['mon_acc'] = EventMonitor(net['ctrl'], 'accepted',
                                  variables=['acc_idx', 'acc_count', 'acc_intensity',
                                             'acc_retries'], name='mon_accepted')
    if args.mode == 'train' and args.snapshot_every > 0:
        dt_snap = args.snapshot_every * net['CYCLE'] * net['E'].clock.dt
        n_syn = net['X'].N * args.n_e                 # train mode: all-to-all
        net['snap_w'] = StateMonitor(net['S_in'], 'w', record=np.arange(n_syn), dt=dt_snap,
                                     name='snap_w')
        net['snap_theta'] = StateMonitor(net['E'], 'theta', record=True, dt=dt_snap,
                                         name='snap_theta')
        net['snap_idx'] = StateMonitor(net['ctrl'], 'idx', record=0, dt=dt_snap,
                                       name='snap_idx')


# =============================================================================
# 5. RUN
# =============================================================================
def run(args):
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    import brian2 as b2
    # Allow several runs (train -> label -> test) in one Python session or
    # Jupyter kernel: the standalone device can only be built once, so reset it.
    if getattr(b2.get_device(), 'has_been_run', False) or \
            getattr(b2.get_device(), 'build_on_run', None) is not None:
        b2.device.reinit()
    b2.set_device('cpp_standalone', directory=args.build_dir, build_on_run=False)
    if args.threads > 0:
        b2.prefs.devices.cpp_standalone.openmp_threads = args.threads
    b2.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # data: label/train use the training set, test the test set
    use_train_set = args.mode in ('train', 'label')
    x, y = load_mnist(args.mnist_dir, train=use_train_set)
    images = x.astype(np.float64) / args.rate_divisor

    w_in, w_ei, w_ie, theta, winfo = get_weights(args, x.shape[1], rng)
    cdir = os.path.join(args.out_dir, 'connections')      # exactly what this run used
    os.makedirs(cdir, exist_ok=True)
    save_matrix(os.path.join(cdir, 'AeAi.npy'), w_ei)
    save_matrix(os.path.join(cdir, 'AiAe.npy'), w_ie)
    if args.mode == 'train':
        save_matrix(os.path.join(cdir, 'XeAe.npy'), w_in)  # initial input weights
    net = build_network(args, images, b2)
    add_synapses(args, net, w_in, w_ei, w_ie, theta)
    add_monitors(args, net)

    # upper bound on simulated time; the controller stops the run when finished
    max_pres = args.n_examples * (1 + (args.max_retries if args.max_retries >= 0 else 1000))
    objs = [o for o in net.values() if isinstance(o, b2.BrianObject)]
    network = b2.Network(*objs)
    network.run(max_pres * net['CYCLE'] * args.dt_ms * b2.ms, report=None, namespace={})

    t_build = time.time()
    print(f'Compiling and running in {args.build_dir} ...', flush=True)
    b2.device.build(directory=args.build_dir, compile=True, run=True, debug=False)
    t_done = time.time()
    sim_time = b2.device._last_run_time
    print(f'Simulation finished: {sim_time:.1f} s of computation '
          f'(compile + run {t_done - t_build:.1f} s)')
    timing = dict(setup_s=t_build - t_start, compile_and_run_s=t_done - t_build,
                  simulation_s=sim_time)
    return net, x, y, w_in, winfo, timing


# =============================================================================
# 6. RESULTS
# =============================================================================
def spike_counts_per_example(args, net):
    """Rebuild the per-example spike counts the original code stored in result_monitor."""
    from brian2 import ms
    dt = args.dt_ms
    STIM, CYCLE = net['STIM'], net['CYCLE']
    acc = net['mon_acc']
    end_step = np.round(np.asarray(acc.t / ms) / dt).astype(np.int64)  # first rest step
    if args.count_window == 'cycle':
        start_step = np.maximum(end_step - CYCLE, 0)
    else:
        start_step = end_step - STIM
    ex_idx = np.asarray(acc.acc_idx).astype(int)

    sp_step = np.round(np.asarray(net['mon_e'].t / ms) / dt).astype(np.int64)
    sp_i = np.asarray(net['mon_e'].i).astype(int)
    counts = np.zeros((args.n_examples, args.n_e), dtype=np.int32)
    k = np.searchsorted(end_step, sp_step, side='right')      # window that ends after spike
    ok = k < len(end_step)
    ok[ok] &= sp_step[ok] >= start_step[k[ok]]
    np.add.at(counts, (ex_idx[k[ok]], sp_i[ok]), 1)

    # consistency check against the controller's own counter
    rec = np.asarray(acc.acc_count).astype(int)
    if not np.array_equal(counts.sum(1)[ex_idx], rec):
        print('WARNING: post-processed spike counts differ from the controller counts')
    return counts, ex_idx


def assign_and_predict(train_counts, train_labels, counts, n_e):
    """Same rules as the original evaluation: each neuron is assigned the digit it
    responds to most; the prediction is the digit with the highest mean response."""
    assignments = -np.ones(n_e, dtype=int)
    max_rate = np.zeros(n_e)
    for c in range(10):
        m = train_labels == c
        if m.sum():
            rate = train_counts[m].sum(0) / m.sum()
            better = rate > max_rate
            assignments[better] = c
            max_rate[better] = rate[better]
    scores = np.zeros((len(counts), 10))
    for c in range(10):
        m = assignments == c
        if m.sum():
            scores[:, c] = counts[:, m].sum(1) / m.sum()
    # same tie rule as the original: np.argsort(...)[::-1][0]
    return np.argsort(scores, axis=1, kind='stable')[:, ::-1][:, 0], assignments


def save_results(args, net, x, y, w_in_initial, winfo, timing):
    from brian2 import ms, mV, volt, second
    out = args.out_dir
    counts, ex_idx = spike_counts_per_example(args, net)
    acc = net['mon_acc']
    done = len(ex_idx)
    labels = y[(args.first_example + np.arange(args.n_examples)) % len(y)]
    n = args.n_examples

    info = dict(args=vars(args), weights=winfo, timing=timing,
                examples_completed=int(done), examples_requested=int(n),
                presentations=int(net['ctrl'].presentations[0]),
                retries_total=int(np.asarray(acc.acc_retries).sum()),
                examples_hitting_retry_cap=int((np.asarray(acc.acc_retries)
                                                >= max(args.max_retries, 0)).sum())
                if args.max_retries >= 0 else 0,
                exc_spikes=int(net['mon_e'].num_spikes),
                inh_spikes=int(net['mon_i'].num_spikes),
                mean_exc_spikes_per_example=float(counts.sum(1).mean()) if n else 0.0,
                simulated_time_s=float(net['E'].clock.t / second))
    if done < n:
        print(f'WARNING: only {done} of {n} examples finished (time limit reached)')

    # one row per finished example: which image, retries needed, final intensity
    np.savez(os.path.join(out, 'examples.npz'),
             example=ex_idx, image=(args.first_example + ex_idx) % len(y),
             label=labels[ex_idx], retries=np.asarray(acc.acc_retries).astype(int),
             intensity=np.asarray(acc.acc_intensity), spikes=np.asarray(acc.acc_count),
             stim_end_s=np.asarray(acc.t / ms) / 1000)

    if args.mode in ('test', 'label'):
        act = os.path.join(out, 'activity')
        os.makedirs(act, exist_ok=True)
        np.save(os.path.join(act, f'resultPopVecs{n}.npy'), counts)
        np.save(os.path.join(act, f'inputNumbers{n}.npy'), labels)
        print(f'Saved spike counts {counts.shape} and labels to {act}')
    else:
        wdir = os.path.join(out, 'weights')
        os.makedirs(wdir, exist_ok=True)
        S = net['S_in']
        w = np.zeros_like(w_in_initial)
        w[np.asarray(S.i), np.asarray(S.j)] = np.asarray(S.w)
        save_matrix(os.path.join(wdir, 'XeAe.npy'), w)
        np.save(os.path.join(wdir, 'theta_A.npy'), np.asarray(net['E'].theta / volt))
        print(f'Saved trained weights and theta to {wdir}')
        if 'snap_w' in net:
            sw, st, si = net['snap_w'], net['snap_theta'], net['snap_idx']
            ex_at = np.asarray(si.idx[0]).astype(int)
            for k in range(1, len(ex_at)):            # k=0 is the initial state
                ws = np.zeros_like(w_in_initial)
                ws[np.asarray(S.i), np.asarray(S.j)] = np.asarray(sw.w[:, k])
                save_matrix(os.path.join(wdir, f'XeAe_ex{ex_at[k]}.npy'), ws)
                np.save(os.path.join(wdir, f'theta_A_ex{ex_at[k]}.npy'),
                        np.asarray(st.theta[:, k] / volt))
            info['snapshots_at_examples'] = ex_at[1:].tolist()
        # accuracy per block, as printed during the original training
        perf = []
        B = args.perf_block
        for b in range(1, n // B):
            prev, cur = slice((b - 1) * B, b * B), slice(b * B, (b + 1) * B)
            pred, _ = assign_and_predict(counts[prev], labels[prev], counts[cur], args.n_e)
            perf.append(float((pred == labels[cur]).mean() * 100))
        info['train_block_accuracy'] = perf
        if perf:
            print('Accuracy per block of', B, 'examples:', ' '.join(f'{v:.1f}' for v in perf))
        if args.save_train_activity:
            np.save(os.path.join(out, 'train_counts.npy'), counts.astype(np.int16))
            np.save(os.path.join(out, 'train_labels.npy'), labels)

    if args.save_spikes:
        sp = dict(e_i=np.asarray(net['mon_e'].i).astype(np.int16),
                  e_t_ms=np.asarray(net['mon_e'].t / ms).astype(np.float64))
        if args.record_inh_spikes:
            sp.update(i_i=np.asarray(net['mon_i'].i).astype(np.int16),
                      i_t_ms=np.asarray(net['mon_i'].t / ms).astype(np.float64))
        np.savez_compressed(os.path.join(out, 'spikes.npz'), stim_ms=args.stim_ms, **sp)
        print(f"Saved {len(sp['e_i'])} excitatory spike times to spikes.npz")

    with open(os.path.join(out, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"{done} examples, {info['presentations']} presentations "
          f"({info['retries_total']} retries), {info['mean_exc_spikes_per_example']:.1f} "
          f"Ae spikes/example, {info['inh_spikes']} Ai spikes")
    print(f"Wall time: setup {timing['setup_s']:.1f} s, compile+run "
          f"{timing['compile_and_run_s']:.1f} s (simulation {timing['simulation_s']:.1f} s)")
    return info


def main(argv=None):
    args = get_args(argv)
    print('Mode %s | n_e %d | %d examples | stim %g ms, rest %g ms, dt %g ms | '
          'retries <= %s, min spikes %d' % (args.mode, args.n_e, args.n_examples,
          args.stim_ms, args.rest_ms, args.dt_ms,
          'unlimited' if args.max_retries < 0 else args.max_retries, args.min_spikes))
    net, x, y, w_in, winfo, timing = run(args)
    save_results(args, net, x, y, w_in, winfo, timing)


if __name__ == '__main__':
    main()
