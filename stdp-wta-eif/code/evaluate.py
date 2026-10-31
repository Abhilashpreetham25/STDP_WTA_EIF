#!/usr/bin/env python3
"""Accuracy from the spike counts saved by dc_standalone.py or EIF.py.

Paper protocol (assignments from the training set, as in Diehl & Cook 2015):
    python evaluate.py --label-dir runs/label_ne400 --test-dir runs/test_ne400
Demo protocol of the original repository (assignments from the test set itself;
optimistic, the original readme reports 91.56% for the pretrained weights):
    python evaluate.py --test-dir runs/test_ne400
Saved assignments (paper protocol without rerunning the 60k label pass):
    python evaluate.py --label-dir runs/label400 --test-dir runs/test400 --save-assignments weights/LIF_400/assignments.npy
    python evaluate.py --test-dir runs/test400 --assignments weights/LIF_400/assignments.npy

For EIF runs it also records the neuron settings of both runs (vd mode, Delta, k, s
settings) and, for --vd-mode homeo/proposal, how many neurons were in region A / B and
at the s limits at the start and end of each run. It warns when the label and test
runs used different settings, because the assignments would then come from a
different network than the one tested.
"""
import argparse, glob, json, os, sys
import numpy as np
# Windows consoles often use cp1252, which cannot print characters such as '−'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_standalone import assign_and_predict

# settings that must match between the label run and the test run
MATCH_KEYS = ['n_e', 'vd_mode', 'vd_e_mv', 'vd_k_mv', 'delta_e_mv', 'homeo', 'proposal',
              's_init', 's_mode', 'dt_ms', 'method', 'stim_ms', 'rest_ms', 'count_window',
              'weights_dir', 'conn_dir']


def load(run_dir):
    act = os.path.join(run_dir, 'activity')
    files = sorted(glob.glob(os.path.join(act, 'resultPopVecs*.npy')))
    if not files:
        sys.exit(f'no resultPopVecs*.npy in {act}')
    n = os.path.basename(files[-1])[len('resultPopVecs'):-4]
    return np.load(files[-1]), np.load(os.path.join(act, f'inputNumbers{n}.npy')).astype(int)


def predict(assignments, counts):
    scores = np.zeros((len(counts), 10))
    for c in range(10):
        m = assignments == c
        if m.sum():
            scores[:, c] = counts[:, m].sum(1) / m.sum()
    return np.argsort(scores, axis=1, kind='stable')[:, ::-1][:, 0]


def run_info(run_dir):
    try:
        with open(os.path.join(run_dir, 'run_info.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def neuron_summary(info):
    """The EIF-related settings and s statistics of one run (empty dict for LIF runs)."""
    if not info:
        return {}
    a = info.get('args', {})
    out = {k: a.get(k) for k in ('vd_mode', 'vd_e_mv', 'vd_k_mv', 'delta_e_mv') if k in a}
    if a.get('vd_mode') in ('homeo', 'proposal'):
        out.update({k: a.get(k) for k in ('s_init', 's_mode', 'homeo' if a.get('vd_mode') == 'homeo'
                                           else 'proposal')})
        out['s_source'] = info.get('weights', {}).get('s_init')
        out['s_start'] = info.get('s_start')
        out['s_end'] = info.get('s_end')
    for k in ('mean_exc_spikes_per_example', 'retries_total', 'examples_hitting_retry_cap'):
        if k in info:
            out[k] = info[k]
    return out


def describe(name, s):
    if not s:
        return
    line = f"{name}: vd-mode {s.get('vd_mode', 'fixed')}, Delta {s.get('delta_e_mv')} mV"
    if s.get('vd_mode') == 'track':
        line += f", k {s.get('vd_k_mv')} mV"
    if s.get('vd_mode') in ('homeo', 'proposal'):
        line += f", s-init {s.get('s_init')}, s-mode {s.get('s_mode')}"
    print(line)
    for when in ('s_start', 's_end'):
        r = s.get(when)
        if r:
            extra = (f", at floor {r['at_floor']}, at ceiling {r['at_ceiling']}"
                     if 'at_floor' in r else '')
            print(f"    {when.replace('_', ' ')}: region A {r['region_A']}, B {r['region_B']}"
                  f"{extra}, s {r['s_min_mV']:.2f} to {r['s_max_mV']:.2f} mV "
                  f"(mean {r['s_mean_mV']:.2f})")
    if 'mean_exc_spikes_per_example' in s:
        print(f"    {s['mean_exc_spikes_per_example']:.1f} spikes/image, "
              f"{s.get('retries_total', 0)} retries")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test-dir', required=True, help='output folder of a --mode test run')
    p.add_argument('--label-dir', default=None, help='output folder of a --mode label run')
    p.add_argument('--assignments', default=None,
                   help='neuron -> digit file saved with --save-assignments (replaces --label-dir)')
    p.add_argument('--save-assignments', default=None, help='write the assignments to this .npy file')
    a = p.parse_args()
    if a.assignments and a.label_dir:
        p.error('give --label-dir or --assignments, not both')
    test_counts, test_labels = load(a.test_dir)
    n_e = test_counts.shape[1]
    if a.assignments:
        assignments = np.load(a.assignments).astype(int)
        if assignments.shape != (n_e,):
            sys.exit(f'{a.assignments} has {assignments.shape[0]} neurons, expected {n_e}')
        protocol = f'assignments from {a.assignments} (paper protocol, saved label run)'
        pred = predict(assignments, test_counts)
    else:
        if a.label_dir:
            lab_counts, lab_labels = load(a.label_dir)
            protocol = f'assignments from {len(lab_labels)} training examples (paper protocol)'
        else:
            lab_counts, lab_labels = test_counts, test_labels
            protocol = 'assignments from the test set itself (demo protocol, optimistic)'
        pred, assignments = assign_and_predict(lab_counts, lab_labels, test_counts, n_e)
    if a.save_assignments:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_assignments)), exist_ok=True)
        np.save(a.save_assignments, assignments)
        print('Saved assignments to', a.save_assignments)
    acc = (pred == test_labels).mean() * 100
    per_class = [((pred == test_labels)[test_labels == c].mean() * 100) for c in range(10)]
    print(f'{protocol}\nTest examples: {len(test_labels)}   neurons: {n_e}   '
          f'accuracy: {acc:.2f}%')
    print('Neurons per digit:', np.bincount(assignments[assignments >= 0], minlength=10).tolist(),
          f'(unassigned: {(assignments < 0).sum()})')
    print('Accuracy per digit:', ' '.join(f'{c}:{v:.1f}' for c, v in enumerate(per_class)))

    # neuron settings of both runs, and a consistency check between them
    ti, li = run_info(a.test_dir), run_info(a.label_dir) if a.label_dir else None
    test_n, label_n = neuron_summary(ti), neuron_summary(li)
    describe('test run', test_n)
    describe('label run', label_n)
    mismatch = {}
    if ti and li:
        ta, la = ti.get('args', {}), li.get('args', {})
        for k in MATCH_KEYS:
            if ta.get(k) != la.get(k):
                mismatch[k] = dict(label=la.get(k), test=ta.get(k))
        if mismatch:
            print('WARNING: the label and test runs used different settings; the neuron '
                  'assignments come from a different network than the one tested:')
            for k, v in mismatch.items():
                print(f"    {k}: label {v['label']}  test {v['test']}")

    # confidence: relative gap between the best and second-best digit score
    scores = np.zeros((len(test_counts), 10))
    for c in range(10):
        m = assignments == c
        if m.any():
            scores[:, c] = test_counts[:, m].sum(1) / m.sum()
    srt = np.sort(scores, 1)
    margin = (srt[:, -1] - srt[:, -2]) / np.maximum(srt[:, -1], 1e-9)
    silent = int((test_counts.sum(1) == 0).sum())

    out = os.path.join(a.test_dir, 'evaluation.json')
    with open(out, 'w') as f:
        json.dump(dict(protocol=protocol, accuracy=acc, per_digit=per_class,
                       label_dir=a.label_dir, assignments_file=a.assignments,
                       neurons_per_digit=np.bincount(assignments[assignments >= 0],
                                                     minlength=10).tolist(),
                       median_confidence_margin=float(np.median(margin)),
                       test_images_without_spikes=silent,
                       test_run=test_n, label_run=label_n,
                       label_test_mismatch=mismatch), f, indent=2)
    print('Saved', out)


if __name__ == '__main__':
    main()
