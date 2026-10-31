#!/usr/bin/env python3
"""One HTML report (plus the PNGs) from the output folders of dc_standalone.py.

Give it whatever runs you have; each section appears only if its inputs exist.

    # trained/pretrained network + its label and test runs (the usual case)
    python visualize.py --label-dir runs/label400 --test-dir runs/test400 ^
                        --weights-dir stdp-mnist/weights --out reports/ne400

    # a training run (block accuracy, receptive-field and theta snapshots)
    python visualize.py --train-dir runs/train400 --out reports/train400

    # compare runs, e.g. LIF vs EIF, side by side on the key numbers
    python visualize.py --test-dir runs/test_lif --label-dir runs/label_lif ^
                        --compare runs/test_eif:runs/label_eif --out reports/lif_vs_eif

Sections
  1  Summary table
  2  Network: receptive fields (labelled by assigned digit), mean field per digit,
     theta and weight distributions
  3  Selectivity: neurons per digit, response heatmap (neurons x digits)
  4  Test performance: confusion matrix, accuracy per digit, accuracy vs confidence,
     misclassified examples with the neuron that won
  5  Activity: spikes per example, retries, input intensity
  6  Spike timing (needs --save-spikes in the test run): rasters, first-spike latency,
     gap between the first and second neuron to fire (near-simultaneous winners)
  7  Training (needs --train-dir): block accuracy, retries and spikes over training,
     how evenly neurons share the spikes, theta and receptive fields over time
"""
import argparse, base64, glob, io, json, os, sys
import numpy as np
# Windows consoles often use cp1252, which cannot print characters such as '−'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_standalone import assign_and_predict, load_matrix, load_mnist

DIGIT_COLORS = plt.get_cmap('tab10').colors
N_IN = 784


# ----------------------------------------------------------------------------- loading
def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_activity(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, 'activity', 'resultPopVecs*.npy')))
    if not files:
        return None, None
    n = os.path.basename(files[-1])[len('resultPopVecs'):-4]
    return (np.load(files[-1]),
            np.load(os.path.join(run_dir, 'activity', f'inputNumbers{n}.npy')).astype(int))


def load_npz(path):
    return dict(np.load(path)) if os.path.isfile(path) else None


def find_weights_dir(args):
    if args.weights_dir:
        return args.weights_dir
    if args.train_dir and os.path.isdir(os.path.join(args.train_dir, 'weights')):
        return os.path.join(args.train_dir, 'weights')
    for d in (args.test_dir, args.label_dir):
        info = read_json(os.path.join(d, 'run_info.json')) if d else None
        if info and isinstance(info.get('weights', {}).get('XeAe'), str):
            return os.path.dirname(info['weights']['XeAe'])
    return None


def load_weights(wdir, n_e, suffix=''):
    if not wdir:
        return None, None
    w = load_matrix(os.path.join(wdir, f'XeAe{suffix}.npy'), N_IN, n_e)
    tf = os.path.join(wdir, f'theta_A{suffix}.npy')
    theta = np.load(tf) * 1e3 if os.path.isfile(tf) else None      # volts -> mV
    return w, theta


def try_mnist(args, run_dir, train):
    cands = [args.mnist_dir] if args.mnist_dir else []
    info = read_json(os.path.join(run_dir, 'run_info.json')) if run_dir else None
    if info:
        cands.append(info['args'].get('mnist_dir'))
    for c in cands:
        if c and os.path.isdir(c):
            try:
                return load_mnist(c, train)[0]
            except SystemExit:
                pass
    return None


# ----------------------------------------------------------------------------- report
class Report:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.parts = []

    def h(self, text):
        self.parts.append(f'<h2>{text}</h2>')

    def p(self, text):
        self.parts.append(f'<p>{text}</p>')

    def table(self, rows, header=None):
        html = '<table>'
        if header:
            html += '<tr>' + ''.join(f'<th>{c}</th>' for c in header) + '</tr>'
        for r in rows:
            html += '<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
        self.parts.append(html + '</table>')

    def fig(self, fig, name, caption):
        fig.savefig(os.path.join(self.out_dir, name + '.png'), dpi=130, bbox_inches='tight')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        self.parts.append(f'<figure><img src="data:image/png;base64,{b64}" alt="{name}">'
                          f'<figcaption>{caption}</figcaption></figure>')

    def save(self, title):
        css = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;
        padding:0 16px;color:#1d1d1f;background:#fff} h1{font-size:24px} h2{font-size:19px;margin-top:36px;
        border-bottom:1px solid #ddd;padding-bottom:4px} figure{margin:16px 0} img{max-width:100%}
        figcaption{font-size:13px;color:#555;margin-top:4px} table{border-collapse:collapse;font-size:14px}
        td,th{border:1px solid #ddd;padding:4px 10px;text-align:left} th{background:#f4f4f6}
        p{font-size:14px;line-height:1.5}"""
        path = os.path.join(self.out_dir, 'report.html')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
                    f'<style>{css}</style></head><body><h1>{title}</h1>'
                    + '\n'.join(self.parts) + '</body></html>')
        return path


# ----------------------------------------------------------------------------- helpers
def grid_shape(n):
    c = int(np.ceil(np.sqrt(n)))
    return int(np.ceil(n / c)), c


def tile_fields(w, order=None):
    """Weight matrix [784, n_e] -> one image with a 28x28 tile per neuron."""
    n_e = w.shape[1]
    order = np.arange(n_e) if order is None else order
    r, c = grid_shape(n_e)
    img = np.full((r * 29 - 1, c * 29 - 1), np.nan)
    for k, j in enumerate(order):
        y, x = divmod(k, c)
        img[y * 29:y * 29 + 28, x * 29:x * 29 + 28] = w[:, j].reshape(28, 28)
    return img, r, c


def first_spike_info(spikes, examples, stim_ms):
    """Per accepted presentation: latency of the first excitatory spike, the gap to the
    first spike of a *different* neuron, and the number of distinct neurons that fired."""
    t, i = spikes['e_t_ms'], spikes['e_i'].astype(int)
    order = np.argsort(t, kind='stable')
    t, i = t[order], i[order]
    ends = examples['stim_end_s'] * 1000.0
    lat, gap, ndist, first_n = [], [], [], []
    for e in ends:
        a, b = np.searchsorted(t, e - stim_ms), np.searchsorted(t, e)
        if b <= a:
            lat.append(np.nan); gap.append(np.nan); ndist.append(0); first_n.append(-1)
            continue
        tt, ii = t[a:b], i[a:b]
        lat.append(tt[0] - (e - stim_ms))
        first_n.append(ii[0])
        other = np.nonzero(ii != ii[0])[0]
        gap.append(tt[other[0]] - tt[0] if len(other) else np.nan)
        ndist.append(len(np.unique(ii)))
    return np.array(lat), np.array(gap), np.array(ndist), np.array(first_n)


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return np.nan
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


# ----------------------------------------------------------------------------- sections
def section_network(R, w, theta, assignments):
    n_e = w.shape[1]
    R.h('2. Network: what each neuron has learned')
    order = None
    if assignments is not None:
        order = np.lexsort((-w.max(0), assignments))       # group by assigned digit
    img, r, c = tile_fields(w, order)
    fig, ax = plt.subplots(figsize=(min(14, 0.5 * c + 2), min(14, 0.5 * r + 1.5)))
    ax.imshow(img, cmap='hot_r', vmin=0, vmax=np.nanpercentile(img, 99.5),
              interpolation='nearest')
    if assignments is not None and n_e <= 1600:
        for k, j in enumerate(order):
            y, x = divmod(k, c)
            a = assignments[j]
            ax.text(x * 29 + 1, y * 29 + 6, '?' if a < 0 else str(a),
                    fontsize=max(3, 7 - n_e // 400), color=DIGIT_COLORS[a] if a >= 0 else 'k',
                    fontweight='bold')
    ax.set_axis_off()
    R.fig(fig, 'receptive_fields',
          'Input weights of every excitatory neuron as a 28×28 image (darker = stronger). '
          + ('Neurons are grouped by the digit they were assigned; the small number is that digit.'
             if assignments is not None else 'Run with --label-dir to group them by digit.'))

    if assignments is not None:
        fig, axs = plt.subplots(1, 10, figsize=(13, 1.8))
        for d in range(10):
            m = assignments == d
            axs[d].imshow(w[:, m].mean(1).reshape(28, 28) if m.any() else np.zeros((28, 28)),
                          cmap='hot_r')
            axs[d].set_title(f'{d}  (n={m.sum()})', fontsize=9)
            axs[d].set_axis_off()
        R.fig(fig, 'mean_field_per_digit',
              'Average receptive field of the neurons assigned to each digit.')

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
    if theta is not None:
        axs[0].hist(theta, bins=40, color='C0')
        axs[0].set_xlabel('θ (mV)')
        axs[0].set_ylabel('neurons')
        sec = axs[0].secondary_xaxis('top', functions=(lambda x: x - 72, lambda x: x + 72))
        sec.set_xlabel('spike detector −72 + θ (mV)')
        axs[0].set_title(f'Adaptive threshold: mean {theta.mean():.1f} mV, '
                         f'range {theta.min():.1f}–{theta.max():.1f}', fontsize=10)
    axs[1].hist(w.ravel(), bins=60, color='C1', log=True)
    axs[1].set_xlabel('input weight')
    axs[1].set_ylabel('synapses (log)')
    axs[1].set_title(f'Weights: mean {w.mean():.3f}, max {w.max():.3f}, '
                     f'sum per neuron {w.sum(0).mean():.1f}', fontsize=10)
    R.fig(fig, 'theta_and_weights',
          'Left: θ per neuron (a high θ means the neuron won often during training). '
          'Right: distribution of all input weights.')


def section_selectivity(R, lab_counts, lab_labels, assignments):
    n_e = lab_counts.shape[1]
    R.h('3. Selectivity: which digit each neuron answers to')
    rates = np.stack([lab_counts[lab_labels == d].mean(0) for d in range(10)], 1)  # [n_e, 10]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1, 2]))
    cnt = np.bincount(assignments[assignments >= 0], minlength=10)
    axs[0].bar(range(10), cnt, color=DIGIT_COLORS)
    axs[0].set_xticks(range(10))
    axs[0].set_xlabel('digit')
    axs[0].set_ylabel('neurons assigned')
    axs[0].set_title(f'Neurons per digit (unassigned: {(assignments < 0).sum()})', fontsize=10)
    order = np.lexsort((-rates.max(1), assignments))
    norm = rates[order] / np.maximum(rates[order].max(1, keepdims=True), 1e-9)
    axs[1].imshow(norm.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axs[1].set_yticks(range(10))
    axs[1].set_ylabel('digit shown')
    axs[1].set_xlabel('neuron (grouped by assigned digit)')
    axs[1].set_title('Mean response, each neuron scaled to its own maximum', fontsize=10)
    R.fig(fig, 'selectivity',
          'A sharp bright line in each column means a neuron that answers to one digit only. '
          'Columns lit for several digits are neurons that are not selective.')
    sel = rates.max(1) / np.maximum(rates.sum(1), 1e-9)
    silent = (lab_counts.sum(0) == 0).sum()
    R.p(f'Selectivity index (response to preferred digit ÷ sum over digits, 0.1 = none, '
        f'1 = perfect): median <b>{np.median(sel[rates.sum(1) > 0]):.2f}</b>. '
        f'Neurons that never fired in the label run: <b>{silent}</b> of {n_e}.')
    return rates


def section_test(R, test_counts, test_labels, pred, assignments, scores, images, examples):
    R.h('4. Test performance')
    correct = pred == test_labels
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (test_labels, pred), 1)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    axs[0].imshow(cmn, cmap='Blues', vmin=0, vmax=100)
    for a in range(10):
        for b in range(10):
            if cm[a, b]:
                axs[0].text(b, a, f'{cmn[a, b]:.0f}' if a == b else str(cm[a, b]),
                            ha='center', va='center', fontsize=7,
                            color='white' if cmn[a, b] > 60 else 'black')
    axs[0].set_xticks(range(10)); axs[0].set_yticks(range(10))
    axs[0].set_xlabel('predicted'); axs[0].set_ylabel('true')
    axs[0].set_title('Confusion matrix (diagonal: % correct; off-diagonal: counts)', fontsize=10)
    per = [correct[test_labels == d].mean() * 100 for d in range(10)]
    axs[1].bar(range(10), per, color=DIGIT_COLORS)
    axs[1].axhline(correct.mean() * 100, color='k', ls='--', lw=1)
    axs[1].set_ylim(min(per) - 8, 100)
    axs[1].set_xticks(range(10))
    axs[1].set_xlabel('digit'); axs[1].set_ylabel('accuracy (%)')
    axs[1].set_title(f'Accuracy per digit (dashed: overall {correct.mean() * 100:.2f}%)',
                     fontsize=10)
    R.fig(fig, 'confusion', 'Where the errors go. The largest off-diagonal counts are '
          'the digit pairs the network confuses most.')
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:5], off.shape))[0]
    R.p('Most frequent confusions: ' + ', '.join(f'{a}→{b} ({off[a, b]})' for a, b in top))

    # confidence = gap between best and second-best digit score
    s = np.sort(scores, 1)
    margin = (s[:, -1] - s[:, -2]) / np.maximum(s[:, -1], 1e-9)
    silent = test_counts.sum(1) == 0
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
    # deciles by rank, so that tied margins (common with small spike counts) still split evenly
    idx = np.full(len(margin), -1)
    nz = np.nonzero(~silent)[0]
    rank = np.empty(len(nz), int)
    rank[np.argsort(margin[nz], kind='stable')] = np.arange(len(nz))
    idx[nz] = rank * 10 // max(len(nz), 1)
    accb = [correct[(idx == k) & ~silent].mean() * 100 if ((idx == k) & ~silent).any() else np.nan
            for k in range(10)]
    axs[0].plot(range(1, 11), accb, 'o-')
    axs[0].set_xlabel('confidence decile (1 = least confident)')
    axs[0].set_ylabel('accuracy (%)')
    axs[0].set_title('Accuracy vs confidence (relative gap between top two digit scores)',
                     fontsize=10)
    tot = test_counts.sum(1)
    axs[1].hist([tot[correct], tot[~correct]], bins=np.arange(0, tot.max() + 2) - 0.5,
                stacked=True, color=['C2', 'C3'], label=['correct', 'wrong'])
    axs[1].set_xlabel('excitatory spikes in the counting window')
    axs[1].set_ylabel('test images')
    axs[1].legend()
    axs[1].set_title(f'Spikes per test image (images with no spike: {silent.sum()})',
                     fontsize=10)
    R.fig(fig, 'confidence', 'Low-confidence and low-spike images are where errors '
          'concentrate. A model change that helps should move errors out of the left side.')

    if images is not None:
        wrong = np.nonzero(~correct)[0][:24]
        if len(wrong):
            img_idx = (examples['image'] if examples is not None and 'image' in examples
                       else np.arange(len(test_labels)))
            fig, axs = plt.subplots(3, 8, figsize=(13, 5.4))
            for ax, k in zip(axs.ravel(), wrong):
                ax.imshow(images[img_idx[k]].reshape(28, 28), cmap='gray_r')
                ax.set_title(f'true {test_labels[k]} → {pred[k]}', fontsize=9,
                             color=DIGIT_COLORS[pred[k]])
                ax.set_axis_off()
            for ax in axs.ravel()[len(wrong):]:
                ax.set_axis_off()
            R.fig(fig, 'misclassified', 'The first 24 misclassified test images.')


def section_activity(R, examples, label):
    R.h(f'5. Activity during the {label} run')
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
    sp = examples['spikes']
    axs[0].hist(sp, bins=np.arange(0, np.percentile(sp, 99.5) + 2) - 0.5, color='C0')
    axs[0].set_xlabel('excitatory spikes per image'); axs[0].set_ylabel('images')
    axs[0].set_title(f'mean {sp.mean():.1f}, median {np.median(sp):.0f}', fontsize=10)
    rt = examples['retries']
    axs[1].bar(*np.unique(rt, return_counts=True), color='C1')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('retries before acceptance'); axs[1].set_ylabel('images (log)')
    axs[1].set_title(f'{(rt > 0).mean() * 100:.1f}% of images needed a retry', fontsize=10)
    labels = examples.get('label')
    if labels is not None:
        r_d = [(rt[labels == d] > 0).mean() * 100 for d in range(10)]
        axs[2].bar(range(10), r_d, color=DIGIT_COLORS)
        axs[2].set_xticks(range(10))
        axs[2].set_xlabel('digit'); axs[2].set_ylabel('% needing a retry')
        axs[2].set_title('Retries by digit', fontsize=10)
    R.fig(fig, f'activity_{label}', 'Retries happen when fewer than the minimum number of '
          'spikes occur; the image is then shown again at higher intensity.')


def section_timing(R, spikes, examples, assignments, pred, test_labels, stim_ms, label):
    R.h('6. Spike timing (who fires first, and how close the race is)')
    lat, gap, ndist, first_n = first_spike_info(spikes, examples, stim_ms)
    # rasters for six images
    ex = examples['example']
    pick = np.linspace(0, len(ex) - 1, 6).astype(int)
    n_e = len(assignments) if assignments is not None else int(spikes['e_i'].max()) + 1
    pos = np.empty(n_e, int)
    order = np.argsort(assignments, kind='stable') if assignments is not None else np.arange(n_e)
    pos[order] = np.arange(n_e)
    fig, axs = plt.subplots(2, 3, figsize=(13, 6), sharey=True)
    t_all, i_all = spikes['e_t_ms'], spikes['e_i'].astype(int)
    for ax, k in zip(axs.ravel(), pick):
        e = examples['stim_end_s'][k] * 1000
        m = (t_all >= e - stim_ms) & (t_all < e)
        cols = [DIGIT_COLORS[assignments[j]] if assignments is not None and assignments[j] >= 0
                else 'k' for j in i_all[m]]
        ax.scatter(t_all[m] - (e - stim_ms), pos[i_all[m]], s=10, c=cols)
        if 'i_t_ms' in spikes:
            mi = (spikes['i_t_ms'] >= e - stim_ms) & (spikes['i_t_ms'] < e)
            ax.scatter(spikes['i_t_ms'][mi] - (e - stim_ms), np.full(mi.sum(), -15),
                       s=6, c='grey', marker='|')
        title = f'image {ex[k]}: true {test_labels[ex[k]]}' if test_labels is not None else \
            f'image {ex[k]}'
        if pred is not None:
            title += f', predicted {pred[ex[k]]}'
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, stim_ms)
    for ax in axs[1]:
        ax.set_xlabel('ms since stimulus onset')
    axs[0, 0].set_ylabel('neuron (grouped by digit)')
    axs[1, 0].set_ylabel('neuron (grouped by digit)')
    fig.tight_layout()
    R.fig(fig, f'rasters_{label}', 'Excitatory spikes during the accepted presentation of six '
          'images, coloured by the digit each neuron is assigned to'
          + ('; inhibitory spikes in grey at the bottom.' if 'i_t_ms' in spikes else '.'))

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.3))
    ok = ~np.isnan(lat)
    if pred is not None and test_labels is not None:
        corr = (pred == test_labels)[ex]
        axs[0].hist([lat[ok & corr], lat[ok & ~corr]], bins=40, stacked=True,
                    color=['C2', 'C3'], label=['correct', 'wrong'])
        axs[0].legend()
    else:
        axs[0].hist(lat[ok], bins=40)
    axs[0].set_xlabel('first-spike latency (ms)'); axs[0].set_ylabel('images')
    axs[0].set_title(f'median {np.nanmedian(lat):.1f} ms', fontsize=10)
    g = gap[~np.isnan(gap)]
    axs[1].hist(g, bins=np.linspace(0, np.percentile(g, 98) if len(g) else 1, 40), color='C4')
    axs[1].set_xlabel('ms until a second, different neuron fires')
    axs[1].set_title(f'{(g < 1).mean() * 100:.1f}% within 1 ms (near-simultaneous)',
                     fontsize=10)
    axs[2].bar(*np.unique(ndist, return_counts=True), color='C5')
    axs[2].set_xlabel('distinct neurons firing per image')
    axs[2].set_title(f'median {np.median(ndist):.0f}', fontsize=10)
    R.fig(fig, f'timing_{label}', 'The race the winner-take-all circuit runs. Left: when the '
          'first spike happens. Middle: how far ahead the winner is before a rival also fires '
          '(small gaps mean near-simultaneous winners). Right: how many neurons get to fire.')
    if pred is not None and test_labels is not None and assignments is not None:
        fn = first_n
        v = fn >= 0
        agree = (assignments[fn[v]] == pred[ex[v]]).mean() * 100
        firstright = (assignments[fn[v]] == test_labels[ex[v]]).mean() * 100
        R.p(f'The first neuron to fire belongs to the predicted digit in <b>{agree:.1f}%</b> of '
            f'images, and to the true digit in <b>{firstright:.1f}%</b> (classifying by the '
            f'first spike alone).')


def section_training(R, tdir, info, examples, n_e, wdir):
    R.h('7. Training run')
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.3))
    perf = info.get('train_block_accuracy') or []
    B = info['args'].get('perf_block', 10000)
    if perf:
        axs[0].plot(np.arange(2, len(perf) + 2) * B, perf, 'o-')
        axs[0].set_xlabel('examples trained'); axs[0].set_ylabel('accuracy (%)')
        axs[0].set_title(f'Block accuracy (blocks of {B}; last {perf[-1]:.1f}%)', fontsize=10)
    else:
        axs[0].text(0.5, 0.5, 'no block accuracy\n(run longer than 2 × --perf-block)',
                    ha='center', va='center', transform=axs[0].transAxes)
    if examples is not None:
        n = len(examples['retries'])
        W = max(n // 50, 1)
        xs = np.arange(n // W) * W + W
        rr = examples['retries'][:n // W * W].reshape(-1, W).mean(1)
        ss = examples['spikes'][:n // W * W].reshape(-1, W).mean(1)
        axs[1].plot(xs, rr, color='C1')
        axs[1].set_xlabel('examples trained'); axs[1].set_ylabel('retries per image')
        axs[1].set_title('Retries during training', fontsize=10)
        axs[2].plot(xs, ss, color='C0')
        axs[2].set_xlabel('examples trained'); axs[2].set_ylabel('spikes per image')
        axs[2].set_title('Spikes per image during training', fontsize=10)
    R.fig(fig, 'training_curves', 'Learning progress, retries and activity, averaged in '
          '50 windows across the run.')

    tc = os.path.join(tdir, 'train_counts.npy')
    if os.path.isfile(tc):
        counts = np.load(tc)
        n = len(counts)
        W = max(n // 20, 1)
        g, top, act = [], [], []
        for k in range(n // W):
            c = counts[k * W:(k + 1) * W].sum(0)
            g.append(gini(c)); top.append(np.sort(c)[::-1][:max(n_e // 100, 1)].sum() / max(c.sum(), 1) * 100)
            act.append((c > 0).mean() * 100)
        xs = np.arange(1, n // W + 1) * W
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
        axs[0].plot(xs, g); axs[0].set_title('Gini of spikes across neurons (0 = equal)', fontsize=10)
        axs[1].plot(xs, top, color='C3')
        axs[1].set_title(f'% of spikes from the top 1% of neurons', fontsize=10)
        axs[2].plot(xs, act, color='C2'); axs[2].set_title('% of neurons that fired', fontsize=10)
        for a in axs:
            a.set_xlabel('examples trained')
        R.fig(fig, 'training_sharing', 'How evenly the work is shared. Rising Gini or top-1% '
              'share means a few neurons are taking over: the failure the adaptive threshold '
              'exists to prevent, and the first thing to check in the EIF experiments.')

    snaps = sorted(glob.glob(os.path.join(wdir, 'XeAe_ex*.npy')),
                   key=lambda p: int(p.rsplit('_ex', 1)[1][:-4])) if wdir else []
    if snaps:
        steps = [int(p.rsplit('_ex', 1)[1][:-4]) for p in snaps]
        if len(snaps) > 6:
            keep = np.linspace(0, len(snaps) - 1, 6).astype(int)
            snaps = [snaps[k] for k in keep]; steps = [steps[k] for k in keep]
        final_w, final_t = load_weights(wdir, n_e)
        th_series = []
        neurons = np.linspace(0, n_e - 1, 8).astype(int)
        fig, axs = plt.subplots(len(neurons), len(snaps) + 1,
                                figsize=(1.4 * (len(snaps) + 1), 1.4 * len(neurons)))
        for col, (p, s) in enumerate(zip(snaps + [None], steps + [None])):
            if p is None:
                w, th = final_w, final_t
                head = 'final'
            else:
                w, th = load_weights(wdir, n_e, f'_ex{s}')
                head = f'{s}'
            if th is not None:
                th_series.append((head, th))
            for row, j in enumerate(neurons):
                axs[row, col].imshow(w[:, j].reshape(28, 28), cmap='hot_r')
                axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
                if row == 0:
                    axs[row, col].set_title(head, fontsize=8)
                if col == 0:
                    axs[row, col].set_ylabel(f'n{j}', fontsize=8)
        R.fig(fig, 'field_evolution', 'Receptive fields of eight neurons at each saved snapshot '
              '(columns: examples trained).')
        if th_series:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.boxplot([t for _, t in th_series], showfliers=False)
            ax.set_xticks(range(1, len(th_series) + 1), [h for h, _ in th_series])
            ax.set_xlabel('examples trained'); ax.set_ylabel('θ (mV)')
            ax.set_title('Adaptive threshold across neurons over training', fontsize=10)
            R.fig(fig, 'theta_evolution', 'Box: middle 50% of neurons; whiskers: range '
                  'without outliers.')


def section_compare(R, rows):
    R.h('Comparison')
    R.table(rows, header=['run', 'accuracy %', 'spikes/image', 'retry %',
                          'median latency ms', 'near-simultaneous %'])


# ----------------------------------------------------------------------------- main
def summarise_test(test_dir, label_dir):
    tc, tl = load_activity(test_dir)
    if tc is None:
        return None
    lc, ll = load_activity(label_dir) if label_dir else (tc, tl)
    pred, assignments = assign_and_predict(lc, ll, tc, tc.shape[1])
    ex = load_npz(os.path.join(test_dir, 'examples.npz'))
    sp = load_npz(os.path.join(test_dir, 'spikes.npz'))
    lat_med = near = '–'
    if sp is not None and ex is not None:
        lat, gap, _, _ = first_spike_info(sp, ex, float(sp['stim_ms']))
        lat_med = f'{np.nanmedian(lat):.1f}'
        g = gap[~np.isnan(gap)]
        near = f'{(g < 1).mean() * 100:.1f}' if len(g) else '–'
    return [test_dir, f'{(pred == tl).mean() * 100:.2f}',
            f'{ex["spikes"].mean():.1f}' if ex is not None else '–',
            f'{(ex["retries"] > 0).mean() * 100:.1f}' if ex is not None else '–', lat_med, near]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test-dir'); ap.add_argument('--label-dir'); ap.add_argument('--train-dir')
    ap.add_argument('--weights-dir', help='XeAe.npy + theta_A.npy (default: taken from the runs)')
    ap.add_argument('--mnist-dir', help='only needed for the misclassified-image gallery')
    ap.add_argument('--compare', nargs='*', default=[],
                    help='more runs to tabulate, each as TEST_DIR or TEST_DIR:LABEL_DIR')
    ap.add_argument('--out', default='report', help='output folder (report.html + PNGs)')
    a = ap.parse_args()
    if not (a.test_dir or a.label_dir or a.train_dir):
        ap.error('give at least one of --test-dir, --label-dir, --train-dir')

    R = Report(a.out)
    test_counts, test_labels = load_activity(a.test_dir) if a.test_dir else (None, None)
    lab_counts, lab_labels = load_activity(a.label_dir) if a.label_dir else (None, None)
    tinfo = read_json(os.path.join(a.train_dir, 'run_info.json')) if a.train_dir else None
    n_e = (test_counts.shape[1] if test_counts is not None else
           lab_counts.shape[1] if lab_counts is not None else tinfo['args']['n_e'])

    assignments = pred = scores = None
    protocol = None
    if test_counts is not None:
        src_c, src_l = (lab_counts, lab_labels) if lab_counts is not None else (test_counts, test_labels)
        protocol = ('paper protocol (assignments from the label run)' if lab_counts is not None
                    else 'demo protocol (assignments from the test run itself, optimistic)')
        pred, assignments = assign_and_predict(src_c, src_l, test_counts, n_e)
        scores = np.zeros((len(test_counts), 10))
        for d in range(10):
            m = assignments == d
            if m.any():
                scores[:, d] = test_counts[:, m].sum(1) / m.sum()
    elif lab_counts is not None:
        _, assignments = assign_and_predict(lab_counts, lab_labels, lab_counts[:1], n_e)
    elif a.train_dir and os.path.isfile(os.path.join(a.train_dir, 'train_counts.npy')):
        tc = np.load(os.path.join(a.train_dir, 'train_counts.npy'))
        tl = np.load(os.path.join(a.train_dir, 'train_labels.npy'))
        k = min(10000, len(tc))
        _, assignments = assign_and_predict(tc[-k:], tl[-k:], tc[:1], n_e)

    wdir = find_weights_dir(a)
    w, theta = load_weights(wdir, n_e)

    # 1. summary
    R.h('1. Summary')
    rows = [['neurons', n_e], ['weights', wdir or 'not found']]
    if test_counts is not None:
        rows += [['test images', len(test_labels)], ['evaluation', protocol],
                 ['accuracy', f'<b>{(pred == test_labels).mean() * 100:.2f}%</b>']]
    for name, d in (('label run', a.label_dir), ('test run', a.test_dir), ('train run', a.train_dir)):
        info = read_json(os.path.join(d, 'run_info.json')) if d else None
        if info:
            args = info['args']
            rows.append([name, f"{d}: {info['examples_completed']} images, "
                         f"{info['retries_total']} retries, "
                         f"{info['mean_exc_spikes_per_example']:.1f} spikes/image, "
                         f"dt {args['dt_ms']} ms, stim {args['stim_ms']}/{args['rest_ms']} ms, "
                         f"seed {args['seed']}"])
    R.table(rows)

    if w is not None:
        section_network(R, w, theta, assignments)
    else:
        R.p('Weights not found: pass --weights-dir to see receptive fields.')
    if lab_counts is not None and assignments is not None:
        section_selectivity(R, lab_counts, lab_labels, assignments)
    if test_counts is not None:
        ex = load_npz(os.path.join(a.test_dir, 'examples.npz'))
        section_test(R, test_counts, test_labels, pred, assignments, scores,
                     try_mnist(a, a.test_dir, train=False), ex)
        if ex is not None:
            section_activity(R, ex, 'test')
        sp = load_npz(os.path.join(a.test_dir, 'spikes.npz'))
        if sp is not None and ex is not None:
            section_timing(R, sp, ex, assignments, pred, test_labels, float(sp['stim_ms']), 'test')
        else:
            R.h('6. Spike timing')
            R.p('Not available: rerun the test with <code>--save-spikes</code> '
                '(and <code>--record-inh-spikes</code> for inhibitory spikes).')
    if tinfo:
        tex = load_npz(os.path.join(a.train_dir, 'examples.npz'))
        section_training(R, a.train_dir, tinfo, tex, n_e,
                         os.path.join(a.train_dir, 'weights'))
        tsp = load_npz(os.path.join(a.train_dir, 'spikes.npz'))
        if tsp is not None and tex is not None and test_counts is None:
            section_timing(R, tsp, tex, assignments, None, None, float(tsp['stim_ms']), 'train')

    if a.compare and a.test_dir:
        rows = [summarise_test(a.test_dir, a.label_dir)]
        for c in a.compare:
            t, _, l = c.partition(':')
            rows.append(summarise_test(t, l or None))
        section_compare(R, [r for r in rows if r])

    path = R.save(f'Diehl & Cook network: {n_e} neurons')
    print('Report:', os.path.abspath(path))


if __name__ == '__main__':
    main()
