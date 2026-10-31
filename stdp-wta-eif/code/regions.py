#!/usr/bin/env python3
"""Which region is each excitatory neuron in, and how does that relate to the errors?

For the frozen theta of a network, and the EIF parameters (vd, Delta), every excitatory
neuron is put in a region by where its spike detector sits relative to the EIF runaway
point Vx:

    detector  vth = v_thresh + theta - offset          (-52 + theta - 20 mV by default)
    runaway   Vx  = E_L - Delta * W_-1(-exp(-(vd - E_L)/Delta))   (unstable fixed point)
    m = (vth - Vx) / Delta                             (distance in units of Delta)

    region A : m < 0          detector below Vx  -> spikes like a LIF neuron
    region B : 0 <= m < K     detector just above Vx -> the runaway decides the spike
    region C : m >= K         detector far above Vx (K = --c-boundary, default 3)

Per-neuron runaway points. The EIF script's --vd-mode decides where Vx comes from
(read from the test run's run_info.json):
    fixed            one vd for all neurons            -> one Vx
    track            vd = vth - k for each neuron      -> Vx per neuron
    homeo, proposal  s = vth - Vx for each neuron      -> Vx = vth - s, so m = s / Delta
For homeo/proposal, s is read from the run itself: s at the START of the test run
(= <weights-dir>/s_A.npy for --s-init trained, or the constant given) and s at the END
(<test-dir>/s_end.npy). --s-at end|start picks which one sets the regions (default
end; for --s-mode frozen the two are identical). With --s-mode adaptive the report
adds a section on how s moved during the test run.

Then, with the assignments from a label run and the predictions on a test run, it
relates each digit's neurons (how many in A/B/C) to that digit's accuracy, its false
positives (other digits predicted as it), and its confusions.

    python regions.py --label-dir runs/eif_label --test-dir runs/eif_test --out reports/regions
    # same, plus the LIF run on the same images to show what changed per digit
    python regions.py --label-dir runs/eif_label --test-dir runs/eif_test ^
                      --baseline-test-dir runs/lif_test --baseline-label-dir runs/lif_label ^
                      --out reports/regions

vd, Delta, E_L, v_thresh and the offset are read from the test run's run_info.json
(options --vd-e-mv / --delta-e-mv of the EIF script); give --vd-mv / --delta-mv to
override them, e.g. to see where a LIF run's neurons WOULD sit for a given vd.
Theta is read from the weights folder the test run used (or --weights-dir).
"""
import argparse, csv, json, os, sys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import Report, load_activity, read_json, DIGIT_COLORS
from dc_standalone import assign_and_predict

REGION_COLORS = {'A': '#4C9BE8', 'B': '#F2A93B', 'C': '#D64545'}
REGIONS = ['A', 'B', 'C']


def runaway_point(vd, delta, e_l):
    """Vx in mV, or None if the neuron has no resting state (vd - E_L < Delta)."""
    a = (vd - e_l) / delta
    if a < 1:
        return None
    # upper root of (v - E_L) = Delta*exp((v - vd)/Delta), i.e. the Lambert-W_-1 solution;
    # solved by bisection so that scipy is not needed. f(vd) >= 0 and f -> -inf above.
    lo, hi = vd, vd + 60 * delta
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (mid - e_l) - delta * np.exp((mid - vd) / delta) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def get_params(a, info):
    args = (info or {}).get('args', {})
    def pick(cli, key, default=None):
        if cli is not None:
            return cli
        if key in args and args[key] is not None:
            return float(args[key])
        return default
    p = dict(vd=pick(a.vd_mv, 'vd_e_mv'), delta=pick(a.delta_mv, 'delta_e_mv'),
             e_l=pick(a.e_l_mv, 'v_rest_e_mv', -65.0),
             v_thresh=pick(a.v_thresh_mv, 'v_thresh_e_mv', -52.0),
             offset=pick(a.offset_mv, 'theta_offset_mv', 20.0))
    mode = args.get('vd_mode', 'fixed') if a.vd_mv is None else 'fixed'
    if p['delta'] is None or (mode == 'fixed' and p['vd'] is None):
        sys.exit('vd / Delta not found in run_info.json: give --vd-mv and --delta-mv')
    if mode == 'fixed' and p['vd'] > 50:
        print(f"WARNING: vd = {p['vd']} mV is the 'EIF off' setting; regions are then "
              f"meaningless (everything is region A). Give --vd-mv to evaluate a real vd.")
    return p


def vx_per_neuron(a, info, p, vth, n_e):
    """Runaway point Vx (mV) for every neuron, and the s values used (mV, or None).
    Returns (vx array, description, s_start, s_end)."""
    args = (info or {}).get('args', {})
    mode = args.get('vd_mode', 'fixed')
    if a.vd_mv is not None:
        mode = 'fixed'                               # explicit override: one vd
    if mode == 'track':
        k = float(args.get('vd_k_mv', 2.0))
        vx = np.array([runaway_point(t - k, p['delta'], p['e_l']) or np.nan for t in vth])
        return vx, f'track: vd = vth − {k} mV per neuron', None, None
    if mode in ('homeo', 'proposal'):
        s_end_path = os.path.join(a.test_dir, 's_end.npy')
        s_end = np.load(s_end_path) * 1e3 if os.path.isfile(s_end_path) else None
        init = args.get('s_init')
        if init == 'trained':
            wdir = a.weights_dir or args.get('weights_dir')
            path = os.path.join(wdir, 's_A.npy')
            if not os.path.isfile(path):
                sys.exit(f'{path} not found: give --weights-dir (the training run\'s weights)')
            s_start = np.load(path) * 1e3
        elif init is not None:
            s_start = np.full(n_e, float(init))
        else:
            s_start = s_end
        if s_start is None:
            sys.exit('no s found for this run (s_end.npy missing and no --s-init in run_info)')
        if mode == 'homeo' and init != 'trained':     # same clip the EIF script applies
            H = {k_.strip(): float(v) for k_, v in
                 (kv.split('=') for kv in args.get('homeo', 'smin=-1.5,smax=2,margin=1').split(','))}
            d1 = vth - (p['e_l'] if args.get('v_reset_e_mv') is None else float(args['v_reset_e_mv']))
            s_start = np.clip(s_start, H.get('smin', -1.5),
                              np.minimum(H.get('smax', 2.0), d1 - p['delta'] - H.get('margin', 1)))
        s_use = s_end if (a.s_at == 'end' and s_end is not None) else s_start
        desc = (f"{mode}: Vx = vth − s per neuron (s at {'end' if s_use is s_end else 'start'} "
                f"of the run; --s-init {init}, --s-mode {args.get('s_mode')})")
        return vth - s_use, desc, s_start, s_end
    vx0 = runaway_point(p['vd'], p['delta'], p['e_l'])
    if vx0 is None:
        sys.exit(f"vd - E_L = {p['vd'] - p['e_l']:.2f} mV < Delta = {p['delta']} mV: the "
                 f"neuron has no resting state; no Vx")
    return np.full(n_e, vx0), f"fixed: vd = {p['vd']:.3f} mV for every neuron", None, None


def theta_mv(a, info):
    path = None
    if a.weights_dir:
        path = os.path.join(a.weights_dir, 'theta_A.npy')
    elif info and isinstance(info.get('weights', {}).get('theta'), str):
        path = info['weights']['theta']
    if not path or not os.path.isfile(path):
        sys.exit(f'theta_A.npy not found ({path}); give --weights-dir')
    return np.load(path) * 1e3, path


def predictions(test_dir, label_dir):
    tc, tl = load_activity(test_dir)
    if tc is None:
        sys.exit(f'no activity/ in {test_dir}')
    lc, ll = load_activity(label_dir) if label_dir else (tc, tl)
    if lc is None:
        sys.exit(f'no activity/ in {label_dir}')
    pred, assign = assign_and_predict(lc, ll, tc, tc.shape[1])
    rates = np.stack([lc[ll == d].mean(0) for d in range(10)], 1)
    return tc, tl, pred, assign, rates


def region_line(s_mV, delta, c_boundary):
    if s_mV is None:
        return '\u2013'
    m_ = s_mV / delta
    return (f'A {(m_ < 0).sum()} \u00b7 B {((m_ >= 0) & (m_ < c_boundary)).sum()} \u00b7 '
            f'C {(m_ >= c_boundary).sum()}  (s {s_mV.min():.2f} to {s_mV.max():.2f} mV)')


def gini(x):
    """0 = every neuron fires equally, 1 = one neuron does all the firing."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float('nan')
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def sharing_stats(counts):
    """How evenly the spikes of a run are shared across neurons."""
    tot = counts.sum(0)
    srt = np.sort(tot)[::-1]
    n = len(tot)
    k1, k10 = max(n // 100, 1), max(n // 10, 1)
    return dict(gini=gini(tot), top1=srt[:k1].sum() / max(tot.sum(), 1) * 100,
                top10=srt[:k10].sum() / max(tot.sum(), 1) * 100,
                silent=int((tot == 0).sum()), spikes_per_image=float(counts.sum(1).mean()),
                tot=tot)


def lorenz(tot):
    x = np.sort(np.asarray(tot, float))
    c = np.concatenate([[0], np.cumsum(x)]) / max(x.sum(), 1)
    return np.linspace(0, 1, len(c)), c


def confusion(true, pred):
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (true, pred), 1)
    return cm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test-dir', required=True, help='test run (frozen weights)')
    ap.add_argument('--label-dir', help='label run for the neuron assignments '
                    '(paper protocol). Without it the test run itself is used (optimistic)')
    ap.add_argument('--baseline-test-dir', help='e.g. the LIF test run on the same images')
    ap.add_argument('--baseline-label-dir', help='label run for the baseline')
    ap.add_argument('--weights-dir', help='folder with theta_A.npy (default: from run_info)')
    ap.add_argument('--vd-mv', type=float); ap.add_argument('--delta-mv', type=float)
    ap.add_argument('--e-l-mv', type=float); ap.add_argument('--v-thresh-mv', type=float)
    ap.add_argument('--offset-mv', type=float)
    ap.add_argument('--c-boundary', type=float, default=3.0,
                    help='region C starts at m = (vth - Vx)/Delta >= this')
    ap.add_argument('--s-at', choices=['start', 'end'], default='end',
                    help='homeo/proposal runs: use s at the end of the test run (default; '
                         'the same as the start when --s-mode frozen) or at the start')
    ap.add_argument('--out', default='regions_report')
    a = ap.parse_args()

    info = read_json(os.path.join(a.test_dir, 'run_info.json'))
    p = get_params(a, info)
    theta, theta_path = theta_mv(a, info)
    vth = p['v_thresh'] + theta - p['offset']
    n_e = len(theta)
    vx, vx_desc, s_start, s_end = vx_per_neuron(a, info, p, vth, n_e)
    m = (vth - vx) / p['delta']
    region = np.where(m < 0, 'A', np.where(m < a.c_boundary, 'B', 'C'))
    vx_text = (f'{vx[0]:.2f} mV' if np.allclose(vx, vx[0])
               else f'{np.nanmin(vx):.2f} to {np.nanmax(vx):.2f} mV (median {np.nanmedian(vx):.2f})')

    tc, tl, pred, assign, rates = predictions(a.test_dir, a.label_dir)
    if tc.shape[1] != n_e:
        sys.exit(f'theta has {n_e} neurons but the test run has {tc.shape[1]}')
    correct = pred == tl
    cm = confusion(tl, pred)
    acc_d = np.array([correct[tl == d].mean() * 100 for d in range(10)])
    fp = cm.sum(0) - np.diag(cm)                       # other digits predicted as d
    base = None
    if a.baseline_test_dir:
        btc, btl, bpred, bassign, _ = predictions(a.baseline_test_dir, a.baseline_label_dir)
        if not np.array_equal(btl, tl):
            print('WARNING: baseline test images differ; per-image comparison skipped')
        bcm = confusion(btl, bpred)
        base = dict(acc=(bpred == btl).mean() * 100,
                    acc_d=np.array([(bpred == btl)[btl == d].mean() * 100 for d in range(10)]),
                    fp=bcm.sum(0) - np.diag(bcm), cm=bcm,
                    same=np.array_equal(btl, tl), pred=bpred, counts=btc, assign=bassign)

    R = Report(a.out)
    os.makedirs(a.out, exist_ok=True)

    # ---------------------------------------------------------------- 1. settings
    R.h('1. Settings and region definition')
    R.table([
        ['test run', a.test_dir],
        ['assignments from', a.label_dir or 'the test run itself (demo protocol, optimistic)'],
        ['theta', theta_path],
        ['E_L / detector', f"{p['e_l']} mV / v_thresh {p['v_thresh']} + θ − {p['offset']} mV"],
        ['vd rule, Δ', f"{vx_desc}; Δ = {p['delta']} mV"],
        ['Vx (runaway point, no input)', f'<b>{vx_text}</b>'],
        *([['regions at start of the run', region_line(s_start, p['delta'], a.c_boundary)],
           ['regions at end of the run', region_line(s_end, p['delta'], a.c_boundary)],
           ['regions used in this report', f"s at the {a.s_at if s_end is not None else 'start'}"]]
          if s_start is not None else []),
        ['regions', f'A: detector below Vx (m &lt; 0) · B: 0 ≤ m &lt; {a.c_boundary:g} · '
                    f'C: m ≥ {a.c_boundary:g}, with m = (detector − Vx)/Δ'],
        ['accuracy', f'{correct.mean() * 100:.2f}% on {len(tl)} test images'
                     + (f' (baseline {base["acc"]:.2f}%)' if base else '')],
    ])
    R.p('Vx is computed without input. Inhibition raises it (to about Vx + Δ·ln(1 + gi)) '
        'and excitation lowers it, so a neuron\'s region is where it sits in a quiet '
        'network, not at every moment.')

    # ---------------------------------------------------------------- 2. distribution
    R.h('2. Where the neurons sit')
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    bins = np.linspace(min(m.min(), -1), m.max() + 0.5, 50)
    for r in REGIONS:
        axs[0].hist(m[region == r], bins=bins, color=REGION_COLORS[r],
                    label=f'{r}: {(region == r).sum()} neurons')
    for x in (0, a.c_boundary):
        axs[0].axvline(x, color='k', lw=0.8, ls='--')
    axs[0].set_xlabel('m = (detector − Vx) / Δ'); axs[0].set_ylabel('neurons')
    axs[0].legend(); axs[0].set_title(f'Vx {vx_text}; detectors {vth.min():.1f} to '
                                      f'{vth.max():.1f} mV', fontsize=10)
    for d in range(10):
        s = assign == d
        axs[1].scatter(np.full(s.sum(), d) + np.random.default_rng(d).uniform(-.25, .25, s.sum()),
                       m[s], s=9, color=DIGIT_COLORS[d])
    for x in (0, a.c_boundary):
        axs[1].axhline(x, color='k', lw=0.8, ls='--')
    axs[1].set_xticks(range(10)); axs[1].set_xlabel('assigned digit')
    axs[1].set_ylabel('m'); axs[1].set_title('Each neuron, by the digit it is assigned to',
                                             fontsize=10)
    R.fig(fig, 'region_distribution', 'Left: how far each detector sits above (or below) '
          'the runaway point. Right: the same per assigned digit; dashed lines are the A/B '
          'and B/C boundaries.')

    # ---------------------------------------------------------------- 3. per digit
    R.h('3. Regions per digit, against accuracy and errors')
    counts = np.array([[((assign == d) & (region == r)).sum() for r in REGIONS]
                       for d in range(10)])
    n_d = counts.sum(1)
    share = counts / np.maximum(n_d[:, None], 1) * 100
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    bottom = np.zeros(10)
    for k, r in enumerate(REGIONS):
        axs[0].bar(range(10), share[:, k], bottom=bottom, color=REGION_COLORS[r], label=r)
        bottom += share[:, k]
    for d in range(10):
        axs[0].text(d, 101, str(n_d[d]), ha='center', fontsize=8)
    axs[0].set_ylim(0, 108); axs[0].set_xticks(range(10)); axs[0].set_xlabel('digit')
    axs[0].set_ylabel('% of the digit\'s neurons'); axs[0].legend(loc='lower right')
    axs[0].set_title('Region mix per digit (number above: neurons)', fontsize=10)

    w = 0.38
    axs[1].bar(np.arange(10) - (w / 2 if base else 0), acc_d, width=w if base else 0.7,
               color=DIGIT_COLORS, label='this run')
    if base:
        axs[1].bar(np.arange(10) + w / 2, base['acc_d'], width=w, color='lightgrey',
                   label='baseline')
        axs[1].legend()
    axs[1].set_ylim(max(0, min(acc_d.min(), base['acc_d'].min() if base else 100) - 10), 100)
    axs[1].set_xticks(range(10)); axs[1].set_xlabel('digit'); axs[1].set_ylabel('accuracy (%)')
    axs[1].set_title('Accuracy per digit', fontsize=10)

    axs[2].bar(np.arange(10) - (w / 2 if base else 0), fp, width=w if base else 0.7,
               color=DIGIT_COLORS)
    if base:
        axs[2].bar(np.arange(10) + w / 2, base['fp'], width=w, color='lightgrey')
    axs[2].set_xticks(range(10)); axs[2].set_xlabel('digit predicted')
    axs[2].set_ylabel('images of other digits')
    axs[2].set_title('False positives: other digits predicted as this one', fontsize=10)
    R.fig(fig, 'per_digit', 'If neurons far past their runaway point (C) over-respond, the '
          'digits they belong to should collect more false positives, and other digits '
          'should lose accuracy to them.')

    # relation plots
    pc = share[:, 2] + share[:, 1] if a.c_boundary > 0 else share[:, 2]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, y, lab in ((axs[0], fp if not base else fp - base['fp'],
                        'false positives' + (' − baseline' if base else '')),
                       (axs[1], acc_d if not base else acc_d - base['acc_d'],
                        'accuracy (%)' + (' − baseline' if base else ''))):
        ax.scatter(share[:, 2], y, c=DIGIT_COLORS, s=60)
        for d in range(10):
            ax.annotate(str(d), (share[d, 2], y[d]), xytext=(4, 3), textcoords='offset points')
        ax.set_xlabel('% of the digit\'s neurons in region C'); ax.set_ylabel(lab)
        if len(set(share[:, 2])) > 2:
            r_ = np.corrcoef(share[:, 2], y)[0, 1]
            ax.set_title(f'Pearson r = {r_:.2f} (10 digits: indicative only)', fontsize=10)
    R.fig(fig, 'relation', 'Each point is a digit. With only ten points a correlation is a '
          'hint, not evidence; the per-image analysis in section 5 is the stronger test.')

    # table
    rows = []
    off = cm.copy(); np.fill_diagonal(off, 0)
    for d in range(10):
        tgt = int(np.argmax(off[d])); src = int(np.argmax(off[:, d]))
        row = [d, n_d[d], *[f'{share[d, k]:.0f}' for k in range(3)],
               f'{theta[assign == d].mean():.1f}' if n_d[d] else '–',
               f'{np.median(m[assign == d]):.1f}' if n_d[d] else '–',
               f'{acc_d[d]:.1f}' + (f' ({acc_d[d] - base["acc_d"][d]:+.1f})' if base else ''),
               f'{fp[d]}' + (f' ({fp[d] - base["fp"][d]:+d})' if base else ''),
               f'{d}→{tgt} ({off[d, tgt]})', f'{src}→{d} ({off[src, d]})']
        rows.append(row)
    R.table(rows, header=['digit', 'neurons', '% A', '% B', '% C', 'mean θ (mV)',
                          'median m', 'accuracy %' + (' (Δ vs base)' if base else ''),
                          'false pos.' + (' (Δ)' if base else ''), 'most lost to',
                          'most taken from'])

    # confusion matrix with the region mix of each predicted digit underneath, columns aligned
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 8.2), sharex=True,
                            gridspec_kw=dict(height_ratios=[3.2, 1]))
    show = cm - base['cm'] if (base and base['same']) else cm
    off_show = show.astype(float); np.fill_diagonal(off_show, np.nan)
    if base and base['same']:
        lim = np.nanmax(np.abs(off_show)) or 1
        axs[0].imshow(off_show, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        ttl = 'Change in confusion counts vs baseline (red = more errors)'
    else:
        axs[0].imshow(off_show, cmap='Reds', aspect='auto')
        ttl = 'Confusion counts (diagonal hidden)'
    for i in range(10):
        for j in range(10):
            if i != j and show[i, j]:
                axs[0].text(j, i, f'{show[i, j]:+d}' if base and base['same'] else show[i, j],
                            ha='center', va='center', fontsize=7)
    axs[0].set_yticks(range(10)); axs[0].set_ylabel('true digit'); axs[0].set_title(ttl, fontsize=10)
    bottom = np.zeros(10)
    for k, r in enumerate(REGIONS):
        axs[1].bar(range(10), share[:, k], bottom=bottom, color=REGION_COLORS[r], label=r,
                   width=0.9)
        bottom += share[:, k]
    axs[1].set_xticks(range(10)); axs[1].set_xlabel('predicted digit')
    axs[1].set_ylabel('% of its neurons'); axs[1].set_ylim(0, 100)
    axs[1].legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.35))
    fig.tight_layout()
    R.fig(fig, 'confusion_vs_regions', 'Each column of the matrix is a predicted digit; the '
          'bar under it is the region mix of that digit\'s neurons. A red column above a red '
          '(region C) bar means the C neurons are pulling other digits to their label.')

    # ---------------------------------------------------------------- 4. activity
    R.h('4. Who does the firing in the test run')
    tot = tc.sum(0)
    rows = []
    for r in REGIONS:
        s = region == r
        rows.append([r, s.sum(), f'{s.mean() * 100:.1f}',
                     f'{tot[s].sum() / max(tot.sum(), 1) * 100:.1f}',
                     f'{tot[s].mean() / len(tl):.3f}' if s.any() else '–',
                     f'{(tot[s] == 0).sum()}' if s.any() else '–'])
    R.table(rows, header=['region', 'neurons', '% of neurons', '% of all test spikes',
                          'spikes per neuron per image', 'silent neurons'])

    # ---------------------------------------------------------------- 5. per image
    R.h('5. Per image: does the most active neuron\'s region predict errors?')
    top = np.argmax(tc, 1)
    has = tc.sum(1) > 0
    rows = []
    for r in REGIONS:
        s = has & (region[top] == r)
        rows.append([r, s.sum(), f'{correct[s].mean() * 100:.1f}' if s.any() else '–'])
    R.table(rows, header=['region of the image\'s most active neuron', 'test images',
                          'accuracy %'])
    # share of the winning digit's spikes coming from C neurons, correct vs wrong
    wd = pred
    c_frac = np.array([tc[k, (assign == wd[k]) & (region == 'C')].sum() /
                       max(tc[k, assign == wd[k]].sum(), 1) for k in range(len(tl))])
    R.p(f'Share of the predicted digit\'s spikes that come from region-C neurons: '
        f'<b>{np.mean(c_frac[correct]) * 100:.1f}%</b> on correct images, '
        f'<b>{np.mean(c_frac[~correct]) * 100:.1f}%</b> on wrong ones. A clearly higher '
        f'share on wrong images means C neurons are pulling predictions to their digit.')
    if base and base['same']:
        flips = dict(fixed=int((~(base['pred'] == tl) & correct).sum()),
                     broken=int(((base['pred'] == tl) & ~correct).sum()))
        R.p(f'Against the baseline on the same images: <b>{flips["broken"]}</b> images '
            f'were right in the baseline and are now wrong; <b>{flips["fixed"]}</b> were '
            f'wrong and are now right.')

    # ---------------------------------------------------------------- 6. s movement
    moved = (s_start is not None and s_end is not None
             and not np.allclose(s_start, s_end, atol=1e-6))
    if moved:
        R.h('6. How s moved during this run (--s-mode adaptive)')
        ds = s_end - s_start
        r_start = np.where(s_start < 0, 'A', np.where(s_start / p['delta'] < a.c_boundary, 'B', 'C'))
        r_end = np.where(s_end < 0, 'A', np.where(s_end / p['delta'] < a.c_boundary, 'B', 'C'))
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
        for d in range(10):
            k = assign == d
            axs[0].scatter(s_start[k], s_end[k], s=10, color=DIGIT_COLORS[d], label=str(d))
        lo_, hi_ = min(s_start.min(), s_end.min()) - .1, max(s_start.max(), s_end.max()) + .1
        axs[0].plot([lo_, hi_], [lo_, hi_], 'k--', lw=.8)
        axs[0].axhline(0, color='grey', lw=.6); axs[0].axvline(0, color='grey', lw=.6)
        axs[0].set_xlabel('s at start (mV)'); axs[0].set_ylabel('s at end (mV)')
        axs[0].legend(fontsize=7, ncol=2, title='digit')
        axs[0].set_title('Each neuron: above the dashed line = became more excitable', fontsize=10)
        axs[1].scatter(tot, ds, s=10, color='C2')
        axs[1].axhline(0, color='k', lw=.8)
        axs[1].set_xlabel('test spikes fired by the neuron'); axs[1].set_ylabel('change in s (mV)')
        if np.std(ds) > 0 and np.std(tot) > 0:
            axs[1].set_title(f'Busy neurons should fall, quiet ones rise (r = '
                             f'{np.corrcoef(tot, ds)[0, 1]:.2f})', fontsize=10)
        R.fig(fig, 's_movement', 'Left: s before and after the run for every neuron, coloured '
              'by assigned digit. Right: change in s against how much the neuron fired.')
        rows = [[f'{x} → {y}', int(((r_start == x) & (r_end == y)).sum())]
                for x in REGIONS for y in REGIONS if ((r_start == x) & (r_end == y)).any()]
        R.table(rows, header=['region at start → at end', 'neurons'])
        rows = [[d, f'{s_start[assign == d].mean():+.2f}', f'{s_end[assign == d].mean():+.2f}',
                 f'{ds[assign == d].mean():+.3f}'] for d in range(10) if (assign == d).any()]
        R.table(rows, header=['digit', 'mean s at start (mV)', 'mean s at end (mV)',
                              'mean change (mV)'])

    # ---------------------------------------------------------------- 7. Gini
    R.h('7. How evenly the firing is shared (Gini)')
    runs = [('this test run', tc, assign)]
    lab_counts = load_activity(a.label_dir)[0] if a.label_dir else None
    if lab_counts is not None:
        runs.append(('this label run', lab_counts, assign))
    if base:
        runs.append(('baseline test run', base['counts'], base['assign']))
        if a.baseline_label_dir:
            bl = load_activity(a.baseline_label_dir)[0]
            if bl is not None:
                runs.append(('baseline label run', bl, base['assign']))
    stats = [(name, sharing_stats(c), asg) for name, c, asg in runs]
    R.table([[name, f"{st['gini']:.3f}", f"{st['top1']:.1f}", f"{st['top10']:.1f}",
              st['silent'], f"{st['spikes_per_image']:.1f}"] for name, st, _ in stats],
            header=['run', 'Gini (0 = equal)', '% of spikes from top 1% of neurons',
                    '% from top 10%', 'silent neurons', 'spikes per image'])
    g_reg = []
    st0 = stats[0][1]
    for r in REGIONS:
        k = region == r
        g_reg.append([r, int(k.sum()), f"{gini(st0['tot'][k]):.3f}" if k.sum() > 1 else '\u2013',
                      f"{st0['tot'][k].sum() / max(st0['tot'].sum(), 1) * 100:.1f}"])
    R.table(g_reg, header=['region', 'neurons', 'Gini within the region (test run)',
                           '% of all test spikes'])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for (name, st, _), c in zip(stats, ('C0', 'C1', 'grey', 'silver')):
        x_, y_ = lorenz(st['tot'])
        axs[0].plot(x_, y_, color=c, lw=1.8, label=f"{name}: Gini {st['gini']:.3f}")
    axs[0].plot([0, 1], [0, 1], 'k--', lw=0.8, label='perfectly equal')
    axs[0].set_xlabel('share of neurons (least active first)'); axs[0].set_ylabel('share of spikes')
    axs[0].legend(fontsize=8); axs[0].set_title('Lorenz curve: the further below the diagonal, '
                                                'the more unequal', fontsize=10)
    test_runs = [(name, st, asg) for name, st, asg in stats if 'test' in name]
    w = 0.8 / len(test_runs)
    for j, ((name, st, asg), c) in enumerate(zip(test_runs, ('C0', 'grey'))):
        gd = [gini(st['tot'][asg == d]) if (asg == d).sum() > 1 else np.nan for d in range(10)]
        axs[1].bar(np.arange(10) + (j - (len(test_runs) - 1) / 2) * w, gd, width=w, color=c,
                   label=name)
    axs[1].set_xticks(range(10)); axs[1].set_xlabel('digit'); axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('Gini within the digit\'s neurons'); axs[1].legend(fontsize=8)
    axs[1].set_title('Within each digit: do a few neurons do all its firing?', fontsize=10)
    R.fig(fig, 'gini', 'Left: how the test (and label) spikes are spread over all neurons. '
          'Right: the same within the neurons assigned to each digit. Higher Gini = a few '
          'neurons take over, the failure the adaptive threshold exists to prevent.')
    if base:
        dg = stats[0][1]['gini'] - [st for name, st, _ in stats if name == 'baseline test run'][0]['gini']
        R.p(f'Gini of this test run minus the baseline test run: <b>{dg:+.3f}</b> '
            f'(negative = firing more evenly shared than the baseline).')

    # ---------------------------------------------------------------- csv
    csv_path = os.path.join(a.out, 'neurons.csv')
    sel = rates.max(1) / np.maximum(rates.sum(1), 1e-9)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['neuron', 'assigned_digit', 'theta_mV', 'detector_mV', 'Vx_mV', 'm',
                     'region', 's_start_mV', 's_end_mV', 'selectivity', 'test_spikes',
                     'times_most_active'])
        tm = np.bincount(top[has], minlength=n_e)
        for j in range(n_e):
            wr.writerow([j, assign[j], f'{theta[j]:.3f}', f'{vth[j]:.3f}', f'{vx[j]:.3f}',
                         f'{m[j]:.3f}', region[j],
                         f'{s_start[j]:.4f}' if s_start is not None else '',
                         f'{s_end[j]:.4f}' if s_end is not None else '',
                         f'{sel[j]:.3f}', int(tot[j]), int(tm[j])])
    R.p(f'Per-neuron table: <code>{os.path.abspath(csv_path)}</code>')
    with open(os.path.join(a.out, 'regions.json'), 'w') as f:
        json.dump(dict(params=p, vd_rule=vx_desc,
                       sharing={name: {k: v for k, v in st.items() if k != 'tot'}
                                for name, st, _ in stats}, Vx_min=float(np.nanmin(vx)),
                       Vx_max=float(np.nanmax(vx)), c_boundary=a.c_boundary, s_at=a.s_at,
                       s_start_mean=float(s_start.mean()) if s_start is not None else None,
                       s_end_mean=float(s_end.mean()) if s_end is not None else None,
                       neurons_per_region={r: int((region == r).sum()) for r in REGIONS},
                       per_digit={str(d): dict(neurons=int(n_d[d]),
                                               share=dict(zip(REGIONS, share[d].round(1).tolist())),
                                               accuracy=float(acc_d[d]), false_pos=int(fp[d]))
                                  for d in range(10)}), f, indent=2)
    path = R.save(f'EIF regions: Δ {p["delta"]} mV, Vx {vx_text}')
    print(f'{vx_desc} | Vx {vx_text} | A {(region == "A").sum()}  B {(region == "B").sum()}  '
          f'C {(region == "C").sum()} | accuracy {correct.mean() * 100:.2f}%')
    print('Report:', os.path.abspath(path))


if __name__ == '__main__':
    main()
