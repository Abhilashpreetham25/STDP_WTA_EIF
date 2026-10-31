# Next steps: from rate coding to inter-spike-interval coding

## Where the current network spends its spikes

| | now (rate code) |
|---|---|
| image presentation | 350 ms + 150 ms rest |
| input spikes per image | about 2,300 (Poisson, rate ∝ pixel, expected count at the starting intensity) |
| output spikes per image | about 18 |
| images shown again because of too few spikes | 1.4–2% |
| readout | spike count per digit; timing ignored |

The EIF experiments ([experiments.md](experiments.md)) show the cost of this. EIF changes *when* a neuron fires. It gives a sharper onset, and the homeostatic version lets more neurons take part (lower Gini). A count over 350 ms averages that difference away, so EIF could only match LIF.

Biology does not work this way. People can categorise a natural image in about 150 ms (Thorpe et al., 1996). That leaves each neuron along the visual pathway time for only a few spikes. In the retina, the relative timing of the first spikes already encodes the image (Gollisch & Meister, 2008). Cortical neurons fire short bursts whose intervals carry information separately from single spikes (Izhikevich et al., 2003; Naud & Sprekeler, 2018).

## The idea: 5–8 spikes per pixel in 150 ms

Each active pixel sends a short train of 5–8 spikes within a 150 ms window. Its intensity is coded in the inter-spike interval (ISI): a bright pixel fires early with short intervals, a dim pixel later with long intervals. With 6 spikes, for example:

    ISI(p) = d_min + (d_max − d_min) · (1 − p/255),    d_min ≈ 2 ms,  d_max ≤ 150 ms / 5 = 30 ms

| | rate (now) | first spike (TTFS) | ISI, 5–8 spikes |
|---|---|---|---|
| window | 350 ms | tens of ms | 150 ms |
| input spikes per image (about 150 active pixels) | about 2,300 | about 150 | about 750–1,200 |
| what one lost or shifted spike costs | almost nothing | that pixel's whole value | one of 4–7 intervals |
| what a neuron can measure | count | one time | a time and several intervals |

## What the evidence says, honestly

The closest test to this network is Guo et al. (2021). They compared rate, time-to-first-spike (TTFS), phase and burst coding in an **unsupervised STDP network on MNIST**:

| coding | accuracy | inference latency | inference spikes (×10⁷) | strength |
|---|---|---|---|---|
| rate | 87.46% | 150 ms | 0.997 | – |
| TTFS | 88.57% | 20 ms | 0.152 | fastest, fewest spikes |
| phase | 88.18% | 30 ms | 5.782 | most robust to input noise |
| burst (short-ISI groups) | 88.39% | 30 ms | 0.570 | fastest learning convergence; best compression and tolerance of hardware faults |

What this means for the ISI idea:
- **Against rate coding, an interval code wins on all three counts.** Inference is 5× faster, uses 1.7× fewer spikes, and is about one point more accurate. On MNIST, Park et al. (2019) also found burst coding faster and more economical than rate coding: 27 instead of 200 time steps at the same accuracy.
- **Against first-spike coding, it does not win on speed or spike economy.** One spike per input is the minimum, and TTFS was also slightly more accurate. What an interval code gains over first-spike is **robustness**: several intervals average out timing jitter, and losing one spike does not lose the input. Guo et al. found TTFS to be the code most sensitive to synaptic noise during training.

So the claim I will test is that **5–8 spikes in 150 ms keeps most of the speed and spike economy of first-spike coding, while being much more robust to noise and lost spikes than it. It also beats rate coding on all three counts.** With few spikes per neuron, accuracy does not have to drop: Stöckl & Maass (2021) classified images with about two spikes per neuron by using spike timing.

## Why STDP and EIF fit this code

- **STDP sees intervals without changes.** The presynaptic trace in this network decays with τ = 20 ms and is reset to 1 at each input spike. A bright pixel's short-interval train keeps the trace high whenever the neuron fires, so STDP still strengthens the brightest pixels. With 2–30 ms intervals, the trace time constants already sit in the right range.
- **EIF should help timing more than counting.**
  - A sharper spike onset lets a neuron follow fast input changes (Fourcaud-Trocmé et al., 2003).
  - EIF models fitted to real cortical neurons predict their spike times accurately (Badel et al., 2008).
  - In my runs, the homeostatic EIF network shared its firing more evenly (Gini 0.24 vs 0.30), and fewer images needed a repeat presentation (138 vs 182 per 10,000 test images). More neurons crossing threshold early is what a timing readout needs.

  Whether this makes EIF beat LIF is the hypothesis, not a result yet.

## Plan

**Step 0: close the open issues**
- Rerun the E3 label and test passes at Δ = 0.5 mV, the training value, and add a second seed.
- Replace the Euler update of v with a semi-implicit one, so strong inhibition cannot make it unstable.
- Retune r0 to the measured firing rate.

**Step 1: measure timing in the existing network (no retraining)**
- Run the test passes with `--save-spikes` for LIF and EIF.
- Measure first-spike latency and its spread, and accuracy from the first k output spikes (k = 1, 3, 5, 10).
- See how early each network can decide.

**Step 2: input codes**
- Add the input codes to the controller of `dc_standalone.py` / `EIF.py` as an option: rate (now), TTFS, and ISI with 5–8 spikes in a 150 ms window.
- Keep STDP, normalisation and θ unchanged.

**Step 3: matched comparison**
- 3 codes × {LIF, homeostatic EIF} × {100, 225, 400, 600} neurons, 2 seeds each.
- Measure:
  - accuracy (paper protocol);
  - time to decision;
  - input and output spikes per image;
  - how many neurons the network actually uses (Gini, near-duplicate receptive fields).
- Robustness: add timing jitter (σ = 1, 2, 5 ms) and delete 10% or 20% of input spikes.

**Step 4: early decision readout**
- Stop an image as soon as one digit's neurons reach k spikes, and read the decision from the first output spikes and their intervals. Easy images then finish early.

**What would confirm the idea:**
- ISI reaches rate-code accuracy in under half the time with about half the input spikes;
- ISI loses clearly less accuracy than TTFS under jitter and spike deletion;
- EIF beats LIF on accuracy or robustness under ISI, at 600 neurons or fewer.

## References

- Badel, L., Lefort, S., Berger, T. K., Petersen, C. C. H., Gerstner, W. & Richardson, M. J. E. (2008). Extracting non-linear integrate-and-fire models from experimental data using dynamic I–V curves. *Biol. Cybern.* 99:361–370.
- Fourcaud-Trocmé, N., Hansel, D., van Vreeswijk, C. & Brunel, N. (2003). How spike generation mechanisms determine the neuronal response to fluctuating inputs. *J. Neurosci.* 23(37):11628–11640.
- Gollisch, T. & Meister, M. (2008). Rapid neural coding in the retina with relative spike latencies. *Science* 319:1108–1111.
- Guo, W., Fouda, M. E., Eltawil, A. M. & Salama, K. N. (2021). Neural coding in spiking neural networks: a comparative study for robust neuromorphic systems. *Front. Neurosci.* 15:638474.
- Izhikevich, E. M., Desai, N. S., Walcott, E. C. & Hoppensteadt, F. C. (2003). Bursts as a unit of neural information: selective communication via resonance. *Trends Neurosci.* 26(3):161–167.
- Masquelier, T. & Thorpe, S. J. (2007). Unsupervised learning of visual features through spike timing dependent plasticity. *PLoS Comput. Biol.* 3(2):e31.
- Naud, R. & Sprekeler, H. (2018). Sparse bursts optimize information transmission in a multiplexed neural code. *PNAS* 115(27):E6329–E6338.
- Park, S., Kim, S., Choe, H. & Yoon, S. (2019). Fast and efficient information transmission with burst spikes in deep spiking neural networks. *Proc. DAC 2019*.
- Stöckl, C. & Maass, W. (2021). Optimized spiking neurons can classify images with high accuracy through temporal coding with two spikes. *Nat. Mach. Intell.* 3:230–238.
- Thorpe, S., Fize, D. & Marlot, C. (1996). Speed of processing in the human visual system. *Nature* 381:520–522.
- Thorpe, S., Delorme, A. & Van Rullen, R. (2001). Spike-based strategies for rapid processing. *Neural Networks* 14(6–7):715–725.
