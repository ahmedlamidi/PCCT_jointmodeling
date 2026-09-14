"""Reference Monte Carlo for pile-up. Two counting modes.

  mode='seminonparalyzable'  Yang, Pelc & Wang, Med Phys 52(6):3658-3674 (2025),
                             doi 10.1002/mp.17746. Fixed dead time tau after each
                             trigger; if the signal is still above threshold when
                             it expires, the next trigger fires immediately.

  mode='paralyzable'         Bierme & Roessl (2012); Roessl et al (2016). Dead
                             time EXTENDS: every photon arriving while the signal
                             is above threshold prolongs the busy period. One
                             count per contiguous above-threshold excursion,
                             recorded at the peak of that excursion. tau is not
                             used. This is what Morovati et al. 2025 use.

The two agree below lam*tau ~ 0.1 and diverge hard above it: at lam*tau = 2.45
the paralyzable model keeps 8.7% of counts, the non-paralyzable family 29%.

Original docstring follows.
--------------------------------------------------------------------------
Reference Monte Carlo for pile-up, matched to Yang's detector model.

Yang's Algorithm 1: discretise time, build the delta pulse train, convolve with
the pulse kernel, then run a non-paralyzable digital counter. Retriggering falls
out naturally -- if the signal is still above threshold when the dead time ends,
the next trigger fires immediately. That is the "seminonparalyzable" behaviour.

Unlike pileup.mc_frames (which used a square window and summed energies), this
uses the SAME triangular pulse as pileup_yang, so the two are comparable.
"""
import numpy as np


def tri_kernel(t_p, T, dt):
    n = int(np.ceil(T / dt)) + 1
    t = np.arange(n) * dt
    k = np.where(t < t_p, t / t_p, np.clip((T - t) / (T - t_p), 0, None))
    return np.clip(k, 0, None)


def mc_pulse_train(dep_pmf, vEo, rate, tau_dead, t_p, T_pulse, Et, eth,
                   n_frames, frame_T, rng, dt=0.5e-9, max_counts=400_000,
                   mode='seminonparalyzable'):
    """Returns counts (n_frames, Nl) and the recorded pulse-height spectrum.

    mode='paralyzable' ignores tau_dead: the busy period is however long the
    signal stays above Et, so overlapping pulses merge into a single count.
    """
    if mode not in ('seminonparalyzable', 'paralyzable'):
        raise ValueError('unknown mode %r' % mode)
    m = vEo >= Et
    p = dep_pmf[m]
    if p.sum() <= 0:
        return np.zeros((n_frames, len(eth))), np.zeros(0)
    p = p / p.sum(); e = vEo[m]
    eth = np.asarray(eth, float); hi = np.append(eth[1:], np.inf)

    L = int(frame_T / dt)
    kern = tri_kernel(t_p, T_pulse, dt)
    n_dead = max(int(round(tau_dead / dt)), 1)
    out = np.zeros((n_frames, len(eth))); recs = []
    for f in range(n_frames):
        n = rng.poisson(rate * frame_T)
        sig = np.zeros(L + len(kern))
        if n:
            idx = np.clip((rng.random(n) * frame_T / dt).astype(int), 0, L - 1)
            E = rng.choice(e, size=n, p=p)
            np.add.at(sig, idx, E)
        s = np.convolve(sig, kern)[:L]
        if mode == 'paralyzable':
            # one count per contiguous above-threshold excursion, at its peak.
            # The busy period extends itself whenever a new pulse lands inside
            # it, which is exactly the extending-dead-time definition.
            m = s >= Et
            if m.any():
                edge = np.diff(m.astype(np.int8))
                starts = np.flatnonzero(edge == 1) + 1
                ends = np.flatnonzero(edge == -1) + 1
                if m[0]:
                    starts = np.r_[0, starts]
                if m[-1]:
                    ends = np.r_[ends, L]
                # Peak of every excursion at once. A Python loop here is the
                # bottleneck at high flux -- air produces thousands of excursions
                # per frame, which made the grid build unusably slow.
                if len(starts):
                    # reduceat runs each segment from starts[k] to starts[k+1],
                    # which is longer than the excursion -- but the extra span is
                    # BELOW threshold by construction, and the excursion peak is
                    # >= threshold, so the maximum is unchanged. No correction
                    # needed, and the whole frame is one vectorised call.
                    pk = np.maximum.reduceat(s, starts)
                    recs.extend(pk.tolist())
                    keep = pk >= eth[0]
                    if keep.any():
                        np.add.at(out[f], np.searchsorted(eth, pk[keep], 'right') - 1, 1)
        else:
            above = np.flatnonzero(s >= Et)
            ptr = 0
            while ptr < len(above):
                i = above[ptr]
                j = min(i + n_dead, L)
                rec = s[i:j].max()
                recs.append(rec)
                if rec >= eth[0]:
                    out[f, np.searchsorted(eth, rec, 'right') - 1] += 1
                ptr = np.searchsorted(above, j)
        if len(recs) > max_counts:
            out = out[:f + 1]
            break
    return out, np.asarray(recs)
