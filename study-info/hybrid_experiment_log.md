## Hybrid Landing-Digit Optuna Studies

| Version | Trials | Notable Fixed Params | Search Space (key vars) | Best Acc % |
|---------|--------|----------------------|-------------------------|-----------|
| V2 | 40 | batch 32, n_heads 8, kernel 9, stride 2, no MixUp | lr, depth {3,4,6}, latent {64,96}, dropout 0-0.2, wd 1e-6–1e-2 | 25.04 |
| V3 | 40 | batch 32, n_heads 8, kernel 9/5, stride 2/4 | lr log-uniform 1e-5–1e-3, depth {3,4,6}, latent {64,96}, dropout 0-0.2, wd 1e-6–1e-2 | 25.04 |
| V4 | 40 | batch 32, n_heads 8, kernel 9, stride {7,9} | lr 2e-5–2e-4, depth {2,3,4}, latent {64,96,128}, dropout 0-0.06, MixUp 0.1-0.4, wd 1e-7–1e-5 | 24.27 |
| V5 | 50 (running) | same as V4 + augmentation knobs | shift_p 0-0.7, time_mask_p 0-0.6, chan_mask_p 0-0.6, noise_std 0.005-0.05 | 24.48 (so far) |

> Log created 2025-07-22.  Update after each study finish. 