# Benchmark dashboard (device=mps)

| Config | Accuracy | Speedup |
|---|---|---|
| small (bs=4,seq=64,d=128) | PASS | 1.22x |
| default (bs=8,seq=128,d=512) | PASS | 1.15x |
| large batch (bs=64,seq=32,d=256) | PASS | 1.13x |
| long seq (bs=2,seq=1024,d=256) | PASS | 1.88x |
| causal+padding (bs=8,seq=128,d=512) | PASS | 1.07x |
