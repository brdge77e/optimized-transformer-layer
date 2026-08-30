# Benchmark dashboard (device=cuda)

| Config | Accuracy | Speedup |
|---|---|---|
| small (bs=4,seq=64,d=128) | PASS | 2.03x |
| default (bs=8,seq=128,d=512) | PASS | 1.13x |
| large batch (bs=64,seq=32,d=256) | PASS | 1.14x |
| long seq (bs=2,seq=1024,d=256) | PASS | 2.03x |
| causal+padding (bs=8,seq=128,d=512) | PASS | 1.14x |
