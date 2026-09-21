| Mode | Total Params (M) | Active Params (M) | MACs (G) | FLOPs (G) | Mean Latency (ms) | P95 Latency (ms) | Peak Alloc. (MiB) | Extra Alloc. (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_bridge | 21.622 | 19.517 | 9.314 | 18.628 | 11.857 | 14.414 | 219.0 | 128.3 |
| full | 21.622 | 21.618 | 9.316 | 18.632 | 12.145 | 12.758 | 219.0 | 128.3 |
| ours | 21.622 | 21.618 | 9.316 | 18.632 | 12.011 | 12.599 | 219.0 | 128.3 |

Notes:
- End-to-end inference includes the feature extractor, bridge, decoder, and classifier.
- Active parameters count only parameters in modules invoked by the selected inference graph.
- FLOP/MAC tools may omit custom CUDA point-cloud operators such as FPS, neighbor search, and grouping.
