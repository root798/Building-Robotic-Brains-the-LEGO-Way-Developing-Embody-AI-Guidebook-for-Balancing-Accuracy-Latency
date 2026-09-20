# Known issues

| Item | Current treatment |
| --- | --- |
| BDD RepPoints v2 | Training contains nonfinite classification losses; AP is retained as a flagged result. The separate v3 FP32-loss repair is in progress. |
| BDD RepPoints preprocessing | v2/v3 retain duplicate normalization and channel conversion from the inherited recipe. Correcting this requires a separate training revision. |
| BDD DiffusionDet | The available checkpoint differs from the one identified by the recorded evaluation. Its historical AP is tied to the recorded digest; matching weights are unavailable. |
| Original BDD RepPoints and EfficientDet | No detections in the recorded evaluation. Replacement recipes are listed separately. |
| WAID ViTDet-B | Training is ongoing; final AP is pending. |
| Legacy configurations | Some use legacy transforms or omitted sampler settings and need the compatible historical MMDetection checkout. Exports preserve the recipes. |

The catalog contains results and checkpoint identities. Binary weight downloads
are not hosted yet. These accuracy records do not include a latency/energy release.
