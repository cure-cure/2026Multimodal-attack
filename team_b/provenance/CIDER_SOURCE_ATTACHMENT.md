# CIDER source attachment

This record pins the exact upstream source used by Task B. It is a provenance attachment, not a claim that Task B is a byte-for-byte reproduction.

| field | value |
|---|---|
| repository | `https://github.com/PandragonXIII/CIDER` |
| fixed commit | `3c08d7e2d52e27b47d1e428a030e8ffda5efae3e` |
| Git tree | `2ca454d00bdeddad272397feb905f576bae1be52` |
| commit date | `2025-01-10T19:06:40+08:00` |
| commit subject | `Merge branch 'More-models'` |
| license blob | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |
| README blob | `de83d994d951070c74599f0ddeb94f97a431eebc` |
| validation CSV blob | `31bacaab5ddff0a4cca5061ebeb62adcfd8b36b4` |
| test CSV blob | `228b1aed50a8324badde27b2b6404931f9133eb5` |

The pinned CSV files have no header and contain six columns. The upstream HarmBench `_ID` is column 6 (zero-based index 5). Earlier Task-B package IDs were derived with a row fallback after reading index 6. Existing `BC...` sample IDs are intentionally preserved; regenerated packages add `id_mapping.csv` from each existing ID to the correct upstream `_ID`. The verified 2,018-row mapping for the existing artifacts is attached as [`CIDER_EXISTING_ID_MAPPING.csv`](CIDER_EXISTING_ID_MAPPING.csv).

Actual model inputs recorded for the run:

- LLaVA: `llava-hf/llava-1.5-7b-hf` at revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, loading only the vision tower, multimodal projector and token embedding components.
- Diffusion checkpoint: canonical OpenAI `256x256_diffusion_uncond.pt`; local SHA256 `a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1`.

Task B is marked `adapted`: it preserves the CIDER statistic direction and official split protocol, but uses an in-memory engineering path and a reduced LLaVA component loader. See the run/freeze record for exact artifacts.
