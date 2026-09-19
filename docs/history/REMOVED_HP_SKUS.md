# Removed hp SKUs (light / champ)

**Date:** 2026-09-19  
**Decision:** CyphaLM ships one hp compile profile — **gate24** only.

## What gate24 is

- Full vendored `v78_flags.ps1` feature set (78/78 grow flags ON)
- `HP_SLOT_MAX=24` at compile time
- `CYPHA_HP_XSIMD=ON` by default (SSE4.1)
- enwik8.8mb measured BPC (Cypha vendored hp, 2026-09-19): **observe 1.611729**, **archive 1.611759**

## Removed profiles

| Profile | Why removed |
|---------|-------------|
| **light** | Bare `features.hpp` defaults (0/78 v78 flags). Fast CI/dev path but **~1.72 BPC** on enwik8MB — not the quality bar. |
| **champ** | `v78_flags.ps1` + `HP_SLOT_MAX=35`. ~**1.610** BPC class but **~15 GB RSS** @ mem 22; OOM-prone on typical dev VMs. Marginal gain vs gate24 not worth SKU complexity. |

## Removed machinery

- `CYPHA_HP_PROFILE` CMake matrix (`light` / `gate24` / `champ`)
- `CYPHA_HP_CHAMP_BUILD`, `apply_hp_champ_recipe()`, `apply_hp_gate24_recipe()`
- `ContextMode::HpChamp`, `ContextMode::HpGate24`
- `bench/config/profiles/cyphalm_hp_champ.json`
- Multi-SKU scripts (`measure_enwik_skus.sh`, `measure_enwik_sku_bpc.sh`)

CLI aliases `gate24`, `champ`, `hp_champ` still parse to `ContextMode::Hp` for old JSON; all map to the same gate24 binary.

## History

Prior dual-SKU work landed on PR #5 (`cursor/bit-tree-gate24-eval-9d44`). Git history retains light/champ CMake and measurement artifacts under `docs/reports/enwik_sku_profiles/` where archived.
