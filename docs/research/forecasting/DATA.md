# Forecasting: data

**Odin Loch** · 2026-10-29 · [FORECAST_PLAN](../../FORECAST_PLAN.md) FC2, FC3, FC4 · audit: [FC0](../../reports/FC0_FORECAST_AUDIT.md) §1 A, F, G

What the module reads, where it comes from, under what licence, in what format, and how each file is checked. The maps in §5 and §6 are proposals, registered for Odin's veto before FC3 and FC4 run.

---

## 1. Format rules

1. **UTF-8, no BOM, LF line ends, one header row.** FC2 adds `*.csv text eol=lf working-tree-encoding=UTF-8` to `.gitattributes`. A CTest (`native_forecast_data_utf8`) fails on any byte sequence under `bench/data/**/*.csv` that is not valid UTF-8, and on a UTF-16 BOM or NUL byte.
2. **Columns are found by header name**, never by position. A missing required column is an error.
3. **Every cell parses or the load fails.** The error names the file, row and column. `parse_int` returning 0 on garbage is gone.
4. **Facts only, no labels.** Committed files hold observed values. Targets, lags and splits are computed in code (`panel.hpp`), in one place, where the origin rule is enforced.
5. **Sorted by period**, then by unit. Loaders check the order and fail if it is broken.
6. **Missing is not zero.** Each file states whether its rows are complete (every unit-period present, zeros written) or sparse (only nonzero). `ucdp_cm.csv` is complete, because VIEWS scores zero months.

---

## 2. Sources

| Dataset | Version | Coverage | Unit | Terms | Use |
|---|---|---|---|---|---|
| UCDP Georeferenced Event Dataset (GED) | 26.1 | 1989–2025 | Event, aggregated to country-month | CC BY 4.0 ([UCDP](https://ucdp.uu.se/downloads/)) | FC7 target and inputs |
| UCDP Candidate Events | 26.0.x monthly (26.0.9: Jan–Sep 2026) | Current year, about a month's lag | Event | CC BY 4.0 | FC7 true-future truth |
| UCDP/PRIO Armed Conflict Dataset | 26.1 | 1946–2025 | Conflict-year | CC BY 4.0 | War-level years (§5), checks |
| CoW Militarized Interstate Disputes | 5.0 (MIDA, MIDB 1816–2014; MIDI, MIDIP 1993–2014) | 1816–2014 | Dispute, participant, incident | Cite the version and article of record ([CoW](https://correlatesofwar.org/data-sets/MIDs)) | FC5 |
| CoW dyadic MID | 4.03 | 1816–2014 | Dyad-dispute | As above | FC5 dyad-years |
| International Crisis Behavior | 16 | 1918–2021; 512 crises, 1,131 actors, 1,000 crisis dyads | Crisis, actor, dyad | Cite Brecher and Wilkenfeld and the v16 codebook ([ICB](https://sites.duke.edu/icbdata/data-collections/)) | FC5 feature and check |
| GDELT 1.0 event files | Daily since 1 April 2013 | 2013– | Event | "100% free and open" ([GDELT](https://gdeltproject.org/data.html)) | FC8 sequences, FC9 |
| GDELT 2.0 event files | 15-minute updates, 65 translated languages | 2015– | Event | As above | FC9 live feed (optional) |
| VIEWS challenge definitions | 2023/24 | — | — | Paper ([Hegre et al. 2024](https://arxiv.org/pdf/2407.11045)) | FC6, FC7 protocol and benchmarks |

**Network.** The cloud container cannot reach these hosts. Fetching runs on Odin's PC (FORECAST_PLAN §8.5). The container and CI read the committed extracts.

**Revisions.** UCDP revises past values between versions, and GED 26.1 is not GED 25.1. Every score names the data version it was computed on. A version change is a new read, never an overwrite of an old one.

---

## 3. Provenance manifest

`bench/data/forecast/MANIFEST.json`. One entry per committed file. A CTest (`native_forecast_manifest`) checks every sha256 and row count, and fails on any file not listed.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://cypha.local/schemas/forecast_data_manifest.schema.json",
  "title": "bench/data/forecast/MANIFEST.json (proposed, lands with FC3)",
  "type": "object",
  "required": ["schema", "files"],
  "properties": {
    "schema": { "const": 1 },
    "files": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["path", "sha256", "rows", "kind", "source", "licence", "retrieved", "derivation"],
        "properties": {
          "path": { "type": "string" },
          "sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
          "rows": { "type": "integer", "minimum": 1 },
          "kind": { "enum": ["real_extract", "synthetic"] },
          "complete": { "type": "boolean", "description": "true when every unit-period is present, zeros written" },
          "source": {
            "type": "object",
            "required": ["name", "version", "url"],
            "properties": { "name": { "type": "string" }, "version": { "type": "string" }, "url": { "type": "string" } }
          },
          "licence": { "type": "string" },
          "citation": { "type": "string" },
          "retrieved": { "type": "string", "format": "date" },
          "derivation": { "type": "string", "description": "the script and arguments that made this file from the source" }
        }
      }
    }
  }
}
```

---

## 4. Committed files (proposed, FC3)

### 4.1 `ucdp_cm.csv`: state-based fatalities, country-month (complete)

| Column | Type | Meaning |
|---|---|---|
| `gwcode` | int | Gleditsch–Ward country code |
| `year` | int | 1989–2025 |
| `month` | int | 1–12 |
| `sb_best` | int ≥ 0 | Sum of GED `best` for state-based events (type 1) in the country-month; 0 written |

- **Which rows exist.** Every GW state alive in that month has one row. That needs the GW state list (and CShapes for borders, if FC7 uses neighbours).
- **Size.** About 200 states × 444 months, roughly 89,000 rows and 2 MB.

### 4.2 `mid_dyad_dispute.csv`: dyadic MID 4.03 (sparse)

| Column | Type | Meaning |
|---|---|---|
| `disno` | int | Dispute number |
| `ccode_a`, `ccode_b` | int | CoW country codes, a < b |
| `start_year`, `end_year` | int | Dyadic dispute years |
| `hihost` | int 1–5 | Highest hostility level in the dyadic dispute |
| `fatality` | int | CoW fatality level code |

- **What code builds from it.** Dyad-years for FC5. Target A is the onset in *t* + 1 of a dyadic MID with `hihost` ≥ 4. Target B is any onset.
- **Why onsets.** Dispute-level hostility cannot be spread over a dispute's years before 1993. Incident data (MIDI) starts in 1993, so the targets are onsets rather than within-dispute escalation. Within-dispute escalation on MIDI 1993–2014 is an optional Target C.

### 4.3 `gdelt_theater_week.csv`: GDELT root counts (complete over theater × week × root)

| Column | Type | Meaning |
|---|---|---|
| `week_start` | date | ISO week, Monday |
| `theater` | string | `GLB`, `TWN`, `UKR`, `MID`, `PRK`, `SAS`, `AFR` |
| `root` | int 1–20 | GDELT `EventRootCode` |
| `n_events` | int ≥ 0 | Events with `IsRootEvent` = 1 assigned to the theater (§6) |
| `n_mentions` | int ≥ 0 | Sum of `NumMentions` |
| `goldstein_mean` | float | Mean `GoldsteinScale`, empty when `n_events` is 0 |

- **A companion file.** `gdelt_week_totals.csv` holds `week_start` and `n_events_all`, the global count. FC8 uses it to normalise for GDELT's changing volume.
- **Presence threshold.** A root is present in a theater-week when `n_events` ≥ τ(theater, root). τ is the median of that pair's nonzero weekly counts on train (to 2019). This is the default, tunable on dev, and τ is stored in the bundle.
- **Size.** About 7 × 20 × 660 weeks, roughly 92,000 rows.

### 4.4 `icb_crisis.csv`: ICB v16 crises (sparse)

The crisis number, trigger and termination years, great-power involvement, the violence level, and the actors as CoW codes. Variable names follow the v16 codebook and are checked against it in FC3.

### 4.5 `theaters.csv`: country → theater rules

The table in §6, as data, so it is versioned and hashed with the rest.

### 4.6 `synthetic_*.csv`: today's placeholders

The current `sample_*` and alias files, re-encoded to UTF-8 and renamed. `kind: synthetic` in the manifest. They stay for fast smokes only and never feed a reported score (FORECAST_PLAN rule 7).

---

## 5. CAMEO root → escalation level (proposed for FC4; registered for veto)

The root names are from the [CAMEO codebook](https://parusanalytics.com/eventdata/cameo.dir/cameocontents.html).

QuadClass is GDELT's four-way grouping: 1 verbal cooperation, 2 material cooperation, 3 verbal conflict, 4 material conflict. The column below shows the usual grouping by root. FC3 reads GDELT's own `QuadClass` column rather than this one.

The levels are the existing `EscalationLevel` names, in their order: Calm < Verbal < Material < ForceThreat < ForceUse < War.

| Root | Name | QuadClass | Today | Proposed |
|---|---|---|---|---|
| 01 | Make public statement | 1 | Verbal | Calm |
| 02 | Appeal | 1 | Verbal | Calm |
| 03 | Agree / express intent to cooperate | 1 | Verbal | Calm |
| 04 | Consult | 1 | Material | Calm |
| 05 | Support diplomatically | 1 | Material | Calm |
| 06 | Cooperate (material) | 2 | Material | Calm |
| 07 | Provide aid | 2 | ForceThreat | Calm |
| 08 | Yield | 2 | ForceThreat | Calm |
| 09 | Investigate | 3 | ForceUse | Verbal |
| 10 | Demand | 3 | ForceUse | Verbal |
| 11 | Disapprove | 3 | ForceUse | Verbal |
| 12 | Reject | 3 | ForceUse | Verbal |
| 13 | Threaten | 3 | ForceUse | ForceThreat |
| 14 | Protest | 4 | ForceUse | Verbal |
| 15 | Exhibit military posture | 4 | War | ForceThreat |
| 16 | Reduce relations | 4 | War | Material |
| 17 | Coerce | 4 | War | Material |
| 18 | Assault | 4 | War | ForceUse |
| 19 | Fight | 4 | War | ForceUse |
| 20 | Unconventional mass violence | 4 | War | ForceUse |

- **War** comes from no single event code. A theater-year is at War when a member state is in a UCDP/PRIO conflict at intensity 2 (at least 1,000 battle-related deaths in the year). This keeps the top level tied to fatalities, not to how news is coded.
- **Protest (14)** is mostly civil unrest. It is Verbal in the escalation scale, even though GDELT groups it with material conflict.
- **Material** here means non-violent material pressure: reducing relations and coercion.
- **The root column.** FC4 reads `EventRootCode` directly. `cameo_code / 10` is removed (FC0 F2).

---

## 6. Theaters (proposed for FC3; registered for veto)

An event is assigned by the countries of its two actors (`Actor1CountryCode`, `Actor2CountryCode`). These are 3-letter CAMEO country codes, mostly ISO 3166 alpha-3, mapped through one committed table. Never by actor names (FC0 G1). The rules are tried in this order, and the first match wins.

| Order | Theater | Event belongs if |
|---|---|---|
| 1 | `UKR` | Both actors in {RUS, UKR, BLR}, or one in {RUS, UKR} and the other a NATO member |
| 2 | `MID` | Both in {ISR, PSE, IRN, LBN, SYR, YEM, IRQ, JOR}, or one in that set and the other USA |
| 3 | `PRK` | One actor PRK, or the pair is KOR with JPN |
| 4 | `TWN` | One actor TWN, or the pair is CHN with one of {USA, JPN, PHL} |
| 5 | `SAS` | Both in {IND, PAK, AFG}, or the pair is CHN with IND |
| 6 | `AFR` | Both in {SDN, SSD, ETH, ERI, SOM, DJI} |
| 7 | `GLB` | Both in {USA, RUS, CHN, GBR, FRA} and not matched above |
| — | unassigned | Dropped from the theater tables and counted in the totals file |

- **Precedence.** USA–CHN without Taiwan is `GLB` under these rules. In the sample data it was `TWN`; the change is deliberate and is part of this proposal.
- **Dyad rows.** For CoW-coded dyads, the same table is applied after mapping CoW codes to ISO3 with the CoW state list.
- **Unit order in the code.** The `Theater` enum order (Global, Taiwan, Ukraine, Middle East, Korea, South Asia, Africa) is unchanged, and so are FC4's ids.

---

## 7. The fetch (FC3)

`scripts/fetch_forecast_data.ps1` stays for Windows. `scripts/fetch_forecast_data.sh` is added, with the same steps. Both write UTF-8, no BOM, LF, and append to `MANIFEST.json`.

| Step | Today | FC3 |
|---|---|---|
| GDELT root | `EventCode` (field 26), then `/ 10` in C++ | `EventRootCode` and `QuadClass` columns |
| GDELT theater | Substring match on actor names | Country-code rules (§6) |
| GDELT range | First 5,000 rows of one day | Every daily file from 2013-04-01 to the cut date, aggregated to theater-week |
| MID | Dyadic 4.03 with theater, great_power, duration and gdp_ratio constant; `escalated` from the previous row | Dyadic 4.03 columns as in §4.2, no derived labels |
| VIEWS-style | Sums sb + os + ns; drops zero months; 2020 onward | UCDP GED state-based only, complete with zeros (§4.1) |
| `-Repair` | Copies UTF-16 samples over UTF-16 aliases | Removed; the encoding CTest replaces it |

---

## 8. Traps

1. **GDELT volume.** The number of sources GDELT reads changes over time, so raw counts trend. FC8 uses counts relative to `gdelt_week_totals.csv` and logs. EVALUATION §4 checks the base rate's drift across splits.
2. **Three country-code systems.** UCDP uses Gleditsch–Ward, CoW uses CoW codes, and GDELT uses ISO3. Conversions go through one table each, committed and hashed. Never through name matching.
3. **MID stops in 2014.** FC5's test window is 2002–2014. Nothing in FC5 claims anything about later years.
4. **The ICB dyad file is "updated" separately from v16.** The manifest names which one is used.
5. **Candidate is not GED.** True-future scores against UCDP Candidate are marked as such and rescored when GED covers the window (EVALUATION §6).
