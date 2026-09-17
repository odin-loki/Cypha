| game_benchmark.py Cypha HRNA Synthetic Game Theory Benchmark Chess Poker Go 150,000 examples deep component profiling Technical Reference v2 February 2026 |
|---|

# 1. Purpose
game_benchmark.py is a deep validation and profiling tool for Cypha HRNA. It generates 150,000 synthetic game theory examples across three domains, trains Cypha on 80% of them, evaluates on the remaining 20%, and produces a detailed breakdown of every component's behaviour — including how often each deliberation method fires, where time is spent, and which classes are hardest to distinguish.

The benchmark is not testing whether Cypha can play chess, poker, or Go. It is testing whether Cypha's internal architecture is operating correctly when confronted with high-complexity, multi-class classification problems with genuine decision-boundary ambiguity. The three game domains were chosen because they provide exactly the right difficulty profile: clear class structure, known-good feature engineering, and natural class pairs that are genuinely hard to separate.

The primary verification targets are the deliberation pipeline components. Vanilla classification with high hippo hit rates is uninteresting — it means the training data is too easy and Cypha never needs to think hard. The benchmark is explicitly designed to force Cypha into deliberation for 35-55% of test examples, which is the regime where the full GRIA cascade, Rocchio, PNQ, MCTS, and DMN all activate and can be profiled.

| Property | Value |
|---|---|
| Domains | chess_evaluation (9 classes), poker_decision (8 classes), go_strategy (10 classes) |
| Examples per domain | 50,000 (600 in --quick mode) |
| Total examples | 150,000 |
| Train / test split | 80% train, 20% test |
| Training epochs | 1 (hardcoded — single-pass by design) |
| Boundary examples | 25% of each domain generated at class boundaries |
| Feature tokens per example | ~60-70 tokens |
| Feature dimension | 512 |
| Resonance dimension | 256 |
| Output | Per-domain accuracy, per-class breakdown, full component profile, JSON report |

# 2. Setup and Usage
## 2.1  Requirements
numpy is the only external dependency. No chess engine, no poker library, no Go engine — all three game AIs are embedded pure-Python implementations inside the file itself.

| pip install numpy  # Cypha.py must be in the same directory python game_benchmark.py |
|---|

## 2.2  Command-line flags
| Flag | Effect |
|---|---|
| (no flags) | Full benchmark: 50,000 examples per domain, all three domains |
| --quick | Smoke-test mode: 600 examples per domain, runs in seconds |
| --domain chess_evaluation | Run only the chess domain |
| --domain poker_decision | Run only the poker domain |
| --domain go_strategy | Run only the Go domain |
| --verbose | Print deliberation trace for each inference (very verbose on 50k examples) |

| python game_benchmark.py --quick python game_benchmark.py --domain go_strategy python game_benchmark.py --domain chess_evaluation --verbose |
|---|

# 3. Synthetic Data Generation
Each domain has an embedded AI class (ChessAI, PokerAI, GoAI) that generates realistic feature strings for a given class label. These are not templates filled with random words — each feature is derived from class-specific parameter ranges that encode real game logic. A tactical_combo position looks different from a fortress_defense position because the material balance, mobility, king safety, and pawn structure parameters are drawn from ranges that reflect how those positions actually look in real chess games.

## 3.1  Boundary examples
25% of examples in each domain are generated as boundary examples. For a given class, there is a defined BOUNDARY_PAIRS map that specifies which other class it is most likely to be confused with. When generating a boundary example, the feature generator partially blends in the statistical signature of the paired class: it draws some numeric parameters from the boundary class ranges and borrows 1-3 vocabulary keywords from the boundary class's token set.

This is what forces deliberation to fire. A clean tactical_combo example is easy — it has a high eval score, complex middlegame phase, and characteristic tactical vocabulary. A boundary example between tactical_combo and piece_sacrifice has a negative material balance (borrowed from piece_sacrifice) while retaining the high mobility and eval signal of tactical_combo. Cypha's hippo fast-path will miss these because they do not match stored episodes cleanly, and the deliberation engine must resolve them.

## 3.2  Data generation pipeline
| for each class in domain.classes: per_cls = n // len(classes) # balanced class distribution boundary_class = BOUNDARY_PAIRS[cls] for i in range(per_cls): if random() < 0.25: features = ai.generate(cls, boundary_target=boundary_class) else: features = ai.generate(cls, boundary_target=None) pairs.append((features, cls))  shuffle(pairs) return pairs[:n] |
|---|

# 4. The Three Embedded Game AIs
## 4.1  ChessAI — 9-class chess position classifier
ChessAI generates feature strings that represent chess positions at different stages and character types. It uses piece-square tables (PSTs), pawn structure analysis, and king safety heuristics — the same features used by real chess engines — to produce statistically realistic position descriptions.

The 9 classes cover the full spectrum of chess position types that require different strategic thinking:

| Class | What it represents | Key distinguishing features |
|---|---|---|
| tactical_combo | Position where forcing moves or tactics exist | High eval, complex middlegame, tactical flags (pin, fork, back rank), high mobility |
| positional_squeeze | Quiet positional advantage through space/mobility | Moderate eval, high white mobility vs low black mobility, open files low |
| endgame_technique | Technical endgame with a clear winning method | Late game phase, low mobility, passed pawns, opposition/key square flags |
| pawn_storm | Aggressive kingside or queenside pawn advance | Opposite castling, open files near king, high pawn storm vocab |
| piece_sacrifice | Material deficit with compensation (attack/initiative) | Negative material balance (-400 to -50), high eval, material_deficit flag |
| fortress_defense | Defending a difficult position by building a fortress | Large material deficit, low mobility for both sides, endgame phase |
| zugzwang | Position where any move worsens the position | Endgame phase, very low mobility both sides, opposition flags |
| opening_theory | Early game following known theoretical lines | Low move number, opening/early middlegame phase, low eval variance |
| endgame_conversion | Converting a winning endgame advantage | Material surplus, late game, moderate to low mobility, key square control |

Feature tokens (~65 per example)
Each ChessAI example contains the following token types:

| opening_X | Opening system played (Sicilian, Ruy Lopez, QGD, KID, etc.) — 20 openings |
|---|---|
| move_N | Full move number (1-80 depending on game phase) |
| phase_X | Game phase (opening, early_middlegame, complex_middlegame, simplified_middlegame, pawn_endgame, rook_endgame, piece_endgame, minor_piece_endgame, queen_endgame) |
| mat_+N | Material balance in centipawns (pawn=100, knight=320, bishop=330, rook=500, queen=900) |
| eval_+N | Position evaluation in centipawns from engine-style assessment |
| king_safety_w_N / king_safety_b_N | King safety score for each side (attackers near king, 0-10) |
| pawn_shield_w_N / pawn_shield_b_N | Pawn shield integrity (0-3, pawns protecting the king) |
| open_file_near_king_w_N / ..._b_N | Open files near the king (enemy rooks can penetrate) |
| mobility_w_N / mobility_b_N | Legal move count for each side |
| open_files_N | Total open files on the board |
| passed_w_N / passed_b_N | Passed pawn count for each side |
| isolated_w_N / isolated_b_N | Isolated pawn count (pawns with no friendly pawns on adjacent files) |
| doubled_w_N / doubled_b_N | Doubled pawn count |
| pawn_islands_w_N / pawn_islands_b_N | Pawn island count (connected groups of pawns) |
| castled_w_X / castled_b_X | Castling status: ks (kingside), qs (queenside), no (not castled) |
| pawn structure token | One of 12 pawn structure types (IQP, hanging pawns, Carlsbad, etc.) |
| piece config token | One of 15 piece configuration types (bishops of same color, rooks on 7th, etc.) |
| piece_activity_X | Overall piece activity level: low, med, high |
| tempo_X | Who has the tempo: w, b, equal |
| depth_N | Search depth the position was assessed at (10-40) |
| tactical flags | Situational flags: pin_detected, fork_threat, back_rank_weak, decisive_advantage, material_up, opposition_active, opposite_castling, etc. |
| class keywords | 8-12 vocabulary tokens specific to the class (e.g. initiative_compensation, king_attack_compensation for piece_sacrifice) |

Boundary pairs
| Class | Boundary partner | Why they are hard to separate |
|---|---|---|
| tactical_combo | piece_sacrifice | Both have high eval and complex middlegame; sacrifice positions often arise from tactics |
| piece_sacrifice | tactical_combo | Material deficit vs balanced with similar eval signals |
| fortress_defense | zugzwang | Both have low mobility and material deficit; differ in whether moves improve or worsen things |
| zugzwang | fortress_defense | Very low mobility for both sides — the key difference is whether the position is static or dynamic |
| pawn_storm | positional_squeeze | Open files and pawn advances appear in both; differ in king safety context |
| endgame_technique | endgame_conversion | Both are winning endgames; technique requires specific method, conversion is more straightforward |
| endgame_conversion | endgame_technique | Converting material advantage — boundary is how clear-cut the win is |
| positional_squeeze | opening_theory | Both can have similar material balance and low eval; differ in move number and phase |
| opening_theory | positional_squeeze | Early middlegame positions where theory ends and positional play begins |

## 4.2  PokerAI — 8-class poker decision classifier
PokerAI generates feature strings representing Texas Hold'em decision scenarios. It uses real GTO (Game-Theory Optimal) concepts: minimum defence frequency (MDF), fold equity, nut advantage, implied odds, and stack-to-pot ratio. Equity is approximated using board-texture-aware Monte Carlo with Gaussian noise on top of a class-specific equity hint, producing realistic variance without the computational cost of full card-level simulation.

| Class | Decision type | Typical equity range | Key context |
|---|---|---|---|
| value_bet | Bet for value with a strong hand | 62-95% | Top pair+, sets, strong draws. Build pot, commit stack. |
| bluff | Bet with air/weak hand using fold equity | 4-22% | Nothing hands, backdoor draws. Credible line, polarized range. |
| check_call | Check or call — pot control with medium hand | 32-62% | Middle pairs, draws, showdown value hands. |
| fold | Fold with no equity or reverse implied odds | 3-22% | Dominated hands, drawing dead, pot committed math fails. |
| pot_control | Bet small or check to keep pot manageable | 48-72% | Medium-strength made hands that can't stand a raise. |
| semi_bluff | Bet drawing hand — equity + fold equity combined | 24-50% | Flush draws, straight draws, combo draws. Aggression builds. |
| check_raise | Check then raise — trap strong hands or punish c-bets | 35-80% | Nut draws, sets, top pair on wet boards. Polar or semi-bluff XR. |
| donk_bet | Lead into the pre-flop aggressor — non-standard line | 28-72% | Range blocking, board texture changes, protection leads. |

Feature tokens (~65 per example)
| street_X | flop, turn, or river |
|---|---|
| position_X | btn, co, hj, mp, ep, bb, sb — table position |
| equity_N | Hero equity percentage (Monte Carlo approximation) |
| eq_bucket_X | Equity bucket: strong (70%+), ahead (55-70%), marginal (40-55%), behind (25-40%), weak (<25%) |
| pot_odds_N | Pot odds percentage for a standard half-pot bet |
| spr_N | Stack-to-pot ratio (effective stack / pot) |
| mdf_N | Minimum defence frequency — how often hero must call to prevent a pure bluff being profitable |
| fold_equity_N | Estimated percentage of time villain folds to a bet |
| nut_advantage_X | Whether hero has more nut combinations than villain (nut_adv, no_nut_adv) |
| implied_odds_X | Implied odds assessment (good, medium, poor) |
| board_texture_X | Board type from 10 options: dry_rainbow, wet_twoflush, paired_dry, monotone_flush, connected_rundown, etc. |
| hand_X | Hand description: top_pair_top_kicker, set, flush_draw, nothing_airball, etc. (20 options) |
| opp_type_X | Opponent type: passive_fish, aggressive_reg, tight_nit, loose_agg, tricky_pro, etc. |
| pot_N | Current pot size |
| bet_N | Bet size being considered |
| stack_eff_N | Effective stack size |
| bd_fd / bd_sd | Backdoor flush draw or straight draw present |
| nut_blocker / nut_blocker_pair | Hero holds a card that blocks the nut hand |
| range_advantage / disadvantage | Whether hero's range has more strong hands than villain on this board |
| class keywords | 8-10 GTO vocabulary tokens specific to the decision (e.g. thin_value, clear_value, overbet_value for value_bet) |

Boundary pairs
| Class | Boundary partner | Why they are hard to separate |
|---|---|---|
| value_bet | pot_control | Both involve betting with a made hand; the line between thin value and pot control is equity threshold |
| bluff | semi_bluff | Both involve betting weak holdings; semi-bluffs have equity, pure bluffs do not |
| check_call | fold | Both involve not betting; differ in whether the hand has any equity or reverse implied odds |
| pot_control | value_bet | Medium-strong hands that could be either depending on board runout and SPR |
| semi_bluff | check_call | Drawing hands can be called or semi-bluffed depending on fold equity and position |
| check_raise | check_call | Both start with a check; differ in whether the hand is strong enough to trap |
| donk_bet | check_call | Both involve not following standard lines; donk has a specific strategic purpose |
| fold | check_call | The critical decision: marginal equity hands that may or may not justify continuing |

## 4.3  GoAI — 10-class Go strategy classifier
GoAI generates feature strings representing Go board situations on a 9x9 grid. It uses a real Go board (numpy int8 array), BFS liberty counting, flood-fill territory estimation, and a vectorised Manhattan-distance influence map. The feature strings describe the strategic character of the current position — what type of Go problem is most pressing — rather than individual move decisions.

Go is the hardest of the three domains. The classes overlap naturally: a life-and-death situation can involve ko; a semeai (capturing race) is related to life-and-death; influence and reduction are strategic complements. The benchmark reflects this difficulty — Go is expected to have the lowest hippo hit rate and highest deliberation rate of the three domains.

| Class | Strategic situation | Key indicators |
|---|---|---|
| territory_lead | Comfortable territory advantage, converting the win | High territory differential, late game phase (move 100+), sealed territory |
| fighting | Active multi-stone battle in progress | Balanced captures, moderate territory, atari chains, complex fighting vocabulary |
| life_death | A group must make two eyes or die | Critical min_liberties (1-3), life-death vocabulary (nakade, vital point, false eye) |
| ko | Ko fight in progress — recapture sequences | High ko_b and ko_w counts (3-8), ko-specific vocabulary, ko master, superko |
| endgame | Dame filling, small endgame moves remain | Very late phase (move 180+), high stone count, counted territory, yose vocabulary |
| influence | Building a large moyo (sphere of influence) | High influence scores, low territory, early-mid game, moyo vocabulary |
| reduction | Approaching or reducing a large moyo | Moderate stone count, reduction sequence vocabulary, probe moves, shoulder hits |
| invasion | Invading deep into enemy territory | Low min_liberties for invading group, escape vocabulary, two-stage invasion |
| opening | Fuseki — establishing framework in opening | Low move count (<15), star points, approach moves, shimari formations |
| semeai | Capturing race between two groups | Both groups in atari, liberty counting vocabulary, outside vs inside liberties |

Feature tokens (~60-70 per example)
| move_N | Move number in the game (1-260) |
|---|---|
| phase_X | Game phase: opening (<30 moves), early_middle (30-80), middle (80-150), late (150-200), endgame (200+) |
| b_terr_N / w_terr_N | Territory estimate for Black and White via flood-fill |
| b_capt_N / w_capt_N | Cumulative captures for each colour |
| ko_b_N / ko_w_N | Number of ko threats held by each side |
| lib_min_N | Minimum liberty count across all groups on the board (1 = atari) |
| lib_avg_N | Average liberty count across all groups |
| dens_N | Board density (stones / total intersections as percentage) |
| moyo_N | Estimated moyo size (unclaimed influence territory) |
| seki_N | Whether a seki (mutual life without eyes) is present |
| b_groups_N / w_groups_N | Number of separate connected groups for each colour |
| b_stones_N / w_stones_N | Total stones on the board for each colour |
| b_atari_N / w_atari_N | Number of groups in atari (1 liberty) for each colour |
| b_max_group_N / w_max_group_N | Size of the largest group for each colour |
| b_eyes_N / w_eyes_N | Estimated eye count for each colour |
| corner_b_N / corner_w_N | Stones in corner 3x3 regions for each colour |
| edge_b_N / edge_w_N | Stones on edge rows/columns for each colour |
| center_b_N / center_w_N | Stones in the center 5x5 region for each colour |
| b_influence_N / w_influence_N | Manhattan-distance decay influence score over empty intersections |
| ko_active_X | Whether a ko is currently active (ko_active or ko_inactive) |
| cutting_points_N | Number of cutting points in the position |
| ladder_X | Whether a ladder is present (ladder_possible or ladder_clear) |
| net_X | Whether a net (geta) is present |
| sente_X | Who holds sente (forcing move priority): sente_b, sente_w, sente_equal |
| class keywords | 7-9 Go vocabulary tokens specific to the strategic situation |

Boundary pairs
| Class | Boundary partner | Why they are hard to separate |
|---|---|---|
| life_death | ko | Both involve critical groups with few liberties; ko fights often arise from life-and-death |
| ko | life_death | Ko recapture sequences often occur in bent-four-in-corner life-death situations |
| territory_lead | endgame | Late-game territory lead transitions smoothly into endgame; both have high territory counts |
| endgame | territory_lead | Endgame dame filling with comfortable lead looks similar to territory consolidation |
| fighting | invasion | Invasions often turn into fights; both have moderate territory and active groups |
| invasion | fighting | A fight that started as an invasion may now be a general battle |
| influence | reduction | These are strategic complements — one side builds moyo, the other reduces it |
| reduction | influence | Reducing a moyo requires entering the influence zone — the positions overlap |
| opening | influence | Early influence-building looks like fuseki; both have low move counts and high moyo |
| semeai | life_death | Capturing races often involve groups that are also fighting for life |

# 5. The Profiler
The benchmark includes a zero-overhead profiler that wraps every significant Cypha component via monkey-patching. No changes are made to Cypha.py — the wrapping is applied at runtime using Python's functools.wraps and direct attribute replacement. Every wrapped function records its call count, total time, and up to 8,000 individual timing samples for percentile calculation.

## 5.1  Instrumented components
| Component | Measurement | Path |
|---|---|---|
| trn.train_step | Full training step per example | Train path |
| trn.forward_train | Forward pass during training | Train path |
| trn.memory_store | AnchorMemory store operation | Train path |
| trn.hippo_store_trn | HippoCypha episodic store | Train path |
| trn.dmn_run | DMN consolidation loop | Train path |
| inf.infer | Full inference call | Inference path |
| inf.encode_features | OmegaEncoder feature extraction | Inference path |
| inf.forward_infer | Forward pass during inference | Inference path |
| inf.hippo_fastpath | HippoCypha fast-path lookup | Inference path |
| inf.adapter_lookup | AnchorMemoryAdapter MP-filtered lookup | Inference path |
| inf.workspace_compete | GlobalWorkspace ignition competition | Inference path |
| thought.note_uncertainty | ThoughtProcessor uncertainty tracking | ThoughtProcessor |
| thought.cascade | ThoughtProcessor hypothesis generation | ThoughtProcessor |
| thought.multi_scale | ThoughtProcessor multi-scale field blend | ThoughtProcessor |
| thought.self_generate | ThoughtProcessor trend detection | ThoughtProcessor |
| thought.resonant_chain | ThoughtProcessor coherence scoring | ThoughtProcessor |
| mem.memory_lookup | Direct AnchorMemory lookup | Memory |
| dlib.deliberate_iter | Iterative deliberation (legacy path) | Deliberation |
| pnq.pnq_lookup | PNQ perturbation-noise-query | Deliberation |
| mcts.mcts_search | MCTS tree search | Deliberation |
| gria.gria_cascade | Full GRIA 3-stage cascade | Deliberation |

## 5.2  Counters
| Counter | What it counts |
|---|---|
| hippo_hit | Inferences resolved by the hippo fast-path (immediate return) |
| hippo_miss | Inferences that required deliberation (hippo did not fire) |
| hippo_hit_rate | Derived: hippo_hit / (hippo_hit + hippo_miss) * 100% |
| gnw_fired | GlobalWorkspace ignition successes after hippo miss |
| gnw_miss | GlobalWorkspace ignition failures (forced deliberation) |
| gnw_fire_rate | Derived: gnw_fired / (gnw_fired + gnw_miss) * 100% |
| path_rocchio | Inferences where Rocchio contributed to deliberation |
| path_mcts | Inferences where MCTS ran |
| path_pnq | Inferences where PNQ ran (approximated by round count >= 2) |
| dmn_calls | Times the DMN consolidation loop was triggered |

## 5.3  Profile report format
After each domain and at the end of the full run, a profile table is printed with columns: component name, call count, mean latency (microseconds), p50, p95, p99, and total time (milliseconds). The report is grouped into five sections: TRAIN PATH, INFER PATH, THOUGHT PROCESSOR, MEMORY, and DELIBERATION.

| ╔══════════════════════════════════════════════════════════════════════════════╗ ║ CYPHA PROFILE — POKER_DECISION (train+infer) ║ ╠══════════════════════════════════════════════════════════════════════════════╣ ║ ─── INFER PATH ║ ║ component calls mean_us p50 p95 p99 total_ms║ ║ ───────────────────────────────────────────────────────────────────────── ║ ║ encode_features 10,000 1,166 1,180 1,200 1,245 11,660 ║ ║ hippo_fastpath 10,000 1,352 1,350 1,479 1,501 13,520 ║ ║ infer 6,780 4,012 3,990 4,323 4,801 27,201 ║ ║ ─── COUNTERS ║ ║ hippo_hit 3,220 ║ ║ hippo_miss 6,780 ║ ║ hippo_hit_rate 32.2% ║ ╚══════════════════════════════════════════════════════════════════════════════╝ |
|---|

# 6. Domain Profile Block
Before training begins on each domain, the benchmark prints a domain profile block showing the statistical properties of the generated dataset. This is important for verifying that the data generator produced a balanced, diverse dataset with the right token density.

| ┌────────────────────────────────────────────────────────────────────┐ │ DOMAIN PROFILE · CHESS_EVALUATION │ ├────────────────────────────────────────────────────────────────────┤ │ Total examples 50,000 │ │ Classes 9 │ │ Gen time 2.14s │ │ Examples/sec 23,364 │ ├────────────────────────────────────────────────────────────────────┤ │ Class count pct avg_tok │ │ ───────────────────────────────────────────────────────── │ │ tactical_combo 5,556 11.1% 65.3 │ │ positional_squeeze 5,556 11.1% 64.8 │ │ endgame_technique 5,556 11.1% 66.1 │ │ ... │ ├────────────────────────────────────────────────────────────────────┤ │ Token stats min=58 mean=65.2 max=74 │ │ Vocabulary size 4,821 │ │ Top-5 tokens material_balanced(12441) slight_advantage(11203) │ └────────────────────────────────────────────────────────────────────┘ |
|---|

The vocabulary size shows how many unique tokens the domain uses. A richer vocabulary gives Cypha more signal to work with. The top-5 tokens reveal which features dominate the corpus — high-frequency tokens like material_balanced appear across many classes and contribute less discriminative power than rare class-specific tokens.

# 7. Training and Evaluation Pipeline
Data is written to a temporary file in the format input|||label with one example per line. Cypha reads this file using the offset-indexed training path (train_file_stateful_offsets), which builds a byte-offset index and reads examples by seeking directly to each offset. This avoids loading the entire dataset into RAM.

| # Training — single epoch, offset-indexed file read cypha.train_file_stateful_offsets(tmp_file, train_offsets, domain_name, epochs=1, verbose=True)  # Evaluation — give_feedback called on every test example for offset in test_offsets: input_text, expected = read_at_offset(file, offset) predicted, confidence = cypha.infer(input_text) if predicted == expected: correct += 1 # give_feedback updates reflexion memory, confusion graph, # cerebellum output model, and Platt calibrator cypha.give_feedback(input_text, predicted, expected, top_margin, history=None) |
|---|

| give_feedback | During evaluation, give_feedback is called with the ground truth label after every inference. This is deliberate — it activates the reflexion failure memory (so the same mistake is less likely on the next similar input), updates the ConfusionGraph with real confusion data, and calibrates the Platt calibrator with real correct/wrong signal. The evaluation set is not truly held-out in the machine learning sense — it is a live learning phase. |
|---|---|

## 7.1  Kappa(D) — complexity metric
For each test example, the benchmark computes kappa(D) — the excess kurtosis of the byte-level differential of the input string. This measures the statistical complexity of the feature string's byte encoding. High kappa(D) values indicate heavy-tailed distributions in the byte stream, which corresponds to feature strings with unusual or rare token combinations. The distribution of kappa(D) across test examples gives a measure of how complex the overall test set is.

Kappa(D) is reported as mean, standard deviation, p5 and p95. A higher mean indicates a more complex test set. This is primarily a diagnostic metric to verify that the synthetic generator is producing statistically diverse inputs rather than repetitive templates.

# 8. Understanding the Output
## 8.1  Per-class accuracy bar chart
| Per-class accuracy: Class Bar Correct Acc ────────────────────────────────────────────────────────── tactical_combo ██████████ 1,089/1,112 ( 97.9%) positional_squeeze █████████░ 1,051/1,111 ( 94.6%) piece_sacrifice ████████░░ 1,002/1,112 ( 90.1%) fortress_defense ████████░░ 986/1,111 ( 88.7%) zugzwang ████████░░ 971/1,111 ( 87.4%) |
|---|

Classes with lower accuracy are the interesting ones — they represent the hardest decision boundaries in the domain. Piece_sacrifice and fortress_defense consistently score lower than tactical_combo because their feature signatures overlap with other classes. This is expected and reflects the quality of the boundary generation.

## 8.2  Sample errors
| Sample errors (first 10): expected=fortress_defense got=zugzwang conf=0.341 input: phase_pawn_endgame mat_-280 eval_-40 king_safety_w_6 mobility_w_4... expected=piece_sacrifice got=tactical_combo conf=0.412 input: phase_complex_middlegame mat_-180 eval_+240 king_safety_w_3... |
|---|

Error analysis reveals the natural confusion structure of the domain. fortress_defense confused with zugzwang is expected — both have low mobility endgame positions. The features that distinguish them (static vs. worsening position) are subtle and may require deliberation to resolve correctly. A low confidence score (0.341) on an error indicates Cypha was uncertain — a higher confidence error indicates the system was wrongly confident.

## 8.3  JSON report
After all domains complete, results are saved to game_benchmark_report_v2.json in the same directory. The JSON contains the full per-domain accuracy, per-class accuracy, training and evaluation times, number of samples, and all errors. This file can be used to track accuracy changes across code iterations.

| { "total_wall_time_s": 412.7, "epochs": 1, "n_per_domain": 50000, "domains": { "chess_evaluation": { "accuracy": 97.3, "n_train": 40000, "n_test": 10000, "per_class_acc": {"tactical_combo": 97.9, ...}, "errors": [...] }, ... } } |
|---|

# 9. Interpreting Results — What to Look For

| Overall accuracy > 95% | Cypha has successfully learned the class manifolds for the domain. The feature engineering and class-specific parameter ranges are well-separated in the 512-dimensional anchor space. |
|---|---|
| Hippo hit rate 30-60% | Healthy deliberation regime. Too high (>80%) means the test data is too similar to training data — reduce the boundary example blending ratio or increase class count. Too low (<20%) means Cypha is not forming stable prototypes. |
| Rocchio firing on ~30% of deliberation cases | Rocchio centroid push is working. If this is 0%, the deliberation pipeline is not routing through the cascade correctly. |
| PNQ and MCTS firing | PNQ fires when round count >= 2. MCTS fires for the hardest boundary cases. These should both show non-zero counts after a full 50k run. |
| DMN calls > 0 | The Default Mode Network consolidation loop is triggering. It fires after a configurable number of training steps. |
| ThoughtProcessor calls = N_train | All five ThoughtProcessor methods should be called on every training step. Zero counts indicate a wiring break. |
| Low accuracy on boundary classes | Expected. fortress_defense, zugzwang, piece_sacrifice (chess); fold, check_call (poker); life_death, ko, semeai (Go) are the natural hard boundaries and will always score lower. |
| Consistent p50 vs p95 for hippo_fastpath | If p95 is much higher than p50 (e.g. 1,200 vs 8,000 us), the memory store has grown unevenly. Consolidation may not be keeping the store compact. |

# 10. Files and Cleanup
Each domain creates a temporary directory with a training data file and checkpoint directory. These are cleaned up automatically at the end of each domain run via shutil.rmtree(). If the benchmark is interrupted (KeyboardInterrupt is caught and handled cleanly), the temporary directory is also removed.

The only persistent output is game_benchmark_report_v2.json in the script directory. This file is overwritten on each run.

| # If interrupted or crashed, clean up manually rm -rf /tmp/cypha_game_* |
|---|

game_benchmark.py   Cypha HRNA   February 2026   All rights reserved
