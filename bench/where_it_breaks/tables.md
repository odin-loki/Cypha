## Table 1. Every word, by text (30 prompts; first 600 bytes)

| | human | T8 | MR |
|---|---|---|---|
| words | 2999 | 3748 | 3887 |
| mean bits a word under K5 + overlay (<unk> left out) | 7.50 | 3.38 | 4.10 |
| <unk> words (outside the 32,000-word vocabulary) | 0.077 | 0.012 | 0.012 |
| answered by order 5 (the word seen after the 4 words before) | 0.116 | 0.508 | 0.374 |
| answered by order 4 | 0.198 | 0.243 | 0.278 |
| answered by order 3 | 0.305 | 0.184 | 0.252 |
| answered by order 2 (only the word before) | 0.242 | 0.049 | 0.080 |
| from the class floor (no context held the word) | 0.061 | 0.004 | 0.004 |
| an order-5 context existed (4 words back matched) | 0.361 | 0.759 | 0.661 |
| the overlay's document held the word in context | 0.190 | 0.445 | 0.368 |
| overlay's gain, bits a word | +0.04 | +0.87 | +0.73 |
| rank 1 (K5's most probable word) | 0.234 | 0.518 | 0.404 |
| rank 1-10 | 0.540 | 0.867 | 0.793 |
| P < 0.01 | 0.435 | 0.123 | 0.190 |
| inside T8's nucleus (T 0.8, nucleus 0.8, after the bans) | 0.481 | 0.916 | 0.791 |
| words in the nucleus (median) | 11 | 6 | 7 |
| learner's bits a word (primed, frozen; copy off) | 10.47 | 7.89 | 7.55 |

## Table 2. The first two words of a sentence against the rest

| | human, first 2 | human, rest | T8, first 2 | T8, rest |
|---|---|---|---|---|
| words | 193 | 2806 | 472 | 3276 |
| bits a word | 6.66 | 6.94 | 4.01 | 3.25 |
| answered by order 5 | 0.119 | 0.115 | 0.453 | 0.516 |
| answered by order 2 or the floor | 0.161 | 0.313 | 0.044 | 0.055 |
| nucleus size (median) | 13 | 11 | 11 | 5 |

## Table 3. Human words K5 finds very unlikely (P < 2^-15, or <unk>)

562 of 2999 human words (0.187).

| kind | words | share of human words |
|---|---|---|
| <unk> (vocabulary hole) | 231 | 0.077 |
| class floor (no context held it) | 148 | 0.049 |
| order 2 only | 120 | 0.040 |
| order 3+ held it, still below 2^-15 | 63 | 0.021 |

Of these, capitalised: 231; numbers: 6.

Human words outside T8's nucleus: 0.519 (1557 of 2999).

## Table 4. T8's broken spots, marked by reading (90 spots in 30 texts), against all T8 words

| | broken spots | all T8 words |
|---|---|---|
| answered by order 5 | 0.233 | 0.508 |
| answered by order 4 | 0.300 | 0.243 |
| answered by order 3 | 0.311 | 0.184 |
| answered by order 2 | 0.111 | 0.049 |
| class floor | 0.022 | 0.004 |
| context matched 4 words back (order-5 context existed) | 0.700 | 0.759 |
| context matched 2 or fewer words back | 0.078 | 0.052 |
| overlay held the word in context | 0.378 | 0.445 |
| rank 1 | 0.300 | 0.518 |
| rank 2-10 | 0.444 | 0.349 |
| rank above 10 (the tail of the nucleus) | 0.256 | 0.133 |
| P >= 0.1 | 0.333 | 0.609 |
| P < 0.01 | 0.278 | 0.123 |
| mean bits | 5.23 | 3.37 |
| nucleus size (median) | 12 | 6 |

Sampling into the tail (rank above 10 or P < 0.02): **34 of 90**. A high-probability wrong word (rank 1-3 and P >= 0.1): **30 of 90**.

Kinds of break, by reading: sense 46; grammar 44.

### Table 4b. The broken spots by the sentence's source, and by kind

| group | spots | T8 words in group | mean bits at spot | answered by order 5 | backed off (order 4 or lower) | tail (rank > 10 or P < 0.02) | high-probability wrong (rank 1-3, P >= 0.1) | <unk> |
|---|---|---|---|---|---|---|---|---|
| plain gate's draft (k5) | 54 | 2229 | 4.01 | 0.24 | 0.76 | 0.28 | 0.39 | 0.00 |
| discourse free sentence | 9 | 746 | 4.44 | 0.11 | 0.89 | 0.33 | 0.33 | 0.00 |
| planner fact sentence | 27 | 773 | 7.94 | 0.26 | 0.67 | 0.59 | 0.22 | 0.07 |
| kind: grammar | 44 | 3748 | 4.26 | 0.25 | 0.75 | 0.30 | 0.43 | 0.00 |
| kind: sense | 46 | 3748 | 6.15 | 0.22 | 0.74 | 0.46 | 0.24 | 0.04 |

T8 words by source: k5 2229, fact 773, disc 746.

- K5's own sentences (k5 + disc): 2975 words; mean bits 2.89; outside the nucleus 0.006; rank above 10 0.092; <unk> 0.002.
- planner fact sentences: 773 words; mean bits 5.09; outside the nucleus 0.384; rank above 10 0.291; <unk> 0.048.

The byte learner at the broken spots: log2 P -9.89 (all T8 words -7.89); context ratio +3.57 bits (all +3.92); below the median ratio at 0.57 of the spots (chance 0.50).

| item | source | before | broken word | P | rank | order answered | context depth | overlay | K5's top word | kind |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | fact | …, when a large enough | **suspension** | 0.003 | 46 | 4 | 4 words | no | to (0.36) | sense |
| 0 | fact | …and successful New Zealand and | **attack** | 0.000 | 773 | 2 | 3 words | no | the (0.08) | sense |
| 0 | fact | …Auckland is a town located | **throughout** | 0.001 | 32 | 2 | 4 words | no | in (0.61) | sense |
| 1 | k5 | …uses the nitric acid ( | **2** | 0.528 | 1 | 5 | 4 words | yes | 2 (0.53) | sense |
| 1 | k5 | …or low order explosive , | **was** | 0.527 | 1 | 5 | 4 words | yes | was (0.53) | grammar |
| 1 | k5 | …added weight of the guns | **.** | 0.185 | 1 | 4 | 4 words | no | . (0.19) | grammar |
| 2 | k5 | …to the ground and the | **sets** | 0.073 | 1 | 3 | 4 words | yes | sets (0.07) | sense |
| 2 | k5 | …in a real sense , | **the** | 0.208 | 1 | 5 | 4 words | yes | the (0.21) | grammar |
| 2 | k5 | …taking an opponent to the | **formation** | 0.005 | 6 | 3 | 4 words | yes | ground (0.71) | sense |
| 3 | fact | …His books include the Food | **Food** | 0.000 | 463 | 1 | 2 words | no | and (0.43) | grammar |
| 3 | disc | …and director Food writing . | **)** | 0.006 | 25 | 3 | 2 words | no | The (0.16) | grammar |
| 3 | disc | …aqueous solution of hydrogen and | **helium** | 0.478 | 1 | 4 | 3 words | no | helium (0.48) | sense |
| 4 | k5 | …controversial , some in the | **New** | 0.141 | 1 | 4 | 4 words | yes | New (0.14) | sense |
| 4 | k5 | …Some religious groups ( e | **.** | 0.982 | 1 | 5 | 4 words | yes | . (0.98) | grammar |
| 4 | k5 | …than 100 % because of | **its** | 0.082 | 4 | 3 | 4 words | no | the (0.23) | sense |
| 5 | fact | …Party System and operated with | **Matt** | 0.000 | 1163 | 2 | 3 words | no | the (0.80) | sense |
| 5 | fact | …, 1765 ) is a | **US** | 0.001 | 114 | 4 | 3 words | no | novel (0.20) | sense |
| 5 | disc | …She is grateful to the | **Great** | 0.026 | 6 | 3 | 3 words | yes | Lord (0.03) | sense |
| 7 | fact | …He was born in | **Chalmers** | 0.440 | 1 | 0 | 4 words | yes | the (0.14) | sense |
| 7 | fact | …of Technology in Gothenburg , | **New** | 0.020 | 8 | 4 | 4 words | no | and (0.07) | sense |
| 7 | fact | …musician . Gothenburg was the | **son** | 0.074 | 2 | 5 | 4 words | no | first (0.25) | sense |
| 9 | k5 | …with the widespread availability of | **oracles** | 0.142 | 1 | 3 | 4 words | yes | oracles (0.14) | sense |
| 9 | k5 | …of divination . The more | **the** | 0.027 | 4 | 4 | 4 words | no | simply (0.08) | grammar |
| 9 | k5 | …study of plant data . | **Used** | 0.467 | 1 | 5 | 4 words | yes | Used (0.47) | grammar |
| 10 | k5 | …shaped like a pair of | **four** | 0.465 | 1 | 5 | 4 words | yes | four (0.47) | sense |
| 10 | k5 | …that for any of the | **movie** | 0.010 | 9 | 4 | 4 words | yes | other (0.03) | sense |
| 10 | k5 | …movie . Thus the " | **(** | 0.009 | 8 | 3 | 4 words | yes | Aegean (0.16) | grammar |
| 11 | fact | …Esperanto from the language of | **Adam** | 0.007 | 7 | 5 | 4 words | no | the (0.74) | sense |
| 11 | disc | …the period of constructed language | **.** | 0.045 | 4 | 3 | 3 words | yes | whose (0.39) | grammar |
| 11 | disc | …to ensure that the correct | **and** | 0.011 | 15 | 3 | 4 words | no | way (0.05) | grammar |
| 12 | k5 | …nephew , Leto Atreides II | **.** | 0.247 | 2 | 5 | 4 words | no | , (0.36) | grammar |
| 12 | k5 | …, the sister of the | **same** | 0.012 | 15 | 4 | 4 words | no | last (0.04) | grammar |
| 12 | k5 | …Paul ( born Paul Atreides | **and** | 0.114 | 2 | 3 | 2 words | yes | , (0.18) | grammar |
| 13 | k5 | …Messianic Jewish Alliance of the | **Democratic** | 0.235 | 1 | 4 | 3 words | no | Democratic (0.23) | sense |
| 13 | k5 | …. The organized Messianic Movement | **.** | 0.068 | 2 | 3 | 4 words | yes | consists (0.44) | grammar |
| 13 | k5 | …" for holding on the | **continent** | 0.001 | 81 | 3 | 3 words | no | part (0.17) | sense |
| 15 | k5 | …The topological dimension of any | **size** | 0.010 | 11 | 3 | 4 words | no | curve (0.13) | sense |
| 15 | k5 | …a point is the slope | **is** | 0.074 | 2 | 3 | 4 words | yes | of (0.82) | grammar |
| 15 | k5 | …. Therefore , the slope | **,** | 0.051 | 3 | 4 | 3 words | yes | is (0.45) | grammar |
| 16 | fact | …town and municipality in the | **county** | 0.877 | 1 | 5 | 4 words | no | county (0.88) | sense |
| 16 | fact | …India . Divisions of Bangladesh | **is** | 0.211 | 2 | 4 | 3 words | no | , (0.22) | grammar |
| 16 | fact | …surrounded by the city of | **Los** | 0.061 | 3 | 5 | 4 words | no | New (0.07) | sense |
| 17 | k5 | …the now - defunct Canadian | **ever** | 0.011 | 6 | 2 | 4 words | yes | Baseball (0.67) | grammar |
| 17 | k5 | …New York City , which | **are** | 0.030 | 5 | 4 | 4 words | no | was (0.14) | grammar |
| 17 | k5 | …in the history of the | **best** | 0.037 | 2 | 3 | 4 words | yes | United (0.08) | grammar |
| 18 | k5 | …used for storing picture data | **.** | 0.625 | 1 | 5 | 4 words | yes | . (0.63) | grammar |
| 18 | k5 | …format supports horizontal and vertical | **integration** | 0.021 | 4 | 4 | 4 words | no | lines (0.10) | sense |
| 18 | k5 | …integration . The last 32 | **being** | 0.278 | 1 | 3 | 2 words | yes | being (0.28) | grammar |
| 20 | fact | …The characters like the greek | **Litsa** | 0.118 | 2 | 0 | 4 words | yes | , (0.14) | sense |
| 20 | disc | …suggests that the monotheistic religions | **.** | 0.132 | 2 | 3 | 3 words | no | , (0.33) | grammar |
| 20 | disc | …by scholars , and a | **USA** | 0.081 | 1 | 5 | 4 words | no | USA (0.08) | sense |
| 21 | fact | …He is also a | **term** | 0.002 | 57 | 4 | 4 words | no | member (0.16) | sense |
| 21 | fact | …California Department of Defense began | **his** | 0.009 | 12 | 2 | 4 words | no | to (0.47) | grammar |
| 21 | fact | …the county of California , | **Hawaii** | 0.002 | 40 | 3 | 3 words | no | and (0.35) | sense |
| 22 | k5 | …collateral proceedings such as corporations | **from** | 0.381 | 1 | 4 | 3 words | yes | from (0.38) | grammar |
| 22 | k5 | …from the bench on the | **street** | 0.059 | 1 | 4 | 3 words | no | street (0.06) | sense |
| 22 | k5 | …next day , the judge | **.** | 0.036 | 4 | 4 | 3 words | no | may (0.38) | grammar |
| 23 | k5 | …She also frequently shaved off | **his** | 0.175 | 1 | 3 | 2 words | no | his (0.18) | sense |
| 23 | k5 | …number of innovations in the | **dust** | 0.074 | 2 | 3 | 4 words | yes | car (0.07) | sense |
| 23 | k5 | …teacher of Class B , | **then** | 0.049 | 2 | 4 | 3 words | no | and (0.11) | grammar |
| 24 | k5 | …source of revenue for the | **original** | 0.009 | 7 | 3 | 4 words | yes | government (0.07) | sense |
| 24 | k5 | …is unique , up to | **10** | 0.016 | 7 | 4 | 4 words | no | the (0.20) | sense |
| 24 | k5 | …series of " decomposed " | **into** | 0.518 | 1 | 4 | 3 words | yes | into (0.52) | grammar |
| 25 | fact | …For almost a | **hero** | 0.000 | 96 | 3 | 4 words | no | century (0.23) | grammar |
| 25 | fact | …and the AR - 15 | **Park** | 0.000 | 569 | 2 | 4 words | no | and (0.13) | sense |
| 25 | disc | …- 15 Park . A | **of** | 0.009 | 8 | 2 | 3 words | no | 2002 (0.19) | grammar |
| 26 | k5 | …became queen of Sweden . | **)** | 0.003 | 34 | 4 | 4 words | no | Maria (0.26) | grammar |
| 26 | k5 | …of the Franks , and | **this** | 0.004 | 33 | 3 | 4 words | no | his (0.07) | grammar |
| 26 | k5 | …The Pope Gregory XI ( | **1922** | 0.020 | 7 | 3 | 4 words | no | the (0.04) | sense |
| 28 | fact | …famous examples of this skepticism | **are** | 0.213 | 1 | 5 | 4 words | no | are (0.21) | grammar |
| 28 | fact | …lake is surrounded by the | **french** | 0.000 | 13142 | 2 | 4 words | no | d (0.30) | sense |
| 28 | fact | …Dublin . Guillaume Morel is | **located** | 0.003 | 26 | 2 | 2 words | no | a (0.15) | sense |
| 29 | fact | …- Kreis is a small | **village** | 0.140 | 2 | 5 | 4 words | no | town (0.24) | sense |
| 29 | fact | …a city built in the | **human** | 0.000 | 266 | 3 | 4 words | no | middle (0.05) | sense |
| 29 | disc | …nigsberg in East Prussia , | **the** | 0.257 | 1 | 4 | 4 words | yes | the (0.26) | grammar |
| 31 | k5 | …indigenous people of the central | **nervous** | 0.083 | 2 | 4 | 4 words | no | government (0.11) | sense |
| 31 | k5 | …several " stepping - stone | **to** | 0.498 | 1 | 4 | 4 words | no | to (0.50) | grammar |
| 31 | k5 | …On November 6 , 2005 | **.** | 0.229 | 2 | 5 | 4 words | no | , (0.29) | grammar |
| 32 | k5 | …is divided into two separate | **countries** | 0.019 | 6 | 4 | 4 words | no | entities (0.06) | sense |
| 32 | k5 | …Although the big investment , | **integrate** | 0.006 | 24 | 2 | 2 words | yes | and (0.13) | grammar |
| 32 | k5 | …. The area around the | **altar** | 0.036 | 4 | 5 | 4 words | no | world (0.08) | sense |
| 33 | k5 | …behaviour of the arrow keys | **.** | 0.505 | 1 | 5 | 4 words | yes | . (0.51) | grammar |
| 33 | k5 | …the original IBM PC keyboard | **.** | 0.662 | 1 | 5 | 4 words | yes | . (0.66) | grammar |
| 33 | k5 | …In the United States , | **which** | 0.025 | 5 | 5 | 4 words | no | the (0.15) | grammar |
| 35 | k5 | …the emperor Maximilian I , | **the** | 0.076 | 3 | 4 | 4 words | yes | Holy (0.22) | grammar |
| 35 | k5 | …language . South of the | **various** | 0.035 | 1 | 3 | 4 words | yes | various (0.03) | grammar |
| 35 | k5 | …as well . In 441 | **Court** | 0.017 | 8 | 1 | 3 words | no | , (0.21) | grammar |
| 36 | fact | …in the center of Europe | **in** | 0.039 | 5 | 3 | 4 words | no | . (0.32) | sense |
| 36 | fact | …Europe . Alps is a | **city** | 0.002 | 44 | 3 | 3 words | no | growing (0.15) | sense |
| 36 | fact | …. It stands north of | **Chicago** | 0.099 | 2 | 5 | 4 words | no | the (0.36) | sense |

## Table 5. The gate: which of its 16 drafts T8 kept

Sentences whose 16 drafts were reproduced exactly: 207 (plain gate 145, discourse free sentences 62); fact sentences or unmatched: 51. The plain gate's choice, recomputed, equals the kept draft in 145 of 145.

| | value |
|---|---|
| kept draft's fluency rank among 16 (1: most fluent), mean | 6.99 (uniform: 8.5) |
| kept draft is the most fluent | 0.193 |
| kept draft in the less fluent half (rank 9-16) | 0.386 |
| mean log2 P a word: kept / mean of 16 / most fluent | -2.88 / -3.12 / -2.03 |
| kept draft's z(prompt association) + z(topic), mean | +3.79 |

## Table 6. The most probable output against the sampled and the human text

| text | mean bits a word under K5 + overlay | words in 460+ bytes |
|---|---|---|
| greedy | 2.10 | 99 |
| beam8 | 1.44 | 95 |
| beam16 | 1.35 | 95 |
| beam8_ratio | 1.79 | 101 |
| human (sampled or written) | 7.50 | — |
| T8 (sampled or written) | 3.38 | — |
| MR (sampled or written) | 4.10 | — |

Repetition (share of word 3-grams that occur earlier in the same text), and the share of sentences that end:

| text | repeated 3-grams | sentences per 100 words |
|---|---|---|
| greedy | 0.031 | 6.3 |
| beam8 | 0.009 | 4.7 |
| beam16 | 0.008 | 4.5 |
| beam8_ratio | 0.033 | 10.8 |
| human | 0.007 | 4.2 |
| T8 | 0.016 | 6.3 |
| MR | 0.017 | 8.1 |

