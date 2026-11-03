## Results

### The run

- **Text.** 38383 pages read, 889 dropped (a 32-byte span of a dev continuation). Train 134,217,700 bytes (28413 pages), val 4,181,633 bytes (1473 pages).
- **Tokens.** Train 34,083,756 tokens, 3.938 bytes per token.
- **M5.** 11,543,552 parameters, batch 2048, 33284 steps, 2.4 h. Best val loss 4.3981 nats per token at step 33264 (of 25 checks); last 4.3985.
- **TR.** 13,891,584 parameters, batch 64, 4160 steps, 3.4 h. Best val loss 3.4964 nats per token at step 4152 (of 25 checks); last 3.4969.

### Bits per byte

| Arm | Val (bits per byte) | GEN-A dev continuations [95%] |
|---|---:|---|
| K5 | 1.6075 | 1.6175 [1.5923, 1.6422] |
| K8 | 1.6040 | 1.6164 [1.5909, 1.6411] |
| M5 | 1.7247 | 1.7035 [1.6750, 1.7308] |
| TR | 1.3440 | 1.3513 [1.3245, 1.3761] |

### SCRIBE v2 on GEN-A dev (250 prompts)

| Arm | Pooled [95%] | T 0.8 [95%] | T 1.0 [95%] | Surface S pooled [95%] | Kept | Mean bytes |
|---|---|---|---|---|---:|---:|
| K5 | 0.360 [0.341, 0.378] | 0.379 [0.352, 0.405] | 0.341 [0.312, 0.371] | 0.908 [0.903, 0.912] | 484 | 391 |
| K8 | 0.334 [0.316, 0.352] | 0.355 [0.334, 0.378] | 0.313 [0.285, 0.340] | 0.907 [0.903, 0.911] | 480 | 397 |
| M5 | 0.359 [0.340, 0.379] | 0.409 [0.382, 0.437] | 0.307 [0.280, 0.335] | 0.874 [0.868, 0.879] | 482 | 397 |
| TR | 0.830 [0.810, 0.849] | 0.912 [0.889, 0.934] | 0.747 [0.719, 0.776] | 0.873 [0.866, 0.881] | 496 | 396 |
| Q0 | 0.736 [0.705, 0.766] | 0.679 [0.629, 0.731] | 0.793 [0.754, 0.828] | 0.848 [0.834, 0.862] | 495 | 402 |

### SCRIBE's axes (mean over kept samples; lower is better)

| Arm | Rung | D | D_L | D_S | D_C | D_I |
|---|---|---:|---:|---:|---:|---:|
| K5 | pooled | 6.13 | 0.00 | 1.01 | 5.11 | 0.00 |
| K5 | T0.8 | 5.62 | 0.00 | 0.67 | 4.95 | 0.00 |
| K5 | T1.0 | 6.64 | 0.00 | 1.36 | 5.28 | 0.00 |
| K8 | pooled | 6.41 | 0.00 | 1.07 | 5.34 | 0.00 |
| K8 | T0.8 | 5.80 | 0.00 | 0.60 | 5.20 | 0.00 |
| K8 | T1.0 | 7.04 | 0.00 | 1.55 | 5.49 | 0.00 |
| M5 | pooled | 6.43 | 0.01 | 1.55 | 4.87 | 0.00 |
| M5 | T0.8 | 5.19 | 0.00 | 0.66 | 4.53 | 0.00 |
| M5 | T1.0 | 7.71 | 0.02 | 2.47 | 5.22 | 0.00 |
| TR | pooled | 1.26 | 0.18 | 0.11 | 0.96 | 0.01 |
| TR | T0.8 | 0.70 | 0.29 | 0.01 | 0.38 | 0.03 |
| TR | T1.0 | 1.82 | 0.07 | 0.21 | 1.54 | 0.00 |
| Q0 | pooled | 8.22 | 0.09 | 0.46 | 0.72 | 6.95 |
| Q0 | T0.8 | 11.95 | 0.13 | 0.41 | 0.57 | 10.83 |
| Q0 | T1.0 | 4.44 | 0.05 | 0.51 | 0.87 | 3.01 |

### Copying and words

| Arm | Copy share (32 B) pooled | T 0.8 | T 1.0 | Samples over the 12-word cap | Invented words |
|---|---:|---:|---:|---:|---:|
| K5 | 0.1823 | 0.2747 | 0.0898 | 0.042 | 0.0062 |
| K8 | 0.1819 | 0.2803 | 0.0835 | 0.042 | 0.0065 |
| M5 | 0.0446 | 0.0733 | 0.0158 | 0.020 | 0.0430 |
| TR | 0.0337 | 0.0517 | 0.0157 | 0.000 | 0.0370 |
| Q0 | 0.0555 | 0.0680 | 0.0431 | 0.002 | 0.0089 |

### The two differences (pooled dev SCRIBE, paired bootstrap over prompts, 2,000 draws)

| Difference | Point | 95% |
|---|---:|---|
| Δsim = M5 − K5 | -0.001 | [-0.028, +0.025] |
| Δstate = TR − M5 | +0.471 | [+0.444, +0.497] |
| K8 − K5 | -0.026 | [-0.054, +0.002] |

### Predictions

| # | Prediction | Result | Scored |
|---|---|---|---|
| N1 | Val bits per byte: M5 below K5 by ≥ 0.05; TR below M5 by ≥ 0.15 | K5 − M5 = -0.1172; M5 − TR = 0.3807 | **miss** |
| N2 | K8's SCRIBE within ±0.02 of K5's | K8 − K5 = -0.026 | **miss** |
| N3 | M5's SCRIBE ≥ K5's + 0.05, with D_S at least 15% lower | M5 − K5 = -0.001; D_S lower by -52.6% | **miss** |
| N4 | TR's SCRIBE ≥ M5's + 0.10, mostly from D_S and D_C | TR − M5 = +0.471; D_S + D_C give 104% of the fall in D | **hit** |
| N5 | TR's dev SCRIBE in [0.30, 0.60] | TR = 0.830 | **miss** |
| N6 | Δstate > Δsim | Δstate +0.471, Δsim -0.001 | **hit** |
| N7 | The neural arms copy less than the count arms (32-byte copy share) | K5 0.1823, K8 0.1819, M5 0.0446, TR 0.0337 | **hit** |

**What it settles (§13.3's rule).** The larger difference is Δstate (+0.471 [+0.444, +0.497]). Its interval excludes 0, so P2, state first, is ported first.

### Five samples per arm (T 0.8, the first five dev prompts)

**K5**

- * Motors New Zealand plant in Petone outside Wellington, until it closed in 1990.* →  It is common to use the same position with respect to the vernal equinox, the league, consisting of Thebes, and the school's hallucinations (the lesser amounts, the Catholic and Orthodox monastics (this was, however, a life of comfort to this material repeatedly tied to the policies of the individu
- * granules of powder and the confinement determine the burn rate of black powder.* →  The concept of "Bob" was not popular when it first came into use. The first in the 1930s. The main difference between the two sides of the track. 90%); and the ethnic Hungarian party to be called "Heidegger, a Republican Party over the city, and that he had the largest mercantile exchange in the wo
- * exhausting the opponent, executing a submission hold, or striking the opponent.* →  With the advent of modern name of the city of Brabant. By the late 18th century, the line "Who Wrote The Bible also records for the military. He also wrote some short stories, in a crushing defeat for the first time. / A riders will not be a complete in the last years of his life in the process. Some
- *de ("Food-Grade," or without added chemical stabilisers) for use as a mouthwash.* →  A common modules. The United States is methylphenidate), Adderall, his own bands through the club to the NAFTA, a secretary to the SLAC has an extensive system of cash flowing in the motor. They may also be offered for sale to the state of the United Nations and the Arab League and the Police Commi
- *dia Census Commission give a breakdown by various parameters including religion.* →  In the 1980s, the team's top rusher on the team with the best record in the league. The league championships. The tension between the two sides of the Anglican Church of Canada, where she lived with her. He was reelected in the mid-19th century, and the smaller, wicker and Lieutenant Commander. He 

**K8**

- * Motors New Zealand plant in Petone outside Wellington, until it closed in 1990.* →  It is interesting to note that the opening in the UK. The US Army, and the country has a "Republican) and a large number of positions, and amber, etc. / South) transported bonus for the NCAA Division III athletic and artistic side. / The Animals, such as the VCRs. The end result is that the businessmen
- * granules of powder and the confinement determine the burn rate of black powder.* →  The third time of creation. The constitutions and using the Internet to give a close approximation of the Earth's surface. The giant (June 23, his Roman and has been a strong advocate for a variety of reasons. Various laws, moral rules and impairment is the large, Turkmenistan, Uzbekistan, and she 
- * exhausting the opponent, executing a submission hold, or striking the opponent.* →  With the advent of modern microprocessor (and on their upper window manager, and many other countries around the world. In the course of the trial in the 1860s. The Toronto's 2000 novelists; but they are less severe. All of the series to the Boston Celtics, and the two became close allies, using th
- *de ("Food-Grade," or without added chemical stabilisers) for use as a mouthwash.* →  A common modules. The result is that in the United States, as well as the Triassic.--all intimately connected with the mainland by Alexander the Great, and each of the three times in the same or similar to the Imperial House. Wade's best to use the following their 56 year maximum copyright infringe
- *dia Census Commission give a breakdown by various parameters including religion.* →  In the 1870s, and the p-adic number-one on the pop charts in 1961. The Soviets lifted their blockade, and he had to be called and told that because of its main characters, or to the jackpot winner in the UK. This title is currently carried by the ancients was revived the Somerset Levels are notorie

**M5**

- * Motors New Zealand plant in Petone outside Wellington, until it closed in 1990.* →  / The Prime Minister of the United States Department of Justice and the Unitedist states. The Tagoren Sorry and Humans. / The warriors found them as The Hearst the Survivallet became the best-known eventual and Dance was one of the two halves of the project, and was subsequently joined the Internationa
- * granules of powder and the confinement determine the burn rate of black powder.* →  The League of Nations of the United States. When the city is also called the Jurassic Park in the north, and the south of the sun, the result is less than any other than the actual bid was then the Sinai Discovery, and Nijmegen-Förusta. The first person for the first time since 1996. It was designe
- * exhausting the opponent, executing a submission hold, or striking the opponent.* →  Two of the Kingdom of its prestige. This way, or originating to a particular clerical part of the original Greek-speaking world and later styles, it was named after the surveillance and has had proven in the country, which he has been invaded, and that he was born in the village of Dennis and his d
- *de ("Food-Grade," or without added chemical stabilisers) for use as a mouthwash.* →  A major tenets of the 1800s to the south. The fortress was a very large but formats will not be inconsistent. The man of the Separatista. On April 4, 2005, the official-issue-trading entertainment. He spent several years after the war ended in the Battle of Belfast, the Symposium, it is called an i
- *dia Census Commission give a breakdown by various parameters including religion.* →  In the 1860s. In the first three letters of a confrontation with the French, Russian Orthodox Church, and the science of the same characters. The CDU's walled the sucked offspring. For the next generation of time in certain circles. The depression has also been suggested that he was proclaimed hims

**TR**

- * Motors New Zealand plant in Petone outside Wellington, until it closed in 1990.* →  After the Montreal Team in 1959, they moved to New South Wales, where they moved, from the north of New South Island to the south of New Hampshire. At that time, the tourists now operated on New Guinea. In 1970, the first BMW bike was brought under construction of the International Space Agency. In
- * granules of powder and the confinement determine the burn rate of black powder.* →  The penetrator has been used to reduce the firing of the fire. This is the use of the penetration and the use of, but the technique is more usually used, for the use of. The same is the case with the embryo. The penetl, a strong-neutral gas (a type of valve) is typically present in a vacuum in the 
- * exhausting the opponent, executing a submission hold, or striking the opponent.* →  Otherwise, they may drive a pass through the bottom with the leg off, and, in turn, the missing opponent can control the line. For example, if a man's opponent, with the other side, in that it can get the next point to the point where his opponent is trying to get the ball. If the right board is to
- *de ("Food-Grade," or without added chemical stabilisers) for use as a mouthwash.* →  Food products in the United States and Canada (e.g. Australia) have been used for more serious cases of anti-drugs. In addition, the Food andoods of the Unitedat, oral, are often used to cover the abdomen and are used to achieve interest in infection. These include the drug's baby, the Drug and Che
- *dia Census Commission give a breakdown by various parameters including religion.* →  The government of India has been underway to many bilateral organizations. The country has been reduced to 37,000 homes, and an estimated population of 1,000. Exceptions of the current country, the Indo-Arab Islamic Authority, the country's third largest city, is situated in the county, the Colombo

**Q0**

- * Motors New Zealand plant in Petone outside Wellington, until it closed in 1990.* →  In 1992, the Holden brand was discontinued in New Zealand. During the 1980s, Holdens were assembled in New Zealand, and the Holden Commodore was the first to be assembled in New Zealand from 1972 to 1990. In the same period, from 1970 to 1990, Holdens were assembled in New Zealand, and the Holden C
- * granules of powder and the confinement determine the burn rate of black powder.* →  For a powder charge to function, the ignition source must provide enough heat to initiate the combustion process. Black powder is used to drive projectiles in muzzleloading firearms and to propel torpedoes in black powder torpedoes. However, black powder is often less desirable than smokeless powde
- * exhausting the opponent, executing a submission hold, or striking the opponent.* →  /  / A dominant position is defined by a top grappler's control over a bottom grappler. Dominant positions include the mount, half guard, side control, and knee on belly. /  / In mount position, the top grappler is straddling the opponent's torso with their knees in the opponent's hip or shoulder area. The
- *de ("Food-Grade," or without added chemical stabilisers) for use as a mouthwash.* →  / Dental bleaching can also be accomplished with hydrogen peroxide. This is used by dentists to whiten teeth. Hydrogen peroxide is also used to remove stains from coffee, tea, and wine. Hydrogen peroxide is also used to remove stains from clothing. / Bleaching powder (calcium hypochlorite) / Bleaching po
- *dia Census Commission give a breakdown by various parameters including religion.* →  In 2001, 78. 43 percent of the population in India were Hindus, and 14. 2 percent were Muslims, according to the 2001 Census. The 2001 Census of India is a significant source of demographic and socio-economic data, as it provides information about the entire country, including population, literacy,

