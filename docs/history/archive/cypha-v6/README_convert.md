convert.py
Dataset Format Conversion & Wire Format Specification
File 3 of 5  ·  600 lines  ·  Step 2 of the pipeline
For someone reading this for the first time

# 1. What Is This File?
convert.py is step two of the pipeline. It reads every raw dataset file downloaded by download.py and transforms it into Cypha wire format — a plain text file where every line is exactly one training sample:

input_string  |||  label_string

That is the only format Cypha.py reads. The triple-pipe delimiter (|||) was chosen because it does not appear in any of the input domains: SQL queries, URLs, PE feature vectors, network flows, email text, RF hex strings, and audio hex strings all use different characters but none of them use three consecutive pipe characters.

| Where this sits in the pipeline download.py → convert.py → benchmark.py Reads 10 raw files (CSV, NPY, ZIP, TAR.GZ). Writes 9 wire-format .txt files. benchmark.py reads those .txt files directly. |
|---|

| python convert.py # convert all datasets python convert.py --status # check conversion progress python convert.py --retry # retry any failures python convert.py --reset # clear state, re-detect from scratch |
|---|

# 2. The Wire Format
Understanding the wire format is the key to understanding the whole pipeline. Once you grasp what each line looks like and why, everything else follows.
## 2.1 Basic Structure
Every line in every output file has exactly the same structure:
INPUT_STRING  |||  LABEL_STRING
There is no header row. There are no quotes. Every newline is a new sample. The input string can be arbitrarily long and contain any characters except newline and |||. The label string is a short identifier with no spaces.

| # Example lines from sql_injection.txt: SELECT id FROM users WHERE active=1|||safe ' OR 1=1 --|||sql_injection SELECT email FROM accounts WHERE user='alice'|||safe 1; DROP TABLE users --|||sql_injection  # Example lines from panoradio_rf.txt: iq:1a2b3c4d5e6f...8192 bytes of hex...|||am iq:ff01fe02fd03...8192 bytes of hex...|||fm  # Example lines from speech_commands.txt: pcm:0000010002000300...int16 PCM hex...|||yes pcm:fffefffefffd...int16 PCM hex...|||no |
|---|

## 2.2 The Three Input Types
The prefix at the start of the input string tells Cypha.py which encoder path to use:

| Prefix | Data type | Encoding | Cypha encoder path |
|---|---|---|---|
| (none) | Text, URLs, feature vectors | UTF-8 plain text | encode_text() — byte seq Ω + token stats |
| "iq:" | RF/radio signal | int8 IQ pairs as lowercase hex | _encode_iq() — STFT power spectral density |
| "pcm:" | Audio signal | int16 PCM samples as lowercase hex | _encode_audio() — mel filterbank |

| Why hex encoding for signals? The wire format is plain text (UTF-8). Binary data cannot be stored directly in a UTF-8 text file. Lowercase hex is used instead of base64 because it is easier to inspect visually and has no padding concerns. The overhead is 2 bytes per binary byte, which is acceptable since the files are only read once (streamed at training time, never loaded into RAM). |
|---|

## 2.3 The IQ Prefix — RF Signal Encoding
RF signals are stored as interleaved int8 I/Q pairs. Given a complex float32 signal S ∈ ℂⁿ:
I[k]  =  clip( round(127 · Re(S[k])), −127, 127 )  ∈ int8
Q[k]  =  clip( round(127 · Im(S[k])), −127, 127 )  ∈ int8
wire[2k]   = I[k]    (even positions = real component)
wire[2k+1] = Q[k]    (odd positions = imaginary component)
The byte array wire ∈ int8^{2n} is then hex-encoded and written as iq:HEXHEXHEX...
When Cypha reads this back, it calls bytes.fromhex(text[3:]), reinterprets as int8, splits into even/odd to recover I and Q, forms complex64 = I + iQ, and runs a 512-point short-time FFT. This produces the power spectral density that the Omega encoder then processes.

| Why not store raw float32? Float32 hex would be 8 characters per sample vs 2 for int8. The 4× size increase matters for a 5 GB RF dataset — it would become 20 GB. The int8 quantisation loses less than 0.4% of the dynamic range and has no measurable effect on classification accuracy. |
|---|

## 2.4 The PCM Prefix — Audio Encoding
WAV audio is stored as raw int16 PCM bytes — the same format the WAV file uses internally, with no resampling or channel mixing at this stage. convert.py reads WAV frames directly:

| with wave.open(wav_path, "rb") as wav: frames = wav.readframes(wav.getnframes()) audio_hex = frames.hex() f.write(f"pcm:{audio_hex}|||{label}\n") |
|---|

When Cypha reads this back, it calls bytes.fromhex(text[4:]), reinterprets as int16 little-endian, normalises to float32 ∈ [−1, 1], and passes through a 26-band mel filterbank with 512-point FFT windows. The mel filterbank output feeds into the Omega encoder.

# 3. State Machine
convert.py uses the same resumable state-machine design as download.py. The state is tracked in convert_state.json.

| { "completed": ["sql_injection", "malware", ...], "failed": { "panoradio_rf": { "error": "Labels file not found: ...", "timestamp": "..." } }, "in_progress": null, "converted_files": { "sql_injection": { "output_file": "sql_injection.txt", "sample_count": 29442, "size_mb": 1.87, "timestamp": "2026-02-21 09:30:12" } } } |
|---|

Conversions are sequential, not parallel. Only one dataset is in_progress at a time. If the process is killed mid-conversion, the output file may be incomplete — on the next run, the in_progress entry causes that dataset to be re-converted from scratch. The output file is always written completely before mark_completed() is called.

# 4. Converter Details
Each dataset has a dedicated converter function. The sections below describe what each one does, what quirks it handles, and what the output looks like.
## 4.1  convert_sql  →  sql_injection.txt
Input: sqli.csv (UTF-16 encoded, ~30k rows, two relevant columns).

### Encoding quirk
The file uses UTF-16 with a byte order mark (BOM: 0xff 0xfe). Python's default UTF-8 reader fails on it. The converter tries three encodings in order — utf-16, latin-1, utf-8 — and uses the first that succeeds.

### Column detection
Rather than hardcoding column names, the converter scans all column names and matches by keyword:
query_col: column name contains "query", "sentence", or "text"
label_col: column name contains "label", "target", or "class"

### Label normalisation
The raw label values vary by dataset version (some use 0/1 integers, some use "normal"/"benign"). The converter normalises to two canonical labels:
0 | "0" | "normal" | "benign" | "safe"  →  safe
anything else  →  sql_injection

| # Sample output lines: SELECT * FROM employees WHERE dept='engineering' LIMIT 10|||safe ' UNION SELECT username, password FROM admin --|||sql_injection |
|---|

## 4.2  convert_phishing_urls_vrbancic  →  phishing_vrbancic.txt
Input: phishing_vrbancic.csv (~88k rows, 48 URL feature columns plus a "phishing" label column).

### URL vs feature vector
If a column named exactly "url" exists, the raw URL string is used as the input. If not (which is the case for this dataset — it only has engineered features), the first 20 feature columns are concatenated as "colname:value" pairs:
| # If no URL column: features = ["NumDots:3", "SubdomainLevel:2", "PathLevel:1", ...] text = " ".join(features) |
|---|

Labels: phishing if the "phishing" column = 1, else legitimate.
| # Sample output: NumDots:3 SubdomainLevel:2 PathLevel:1 UrlLength:52 ...|||phishing NumDots:1 SubdomainLevel:0 PathLevel:2 UrlLength:28 ...|||legitimate |
|---|

## 4.3  convert_malware  →  malware.txt
Input: malware.csv (~10k rows, PE feature columns plus a label column).

### Feature serialisation
Every non-label column where the value is non-null and non-zero is included as "colname:value". Zero values are dropped because they add no information and would dominate the token count. At most 50 features per row are included (the most informative features appear first in the column ordering):
| # Sample output: SizeOfCode:24576 MajorLinkerVersion:14 SizeOfInitializedData:8192 ...|||benign SizeOfCode:98304 VirtualAlloc:1 CreateRemoteThread:1 WriteProcessMemory:1 ...|||malware |
|---|

## 4.4  convert_network  →  network_intrusion.txt
Input: network_intrusion.csv (NSL-KDD format, no header row, 41 feature columns, last column is label).

### No header row
NSL-KDD uses no column names — pandas assigns numeric column indices 0, 1, 2, …, 41. The label is always the last column (index 41). The converter uses the first 20 feature columns.

### Integer column name renaming bug
pandas itertuples() renames integer column names to _0, _1, etc. (reserved attribute names). The converter avoids this by working directly with the numpy values array (df.values) and using integer indices rather than attribute access:
| # BUG: this breaks when column names are integers for row in df.itertuples(): val = row.0 # SyntaxError or AttributeError  # FIX: use values array directly arr = df.values for row in arr: val = row[feat_idx[i]] # always works |
|---|

### Binary label mapping
All 39 attack types in NSL-KDD (neptune, back, portsweep, warezclient, etc.) are mapped to a single "anomaly" label for binary classification. Only "normal" stays as is:
"normal" in label_val.lower()  →  normal
anything else  →  anomaly
| # Sample output: f0:0.00 f1:0.00 f2:0.00 f3:215.00 f4:45076.00 ...|||normal f0:0.00 f1:0.00 f2:0.00 f3:105.00 f4:146.00 ...|||anomaly |
|---|

## 4.5  convert_emails  →  phishing_emails.txt
Input: phishing_emails.csv (SpamAssassin corpus, UTF-8 or latin-1, text + label columns).

### Column detection
text_col: column name contains "text", "body", "email", or "message"
label_col: column name contains "label", "spam", or "phish"

### Truncation
Email bodies are truncated to 500 characters. The Omega encoder's byte sequence features are dominated by the statistical signature of the first few hundred bytes, so longer inputs add noise rather than signal. Minimum length is 10 characters — shorter "emails" are dropped.
text  =  raw_email_body.strip()[:500]
| # Sample output (truncated for display): Dear valued customer, your account has been suspended. Click here immediately to...|||phishing Hi team, the Q3 report is attached. Let me know if you have questions...|||safe |
|---|

## 4.6  convert_panoradio_rf  →  panoradio_rf.txt
Input: dataset_panoradio_hf.npy (5 GB NumPy float32 array) + dataset_panoradio_hf_tags.csv (labels).

| Memory-mapped loading The 5 GB NPY file is loaded with np.load(path, mmap_mode="r"). This does not read the file into RAM — it maps the file into virtual address space. Only the pages actually accessed (one signal row at a time) are loaded from disk. Peak RAM usage during conversion is dominated by the output buffer and pandas, not by the signal data. |
|---|

### Label column detection
The tags CSV has no standardised column schema. The converter tries a list of known column names (label, tag, class, category, modulation, signal_type, mode, type) and falls back to the second column if none match.

### Signal encoding
Each row of the NPY array is a complex float32 signal. The converter scales to int8 range, interleaves I/Q, and hex-encodes:
| signal = data[i] # float32 complex row I = (signal.real * 127).astype(np.int8) # scale to [-127, 127] Q = (signal.imag * 127).astype(np.int8) iq = np.empty(len(I) + len(Q), dtype=np.int8) iq[0::2] = I # even indices = I iq[1::2] = Q # odd indices = Q hex_str = iq.tobytes().hex() f.write(f"iq:{hex_str}|||{label}\n") |
|---|

Output labels include: am, fm, lsb, usb, cw, rtty, and others depending on the tag file version. These are the radio modulation modes captured in the HF (high-frequency, 3–30 MHz) band.
## 4.7  convert_phiusiil  →  phiusiil_phishing.txt
Input: phiusiil_phishing.zip (UCI ML Repository, 235k rows, 111 URL feature columns).

### ZIP extraction
The ZIP is extracted to ./phiusiil_extracted/. The converter then walks the directory tree to find any .csv file. This handles nested zip structures where the CSV may be one or two directories deep.

### URL vs feature column detection
Unlike the Vrbancic dataset, PHIUSIIL may contain an actual URL column. The detector checks:
Exact column name match: "url", "domain", "address", "link"
If no exact match, inspect the values of each column — if any sample starts with "http" or "www.", that column is the URL column.
If still no match, raise an error with the available column list.

Labels: 1 → phishing, 0 → legitimate (also accepts string variants "phishing"/"legitimate").
## 4.8  convert_speech_commands  →  speech_commands.txt
Input: speech_commands.tar.gz (TensorFlow speech commands v0.02, 35 word directories, ~105k WAV files).

### Extraction with corruption guard
The TAR.GZ is extracted to ./speech_commands_extracted/. The extractor counts subdirectories before and after — if fewer than 30 subdirectories exist, the extraction is deemed incomplete and retried from scratch (removing the partial extraction first). Speech Commands v0.02 has 35 word directories plus a background noise directory.
| def count_subdirs(d): return sum(1 for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)) and not x.startswith("."))  if count_subdirs(extract_dir) < 30: shutil.rmtree(extract_dir) # remove partial re_extract() |
|---|

### Background noise exclusion
The directory _background_noise_ is excluded — it contains long ambient recordings used for data augmentation during neural network training, not labelled word examples. Including it would create a spurious "background_noise" class.

### WAV reading
Python's built-in wave module reads the WAV frames. No resampling. The raw int16 bytes are hex-encoded directly — no normalisation, no channel conversion at this stage. Cypha's PCM encoder handles the int16 → float conversion internally.
| # Sample output: pcm:0000000000000000010000000000000001000000...|||yes pcm:fffffffefffffffdfffffffeffff...|||no |
|---|

## 4.9  convert_esc50  →  esc50.txt
Input: esc50.zip (ESC-50 environmental sounds, 2000 WAV files, 50 classes × 40 samples each).

### Metadata-driven
Unlike Speech Commands (where labels come from directory names), ESC-50 stores labels in a metadata CSV at ESC-50-master/meta/esc50.csv. The converter reads the metadata to pair each WAV filename with its category label. This is more robust than parsing directory structure.
| # esc50.csv structure (first few columns): filename,fold,target,category,... 1-100032-A-0.wav,1,0,dog,... 1-100038-A-14.wav,1,14,chirping_birds,... |
|---|

### Re-extraction guard
If the metadata file cannot be found after extraction, the extractor assumes the ZIP was corrupt or partially extracted and retries. The metadata path is searched recursively using glob.glob("**/meta/esc50.csv", recursive=True) to handle any nesting depth.

# 5. Output Files
After a complete run, nine .txt files are written to the current directory:

| Output file | Samples | Approx size | Input prefix | Labels |
|---|---|---|---|---|
| sql_injection.txt | ~29k | ~2 MB | none | safe, sql_injection |
| phishing_vrbancic.txt | ~88k | ~22 MB | none | legitimate, phishing |
| malware.txt | ~10k | ~3 MB | none | benign, malware |
| network_intrusion.txt | ~126k | ~15 MB | none | normal, anomaly |
| phishing_emails.txt | ~10k | ~3 MB | none | safe, phishing |
| panoradio_rf.txt | ~400k | ~2–4 GB | "iq:" | am, fm, lsb, usb, cw, rtty, ... |
| phiusiil_phishing.txt | ~235k | ~60 MB | none | legitimate, phishing |
| speech_commands.txt | ~105k | ~1.5 GB | "pcm:" | 35 word classes (yes, no, go, ...) |
| esc50.txt | 2,000 | ~600 MB | "pcm:" | 50 sound classes (dog, rain, ...) |

panoradio_rf.txt and speech_commands.txt are large because each signal is hex-encoded (2× the binary size). They are only read once at benchmark time via byte-offset streaming — they are never loaded into RAM in their entirety.

# 6. The convert_dataset() Orchestrator
All nine converter functions are invoked through a single orchestrator function:

| def convert_dataset(state, name, input_file, output_file, converter_func): if is_completed(state, name): return # skip already done if not os.path.exists(input_file): # missing raw file mark_failed(state, name, f"Input not found: {input_file}") return  state["in_progress"] = name save_state(state)  try: sample_count = converter_func(input_file, output_file) if sample_count == 0: raise ValueError("No samples converted") size_mb = os.path.getsize(output_file) / (1024*1024) mark_completed(state, name, output_file, sample_count, size_mb) except Exception as e: mark_failed(state, name, f"{type(e).__name__}: {e}") traceback.print_exc() |
|---|

The orchestrator guarantees that:
A dataset is never marked completed unless its converter returns sample_count > 0.
Any exception (including import errors, missing dependencies, malformed data) is caught, logged, and stored in the failed dict without aborting the remaining conversions.
KeyboardInterrupt propagates cleanly with a clear resume message.
# 7. Function Reference

| Function | Input | Purpose |
|---|---|---|
| load_state() | — | Read convert_state.json. Returns empty state if missing. |
| save_state(state) | state dict | Write state to convert_state.json. |
| mark_completed(...) | state, name, output, n, mb | Record successful conversion. |
| mark_failed(state, name, err) | state, str, str | Record failure with error message. |
| is_completed(state, name) | state, str | True if name in state["completed"]. |
| show_status(state) | state dict | Print formatted conversion status. |
| convert_sql(in, out) | CSV (UTF-16) | SQL injection CSV → wire format. |
| convert_phishing_urls_vrbancic(in, out) | CSV | Phishing URL features → wire format. |
| convert_malware(in, out) | CSV | Malware PE features → wire format. |
| convert_network(in, out) | CSV (no header) | NSL-KDD flows → wire format. |
| convert_emails(in, out) | CSV | Email bodies → wire format. |
| convert_panoradio_rf(in, out) | NPY + CSV | 5 GB RF signals → iq: wire format via mmap. |
| convert_phiusiil(in, out) | ZIP | PHIUSIIL phishing ZIP → wire format. |
| convert_speech_commands(in, out) | TAR.GZ | Speech Commands WAVs → pcm: wire format. |
| convert_esc50(in, out) | ZIP | ESC-50 WAVs → pcm: wire format using metadata CSV. |
| convert_dataset(state, name, in, out, func) | all | Orchestrator: skip/guard/exception-wrap any converter. |
| main() | — | Entry point. Parses flags, runs all conversions. |

# 8. Troubleshooting
## 8.1  panoradio_rf fails with "Labels file not found"
Both dataset_panoradio_hf.npy and dataset_panoradio_hf_tags.csv must be present. The tags file is a separate download (panoradio_tags in download.py). Check that both exist:
| ls -lh dataset_panoradio_hf.npy dataset_panoradio_hf_tags.csv |
|---|

## 8.2  speech_commands fails with tarfile errors
If speech_commands.tar.gz is partially downloaded or corrupt, tarfile.open() will raise an exception. Delete the TAR file, re-run download.py --retry, and then re-run convert.py --retry.
## 8.3  "No samples converted"
This means the converter ran without raising an exception but wrote zero valid lines. Common causes:
The column detection failed — the auto-detected query_col, label_col, or url_col is wrong. Check the printed "Columns:" line and compare with what the converter is looking for.
The input file is empty or a partial download. Check file size with ls -lh.
All rows were skipped due to the minimum length filter (len(text) < 3 or < 10). This can happen if the column mapping picked the wrong column.
## 8.4  Slow conversion for panoradio_rf
Converting 400k signals at ~5 KB each (after hex encoding) takes 15–30 minutes. The mmap approach ensures it does not run out of RAM, but disk I/O is the bottleneck. Use an SSD if possible.

End of convert.py reference.  Next: benchmark.py
