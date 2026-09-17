#!/usr/bin/env python3
"""
COMPLETE DATASET CONVERTER - ALL 8 DATASETS
Converts all downloaded datasets to Cypha format
State managed - resumes automatically
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import traceback
import zipfile

STATE_FILE = "convert_state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except: pass
    return {"completed": [], "failed": {}, "in_progress": None, "converted_files": {}}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def mark_completed(state, name, output, samples, size_mb):
    if name not in state["completed"]:
        state["completed"].append(name)
    state["converted_files"][name] = {
        "output_file": output,
        "sample_count": samples,
        "size_mb": size_mb,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    if name in state["failed"]:
        del state["failed"][name]
    state["in_progress"] = None
    save_state(state)

def mark_failed(state, name, error):
    state["failed"][name] = {
        "error": str(error),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    state["in_progress"] = None
    save_state(state)

def is_completed(state, name):
    return name in state["completed"]

def show_status(state):
    print("\n" + "="*70)
    print("  CONVERSION STATUS")
    print("="*70)
    print(f"\n✅ Completed: {len(state['completed'])}")
    total_samples = 0
    for name in state["completed"]:
        if name in state["converted_files"]:
            info = state["converted_files"][name]
            print(f"  ✓ {name:30s} {info['sample_count']:>8,} samples  ({info['size_mb']:.1f} MB)")
            total_samples += info["sample_count"]
    print(f"\n❌ Failed: {len(state['failed'])}")
    for name, info in state["failed"].items():
        print(f"  ✗ {name:30s} {info['error']}")
    if total_samples > 0:
        print(f"\n💾 Total: {total_samples:,} samples")
    print("="*70)

# =====================================================================
# TEXT CONVERTERS
# =====================================================================

def convert_sql(input_file, output_file):
    print(f"\n  Reading {input_file}...")
    # sqli.csv has UTF-16 BOM (0xff 0xfe) — try encodings in order
    for enc in ('utf-16', 'latin-1', 'utf-8'):
        try:
            df = pd.read_csv(input_file, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError("Could not decode CSV with utf-16, latin-1, or utf-8")
    print(f"  Columns: {df.columns.tolist()}")
    
    query_col = label_col = None
    for col in df.columns:
        c = col.lower()
        if 'query' in c or 'sentence' in c or 'text' in c: query_col = col
        if 'label' in c or 'target' in c or 'class' in c: label_col = col
    
    if not query_col or not label_col:
        raise ValueError(f"Columns: {df.columns.tolist()}")
    
    print(f"  Using: {query_col} → {label_col}")
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in df.itertuples(index=False):
            try:
                query = str(getattr(row, query_col)).strip()
                if len(query) < 3: continue
                raw = getattr(row, label_col)
                label = 'safe' if raw in [0, '0', 'normal', 'benign', 'safe'] else 'sql_injection'
                f.write(f"{query}|||{label}\n")
                count += 1
                if count % 1000 == 0: print(f"  {count:,} samples", end='\r')
            except: pass
    print(f"\n  ✓ {count:,} samples")
    return count

def convert_phishing_urls_vrbancic(input_file, output_file):
    print(f"\n  Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Columns: {df.columns.tolist()[:10]}...")
    
    # This dataset has features, not raw URLs - need to extract URL column if exists
    # If no URL column, create text from features
    if 'url' in [c.lower() for c in df.columns]:
        url_col = [c for c in df.columns if c.lower() == 'url'][0]
    else:
        url_col = None
    
    label_col = 'phishing'  # The dataset has a 'phishing' column
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in df.itertuples(index=False):
            try:
                if url_col:
                    text = str(getattr(row, url_col)).strip()
                else:
                    features = []
                    for col in df.columns[:20]:
                        if col != label_col:
                            val = getattr(row, col)
                            if pd.notna(val):
                                features.append(f"{col}:{val}")
                    text = " ".join(features)
                if len(text) < 3: continue
                label = 'phishing' if getattr(row, label_col) == 1 else 'legitimate'
                f.write(f"{text}|||{label}\n")
                count += 1
                if count % 1000 == 0: print(f"  {count:,} samples", end='\r')
            except: pass
    print(f"\n  ✓ {count:,} samples")
    return count

def convert_malware(input_file, output_file):
    print(f"\n  Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    label_col = None
    for col in df.columns:
        if 'label' in col.lower() or 'class' in col.lower() or 'malware' in col.lower():
            label_col = col
            break
    
    if not label_col:
        raise ValueError("No label column")
    
    print(f"  Label: {label_col}")
    count = 0
    feat_cols = [c for c in df.columns if c != label_col]
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in df.itertuples(index=False):
            try:
                features = [f"{col}:{getattr(row, col)}"
                            for col in feat_cols
                            if pd.notna(getattr(row, col)) and getattr(row, col) != 0]
                if not features: continue
                text = " ".join(features[:50])
                label = str(getattr(row, label_col))
                f.write(f"{text}|||{label}\n")
                count += 1
                if count % 1000 == 0: print(f"  {count:,} samples", end='\r')
            except: pass
    print(f"\n  ✓ {count:,} samples")
    return count

def convert_network(input_file, output_file):
    print(f"\n  Reading {input_file}...")
    df = pd.read_csv(input_file, header=None)
    
    label_col = df.columns[-1]
    feat_cols = list(df.columns[:20])
    if label_col in feat_cols:
        feat_cols.remove(label_col)
    count = 0
    # Use values array — avoids itertuples integer-column renaming bug
    feat_idx = [df.columns.get_loc(c) for c in feat_cols]
    label_idx = df.columns.get_loc(label_col)
    arr = df.values
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in arr:
            try:
                features = []
                for i, col in zip(feat_idx, feat_cols):
                    val = row[i]
                    if pd.notna(val):
                        try: features.append(f"f{col}:{float(val):.2f}")
                        except: features.append(f"f{col}:{val}")
                text = " ".join(features)
                label_val = str(row[label_idx]).strip()
                label = 'normal' if 'normal' in label_val.lower() else 'anomaly'
                f.write(f"{text}|||{label}\n")
                count += 1
                if count % 1000 == 0: print(f"  {count:,} samples", end='\r')
            except: pass
    print(f"\n  ✓ {count:,} samples")
    return count

def convert_emails(input_file, output_file):
    print(f"\n  Reading {input_file}...")
    for enc in ('utf-8', 'latin-1', 'utf-16'):
        try:
            df = pd.read_csv(input_file, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError("Could not decode emails CSV")
    
    text_col = label_col = None
    for col in df.columns:
        c = col.lower()
        if 'text' in c or 'body' in c or 'email' in c or 'message' in c: text_col = col
        if 'label' in c or 'spam' in c or 'phish' in c: label_col = col
    
    if not text_col or not label_col:
        raise ValueError(f"Columns: {df.columns.tolist()}")
    
    print(f"  Using: {text_col} → {label_col}")
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in df.itertuples(index=False):
            try:
                text = str(getattr(row, text_col)).strip()[:500]
                if len(text) < 10: continue
                raw = getattr(row, label_col)
                label = 'safe' if raw in [0, '0', 'ham', 'safe'] else 'phishing'
                f.write(f"{text}|||{label}\n")
                count += 1
                if count % 1000 == 0: print(f"  {count:,} samples", end='\r')
            except: pass
    print(f"\n  ✓ {count:,} samples")
    return count

# =====================================================================
# SIGNAL CONVERTERS
# =====================================================================

def convert_panoradio_rf(input_file, output_file):
    """Convert Panoradio HF RF signal dataset - all samples via mmap (no RAM load)"""
    print(f"\n  Loading {input_file} via mmap...")

    # mmap_mode='r' memory-maps the file — only pages actually accessed are loaded
    data = np.load(input_file, mmap_mode='r')
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")

    csv_file = input_file.replace('.npy', '_tags.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Labels file not found: {csv_file}")

    labels_df = pd.read_csv(csv_file)
    print(f"  Labels: {len(labels_df)} rows, columns: {labels_df.columns.tolist()}")

    # Strip whitespace from column names, then detect label column
    labels_df.columns = labels_df.columns.str.strip()
    label_col = None
    for col in labels_df.columns:
        if col.lower() in ('label', 'tag', 'class', 'category', 'modulation',
                           'signal_type', 'mode', 'type'):
            label_col = col
            break
    if label_col is None:
        label_col = labels_df.columns[1] if len(labels_df.columns) > 1 else labels_df.columns[-1]
    print(f"  Using label column: '{label_col}'")
    print(f"  Classes: {labels_df[label_col].unique()}")

    n_samples = min(len(data), len(labels_df))
    print(f"  Converting all {n_samples:,} samples...")

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(n_samples):
            try:
                signal = data[i]
                signal_real_int = (signal.real * 127).astype(np.int8)
                signal_imag_int = (signal.imag * 127).astype(np.int8)
                signal_iq = np.empty(len(signal_real_int) + len(signal_imag_int), dtype=np.int8)
                signal_iq[0::2] = signal_real_int
                signal_iq[1::2] = signal_imag_int
                signal_hex = signal_iq.tobytes().hex()
                label = labels_df.iloc[i][label_col]
                f.write(f"iq:{signal_hex}|||{label}\n")
                count += 1
                if count % 1000 == 0:
                    print(f"  {count:,}/{n_samples:,} samples", end='\r')
            except Exception as e:
                if count == 0:
                    raise
    print(f"\n  ✓ {count:,} samples converted")
    return count

def convert_phiusiil(input_file, output_file):
    """Convert PhiUSIIL phishing URL dataset"""
    print(f"\n  Extracting {input_file}...")
    
    # Extract zip
    extract_dir = "./phiusiil_extracted"
    with zipfile.ZipFile(input_file, 'r') as zf:
        zf.extractall(extract_dir)
    
    # Find CSV
    csv_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        raise FileNotFoundError("No CSV in extracted archive")
    
    csv_file = csv_files[0]
    print(f"  Reading {csv_file}...")
    
    df = pd.read_csv(csv_file)
    print(f"  Records: {len(df):,}")
    
    # Find columns — strict URL match first, then fallback
    url_col = label_col = None
    cols_lower = {col: col.lower() for col in df.columns}

    # URL column: must be exactly 'url' or start with 'url' or be 'domain'
    for col, c in cols_lower.items():
        if c in ('url', 'domain', 'address', 'link') or c.startswith('url_') or c == 'url':
            url_col = col
            break
    # Fallback: column whose values look like URLs (start with http)
    if url_col is None:
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(5).tolist()
            if any(v.startswith('http') or v.startswith('www.') for v in sample):
                url_col = col
                break

    for col, c in cols_lower.items():
        if c in ('label', 'phishing', 'status', 'class', 'result', 'target'):
            label_col = col
            break
    
    if not url_col:
        raise ValueError(f"No URL column. Have: {df.columns.tolist()}")
    if not label_col:
        raise ValueError(f"No label column. Have: {df.columns.tolist()}")
    
    print(f"  Using: {url_col} → {label_col}")
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in df.itertuples(index=False):
            try:
                url = str(getattr(row, url_col)).strip()
                if len(url) < 5 or url == 'nan': continue
                label_val = getattr(row, label_col)
                if label_val in [0, '0', 'legitimate', 'safe', 'benign']:
                    label = 'legitimate'
                elif label_val in [1, '1', 'phishing', 'phish', 'malicious']:
                    label = 'phishing'
                else:
                    label = str(label_val)
                f.write(f"{url}|||{label}\n")
                count += 1
                if count % 10000 == 0:
                    print(f"  {count:,} samples", end='\r')
            except: pass
    
    print(f"\n  ✓ {count:,} samples converted")
    return count

# =====================================================================
# AUDIO CONVERTERS
# =====================================================================

def convert_speech_commands(input_file, output_file):
    """Convert Google Speech Commands dataset"""
    import tarfile
    import wave
    import glob

    extract_dir = "./speech_commands_extracted"

    # Count subdirs — Speech Commands v0.02 has 35 word dirs + extras
    # If we have fewer than 30, extraction was incomplete — redo it
    import shutil
    def count_subdirs(d):
        if not os.path.exists(d): return 0
        return sum(1 for x in os.listdir(d)
                   if os.path.isdir(os.path.join(d, x)) and not x.startswith('.'))

    if count_subdirs(extract_dir) < 30:
        print(f"\n  Extracting {input_file} (found {count_subdirs(extract_dir)} dirs, need 30+)...")
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(input_file, 'r:gz') as tf:
            try:
                tf.extractall(extract_dir, filter='data')
            except TypeError:
                tf.extractall(extract_dir)
    else:
        print(f"\n  Using cached extraction: {extract_dir} ({count_subdirs(extract_dir)} dirs)")
    
    print(f"  Finding audio files...")
    audio_files = []
    for word_dir in glob.glob(os.path.join(extract_dir, '*')):
        if os.path.isdir(word_dir) and not word_dir.endswith('_background_noise_'):
            word = os.path.basename(word_dir)
            wavs = glob.glob(os.path.join(word_dir, '*.wav'))
            for wav_file in wavs:
                audio_files.append((wav_file, word))
    
    print(f"  Found {len(audio_files)} audio files")
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for wav_path, label in audio_files:
            try:
                with wave.open(wav_path, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio_hex = frames.hex()
                
                f.write(f"pcm:{audio_hex}|||{label}\n")
                count += 1
                
                if count % 500 == 0:
                    print(f"  {count:,} samples", end='\r')
                
            except: pass
    
    print(f"\n  ✓ {count:,} samples converted")
    return count

def convert_esc50(input_file, output_file):
    """Convert ESC-50 environmental sound dataset"""
    import zipfile
    import glob
    import wave

    import shutil
    extract_dir = "./esc50_extracted"

    def do_extract():
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        print(f"\n  Extracting {input_file}...")
        with zipfile.ZipFile(input_file, 'r') as zf:
            zf.extractall(extract_dir)

    # Only extract if dir missing or metadata not found
    if not os.path.exists(extract_dir):
        do_extract()
    else:
        print(f"\n  Using cached extraction: {extract_dir}")

    meta_files = glob.glob(os.path.join(extract_dir, '**/meta/esc50.csv'), recursive=True)
    if not meta_files:
        # Partial/corrupt extraction — redo it
        print("  Metadata missing, re-extracting...")
        do_extract()
        meta_files = glob.glob(os.path.join(extract_dir, '**/meta/esc50.csv'), recursive=True)
    if not meta_files:
        raise FileNotFoundError("ESC-50 metadata not found after re-extraction")
    
    meta_file = meta_files[0]
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(meta_file)), 'audio')
    
    meta = pd.read_csv(meta_file)
    print(f"  Found {len(meta)} files in metadata")
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in meta.itertuples(index=False):
            try:
                filename = row.filename
                category = row.category
                wav_path = os.path.join(audio_dir, filename)
                if not os.path.exists(wav_path):
                    continue
                with wave.open(wav_path, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio_hex = frames.hex()
                f.write(f"pcm:{audio_hex}|||{category}\n")
                count += 1
                if count % 200 == 0:
                    print(f"  {count:,} samples", end='\r')
            except: pass
    
    print(f"\n  ✓ {count:,} samples converted")
    return count

# =====================================================================
# MAIN CONVERTER
# =====================================================================

def convert_dataset(state, name, input_file, output_file, converter_func):
    if is_completed(state, name):
        print(f"\n✓ {name} already converted")
        return
    
    if not os.path.exists(input_file):
        mark_failed(state, name, f"Input not found: {input_file}")
        print(f"\n✗ Missing: {input_file}")
        return
    
    print("\n" + "="*70)
    print(f"  CONVERTING: {name.upper()}")
    print("="*70)
    
    state["in_progress"] = name
    save_state(state)
    
    try:
        sample_count = converter_func(input_file, output_file)
        
        if sample_count == 0:
            raise ValueError("No samples converted")
        
        size_mb = os.path.getsize(output_file) / (1024*1024)
        mark_completed(state, name, output_file, sample_count, size_mb)
        
        print(f"\n✅ Success: {output_file} ({sample_count:,} samples)")
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n❌ Failed: {error_msg}")
        traceback.print_exc()
        mark_failed(state, name, error_msg)

def main():
    print("="*70)
    print("  COMPLETE DATASET CONVERTER - ALL 10 → 9 DATASETS")
    print("="*70)
    
    if "--status" in sys.argv:
        show_status(load_state())
        return
    
    if "--reset" in sys.argv:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            print("\n✓ State reset\n")
    
    retry_only = "--retry" in sys.argv
    state = load_state()
    
    # All conversions
    conversions = [
        ("sql_injection", "sqli.csv", "sql_injection.txt", convert_sql),
        ("phishing_urls_vrbancic", "phishing_vrbancic.csv", "phishing_vrbancic.txt", convert_phishing_urls_vrbancic),
        ("malware", "malware.csv", "malware.txt", convert_malware),
        ("network_intrusion", "network_intrusion.csv", "network_intrusion.txt", convert_network),
        ("phishing_emails", "phishing_emails.csv", "phishing_emails.txt", convert_emails),
        ("panoradio_rf", "dataset_panoradio_hf.npy", "panoradio_rf.txt", convert_panoradio_rf),
        ("phiusiil_phishing", "phiusiil_phishing.zip", "phiusiil_phishing.txt", convert_phiusiil),
        ("speech_commands", "speech_commands.tar.gz", "speech_commands.txt", convert_speech_commands),
        ("esc50", "esc50.zip", "esc50.txt", convert_esc50),
    ]
    
    if retry_only:
        conversions = [(n, i, o, f) for n, i, o, f in conversions if n in state["failed"]]
    
    # Convert all
    for name, inp, out, func in conversions:
        if retry_only or not is_completed(state, name):
            try:
                convert_dataset(state, name, inp, out, func)
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted - run again to resume")
                sys.exit(0)
    
    # Summary
    print("\n" + "="*70)
    print("  CONVERSION COMPLETE")
    print("="*70)
    show_status(state)
    
    if state["completed"]:
        print("\n📊 Next: python benchmark_datasets.py")
    if state["failed"]:
        print("\n⚠️  Retry: python convert_datasets.py --retry")
    print("="*70)

if __name__ == "__main__":
    main()
