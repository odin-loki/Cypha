#!/usr/bin/env python3
"""
COMPLETE DATASET DOWNLOADER - ALL 8 DATASETS
5 Security + 2 Signal + 1 Phishing
State managed - resumes automatically
"""

import os
import sys
import urllib.request
import json
import time
import random

STATE_FILE = "download_state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except: pass
    return {"completed": [], "failed": {}, "in_progress": None, "downloaded_files": {}}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def mark_completed(state, name, filename, size_mb):
    if name not in state["completed"]:
        state["completed"].append(name)
    state["downloaded_files"][name] = {
        "filename": filename,
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
    print("  DOWNLOAD STATUS")
    print("="*70)
    print(f"\n✅ Completed: {len(state['completed'])}")
    total_size = 0
    for name in state["completed"]:
        if name in state["downloaded_files"]:
            info = state["downloaded_files"][name]
            print(f"  ✓ {name:30s} {info['filename']:30s} ({info['size_mb']:.1f} MB)")
            total_size += info["size_mb"]
    print(f"\n❌ Failed: {len(state['failed'])}")
    for name, info in state["failed"].items():
        print(f"  ✗ {name:30s} {info['error']}")
    if state["in_progress"]:
        print(f"\n⏳ In Progress: {state['in_progress']}")
    total = len(state["completed"]) + len(state["failed"])
    if total > 0:
        print(f"\n📊 Success: {len(state['completed'])}/{total} ({100*len(state['completed'])/total:.1f}%)")
        print(f"💾 Downloaded: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print("="*70)

def download_file(url, filename, desc="Downloading", timeout=900, max_retries=3):
    print(f"\n{desc}")
    print(f"  URL: {url}")
    print(f"  File: {filename}")
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total = response.getheader('Content-Length')
                
                if total:
                    total = int(total)
                    print(f"  Size: {total/1024/1024:.1f} MB")
                    downloaded = 0
                    
                    with open(filename, 'wb') as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk: break
                            f.write(chunk)
                            downloaded += len(chunk)
                            pct = (downloaded / total) * 100
                            bar = '█' * int(40 * downloaded / total) + '░' * (40 - int(40 * downloaded / total))
                            print(f"  [{bar}] {pct:.1f}% ({downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB)", end='\r')
                    
                    print(f"\n  ✓ Downloaded: {downloaded / (1024*1024):.1f} MB")
                    return True, downloaded / (1024*1024)
                else:
                    content = response.read()
                    with open(filename, 'wb') as f:
                        f.write(content)
                    size_mb = len(content) / (1024*1024)
                    print(f"  ✓ Downloaded: {size_mb:.1f} MB")
                    return True, size_mb
                    
        except Exception as e:
            print(f"\n  ✗ Error: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 2)
                print(f"  ⏳ Retry in {wait:.1f}s ({attempt+2}/{max_retries})...")
                time.sleep(wait)
    
    return False, 0

# =====================================================================
# TEXT SECURITY DATASETS
# =====================================================================

def download_sql_injection(state):
    name = "sql_injection"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  1. SQL INJECTION DETECTION"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "https://raw.githubusercontent.com/ajinmathew/SQL-data/master/sqli.csv", "sqli.csv"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name}")
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

def download_phishing_urls_vrbancic(state):
    name = "phishing_urls_vrbancic"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  2. PHISHING URLS (88K SAMPLES)"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv", "phishing_vrbancic.csv"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name}")
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

def download_malware(state):
    name = "malware"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  3. ANDROID MALWARE"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "https://raw.githubusercontent.com/mburakergenc/Malware-Detection-using-Machine-Learning/master/data.csv", "malware.csv"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name}")
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

def download_network_intrusion(state):
    name = "network_intrusion"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  4. NETWORK INTRUSION (NSL-KDD)"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt", "network_intrusion.csv"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name}")
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

def download_phishing_emails(state):
    name = "phishing_emails"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  5. PHISHING EMAILS"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/SpamAssasin.csv", "phishing_emails.csv"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name}")
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

# =====================================================================
# SIGNAL DATASETS
# =====================================================================

def download_panoradio_rf(state):
    name = "panoradio_rf"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  6. PANORADIO HF RF SIGNALS (5 GB)"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "http://www.panoradio-sdr.de/wp-content/uploads/dataset_panoradio_hf.npy", "dataset_panoradio_hf.npy"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name} (5 GB, ~10-20 min)", timeout=1800)
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

def download_panoradio_tags(state):
    name = "panoradio_tags"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  7. PANORADIO TAGS (LABELS)"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "http://www.panoradio-sdr.de/wp-content/uploads/dataset_panoradio_hf_tags.csv", "dataset_panoradio_hf_tags.csv"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name}")
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

def download_phiusiil(state):
    name = "phiusiil_phishing"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  8. PHIUSIIL PHISHING URLS (235K)"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip", "phiusiil_phishing.zip"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name}")
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

# =====================================================================
# AUDIO DATASETS
# =====================================================================

def download_speech_commands(state):
    name = "speech_commands"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  9. GOOGLE SPEECH COMMANDS (1.5 GB)"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", "speech_commands.tar.gz"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name} (1.5 GB, ~5-10 min)", timeout=1800)
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

def download_esc50(state):
    name = "esc50"
    if is_completed(state, name): return
    print("\n" + "="*70); print("  10. ESC-50 AUDIO (ENVIRONMENTAL SOUNDS)"); print("="*70)
    state["in_progress"] = name; save_state(state)
    
    url, fn = "https://github.com/karolpiczak/ESC-50/archive/master.zip", "esc50.zip"
    if os.path.exists(fn) and os.path.getsize(fn) > 1000000:
        sz = os.path.getsize(fn)/(1024*1024)
        print(f"\n✓ Found existing: {fn} ({sz:.1f} MB)")
        mark_completed(state, name, fn, sz)
        return
    
    success, sz = download_file(url, fn, f"Downloading {name}")
    if success: mark_completed(state, name, fn, sz)
    else: mark_failed(state, name, "Download failed")

# =====================================================================
# MAIN
# =====================================================================

def main():
    print("="*70)
    print("  COMPLETE DATASET DOWNLOADER - ALL 10 DATASETS")
    print("  5 Security + 2 Signal + 1 Phishing + 2 Audio")
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
    
    # All downloads
    downloads = [
        ("sql_injection", download_sql_injection),
        ("phishing_urls_vrbancic", download_phishing_urls_vrbancic),
        ("malware", download_malware),
        ("network_intrusion", download_network_intrusion),
        ("phishing_emails", download_phishing_emails),
        ("panoradio_rf", download_panoradio_rf),
        ("panoradio_tags", download_panoradio_tags),
        ("phiusiil_phishing", download_phiusiil),
        ("speech_commands", download_speech_commands),
        ("esc50", download_esc50),
    ]
    
    if retry_only:
        downloads = [(n, f) for n, f in downloads if n in state["failed"]]
        if not downloads:
            print("\n✓ No failed downloads\n")
            return
    
    # Download all
    for name, func in downloads:
        if not retry_only and is_completed(state, name):
            print(f"\n✓ {name} already downloaded")
            continue
        
        try:
            func(state)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted - run again to resume")
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            mark_failed(state, name, str(e))
    
    # Summary
    print("\n" + "="*70)
    print("  DOWNLOAD COMPLETE")
    print("="*70)
    show_status(state)
    
    if state["completed"]:
        print("\n📊 Next: python convert_datasets.py")
    if state["failed"]:
        print("\n⚠️  Retry: python download_datasets.py --retry")
    print("="*70)

if __name__ == "__main__":
    main()
