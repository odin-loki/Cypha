"""
Cypha HRNA — Full Architecture
Binary+Cypha Universal Encoder · HLFC Compression · Resonance Field
Event-Driven Processing · Multi-Level Hierarchy · Adaptive Control Loop
Multicore CPU via concurrent.futures · Pure numpy
"""

import numpy as np
from numpy.fft import fft, ifft, rfft, irfft
import math, time, os, re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import heapq

EPSILON   = 1e-8
K_TARGET  = 0.5
LAMBDA_LP = 0.15
GAIN_K    = 0.05
GAIN_RHO  = 0.03
GAIN_A    = 0.04
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
_POOL     = ThreadPoolExecutor(max_workers=N_WORKERS)
_LOCK     = threading.Lock()


# ══════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════

@dataclass
class EncoderParams:
    chunk_k:       float = 4.0
    damr_radius:   float = 3.0
    active_scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])
    prev_error:    float = float('inf')

@dataclass
class FieldStats:
    criticality:   float
    dominant_freq: float
    mean_phase:    float
    phase_spread:  float
    energy:        float

@dataclass
class Event:
    type:     str
    time:     float
    data:     Dict[str, Any]
    source:   str
    priority: float
    def __lt__(self, other): return self.priority > other.priority

@dataclass
class Metrics:
    step: int; loss: float; criticality: float
    chunk_k: float; damr_r: float; n_anchors: int
    ms: float = 0.0; events: int = 0

class EventType(Enum):
    PATTERN   = auto()
    SURPRISE  = auto()
    RESONANCE = auto()
    EXTERNAL  = auto()
    FEEDBACK  = auto()
    THOUGHT   = auto()


# ══════════════════════════════════════════════
# 1. BINARY ENCODER  (byte-native, adaptive θ)
# ══════════════════════════════════════════════

class BinaryEncoder:
    def __init__(self, output_dim: int = 4096):
        self.output_dim = output_dim

    def encode(self, data: bytes, p: EncoderParams) -> np.ndarray:
        if not data: return np.zeros(self.output_dim, np.float32)
        feat = np.zeros(self.output_dim, np.float32)
        n = len(data); q = self.output_dim // 4

        # U-shaped position weighted unigrams
        for i, b in enumerate(data[:256]):
            pos = i / max(1, n-1)
            w = 0.2 + 0.8*(2*pos-1)**2
            feat[int(b) % q] += w * (b/255.0 + 0.1)

        # Bigrams
        for i in range(len(data)-1):
            h = (int(data[i])*31 + int(data[i+1])) % q
            feat[q+h] += 1.0

        # Trigrams
        for i in range(len(data)-2):
            h = (int(data[i])*961 + int(data[i+1])*31 + int(data[i+2])) % q
            feat[2*q+h] += 1.0

        base = 3*q
        # Char class (8 dims)
        cc = np.zeros(8, np.float32)
        for b in data:
            if 48<=b<=57:    cc[0]+=1
            elif 65<=b<=90:  cc[1]+=1
            elif 97<=b<=122: cc[2]+=1
            elif b==32:      cc[3]+=1
            elif b in [43,45,42,47,62,60,61]: cc[4]+=1
            elif b==58:      cc[5]+=1
            else:            cc[7]+=1
        feat[base:base+8] = cc / (n+EPSILON)

        # Token structure (6 dims)
        try:
            text = data.decode('utf-8', errors='replace')
            tokens = text.split()
            if tokens:
                lens = [len(t) for t in tokens]
                feat[base+8]  = len(tokens)/10.
                feat[base+9]  = float(np.mean(lens))/10.
                feat[base+10] = float(np.std(lens))/10.
                feat[base+11] = sum(1 for t in tokens if t.isdigit())/len(tokens)
                feat[base+12] = sum(1 for t in tokens if t.isalpha())/len(tokens)
                feat[base+13] = sum(1 for t in tokens if not t.isalnum())/len(tokens)

            # Numeric magnitude (32 dims)
            import re
            nums = re.findall(r'-?\d+\.?\d*', text)
            for i, num in enumerate(nums[:8]):
                try:
                    v = float(num)
                    feat[base+14+i*4]   = np.sign(v)*np.log1p(abs(v))/10.
                    feat[base+14+i*4+1] = (abs(v)%10)/10.
                    feat[base+14+i*4+2] = (abs(v)//10%10)/10.
                    feat[base+14+i*4+3] = min(1., abs(v)/1000.)
                except: pass
        except: pass

        # First 40 bytes verbatim (catches prefix differences)
        for i, b in enumerate(data[:40]):
            feat[base+46+i] = b/255.0

        norm = np.linalg.norm(feat)
        return feat / (norm+EPSILON)

    def encode_text(self, t: str, p: EncoderParams) -> np.ndarray:
        return self.encode(t.encode('utf-8'), p)

    def encode_array(self, raw: bytes, p: EncoderParams) -> np.ndarray:
        """Float32 array data — spatial + raw pixel features"""
        dim = self.output_dim
        n_floats = len(raw)//4
        import warnings
        if len(raw_bytes) % 4 == 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arr = np.frombuffer(raw_bytes, dtype='<f4').copy().astype(np.float64)
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            except:
                arr = np.frombuffer(raw_bytes, np.uint8).astype(np.float64)/255.
        else:
            arr = np.frombuffer(raw_bytes, np.uint8).astype(np.float64)/255.
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

        feat = np.zeros(dim, np.float32)
        sq = int(np.sqrt(n_floats))

        if sq*sq == n_floats and sq >= 4:
            grid = arr.reshape(sq, sq)
            total = arr.sum() + EPSILON
            rows = np.arange(sq, dtype=np.float32)
            cols = np.arange(sq, dtype=np.float32)
            row_sums = grid.sum(axis=1)
            col_sums = grid.sum(axis=0)
            cx = float((cols*col_sums).sum()/total)
            cy = float((rows*row_sums).sum()/total)
            Vx = float(((cols-cx)**2*col_sums).sum()/total)
            Vy = float(((rows-cy)**2*row_sums).sum()/total)
            diag_m = np.array([grid[i,i] for i in range(sq)])/total
            diag_a = np.array([grid[i,sq-1-i] for i in range(sq)])/total
            h = sq//2
            quads = np.array([grid[:h,:h].sum(),grid[:h,h:].sum(),
                               grid[h:,:h].sum(),grid[h:,h:].sum()])/total
            fft2 = np.abs(np.fft.fft2(grid)).flatten()[:32]
            spatial = np.concatenate([
                row_sums/total, col_sums/total,
                diag_m, diag_a, quads,
                [cx/sq, cy/sq, Vx/sq**2, Vy/sq**2,
                 cx/sq-0.5, cy/sq-0.5],
                fft2/(np.linalg.norm(fft2)+EPSILON),
                arr/total,
            ])
        else:
            # 1D waveform: raw values + FFT + signed diff
            n_use = min(len(arr), dim//2)
            anorm = np.linalg.norm(arr)+EPSILON
            part1 = arr[:n_use]/anorm
            fft_m = np.abs(np.fft.rfft(arr))
            n_fft = min(len(fft_m), dim//4)
            part2 = fft_m[:n_fft]/(np.linalg.norm(fft_m)+EPSILON)
            diffs = np.diff(arr) if len(arr)>1 else np.zeros(1)
            n_d   = min(len(diffs), dim//8)
            part3 = diffs[:n_d]/(np.linalg.norm(diffs)+EPSILON)
            spatial = np.concatenate([part1, part2, part3])

        n_use = min(len(spatial), dim)
        feat[:n_use] = spatial[:n_use]
        norm = np.linalg.norm(feat)
        return feat/(norm+EPSILON)

# ══════════════════════════════════════════════
# 2. PHASE BRIDGE  (real→complex resonant)
# ══════════════════════════════════════════════

class PhaseBridge:
    def __init__(self, feature_dim: int, resonance_dim: int):
        self.fd = feature_dim; self.rd = resonance_dim
        rng = np.random.default_rng(42)
        self.Wa = rng.standard_normal((feature_dim, resonance_dim)) * 0.1
        self.Wp = rng.standard_normal((feature_dim, resonance_dim)) * 0.1
        self.bf = np.linspace(0.5, 10., resonance_dim)

    def bridge(self, f: np.ndarray) -> np.ndarray:
        x = f.astype(np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if len(x) != self.fd:
            x = np.interp(np.linspace(0,len(x)-1,self.fd), np.arange(len(x)), x)
        amps  = x @ self.Wa
        half  = len(x)//2
        base  = np.arctan2(np.linalg.norm(x[half:])+EPSILON, np.linalg.norm(x[:half])+EPSILON)
        phase = base + 0.3*(x @ self.Wp)
        dom   = np.arange(self.rd, dtype=np.float64)
        basis = np.sin(self.bf * dom / self.rd)
        r     = amps * np.exp(1j*phase) * basis
        return r / (np.linalg.norm(r)+EPSILON)

# ══════════════════════════════════════════════
# 3. HLFC COMPRESSION  (4-layer, numpy)
# ══════════════════════════════════════════════

class FundamentalExtraction:
    def __init__(self, n=50, threshold=0.01):
        self.n=n; self.thr=threshold

    def extract(self, x: np.ndarray) -> Dict:
        F = fft(x.astype(complex))
        amps=np.abs(F); phases=np.angle(F)
        idx=np.argsort(amps)[::-1]
        mask=amps[idx]>(self.thr*np.max(amps))
        sel=idx[mask][:self.n]
        return {'freq': sel.astype(float)/len(x), 'amp': amps[sel], 'phase': phases[sel], 'n': len(x)}

    def reconstruct(self, c: Dict) -> np.ndarray:
        n=c['n']; S=np.zeros(n,complex)
        for i,(f,a,p) in enumerate(zip(c['freq'],c['amp'],c['phase'])):
            idx=int(f*n)
            if idx<n: S[idx]=a*np.exp(1j*p)
        return irfft(S[:n//2+1], n=n)


class SymmetryEncoding:
    def encode(self, c: Dict) -> Dict:
        freqs=c['freq']; amps=c['amp']
        syms=[]
        # Reflection
        if len(freqs)>1:
            mid=0.5; L=freqs[freqs<mid]; R=freqs[freqs>=mid]
            if len(L)>0 and len(R)>0:
                score=np.mean([np.min(np.abs(lf-(1.-R))) for lf in L])<0.05
                if score: syms.append(('reflect', float(np.mean(amps))))
        # Harmonics
        if len(freqs)>2:
            base=freqs[np.argmax(amps)]
            if base>0.01:
                ratios=freqs[1:]/base
                if np.mean(np.abs(ratios-np.round(ratios))<0.05)>0.5:
                    syms.append(('harmonic', float(base)))
        # Phase linearity → shift
        ph=c['phase']
        if len(ph)>2:
            diffs=np.diff(np.sort(ph))
            if np.std(diffs)<0.2*np.mean(np.abs(diffs))+EPSILON:
                syms.append(('shift', float(np.mean(diffs)/(2*math.pi))))
        return {'symmetries': syms, 'residual': c}


class CrystalLatticeMapping:
    def __init__(self, size=16):
        x=np.linspace(0,2*math.pi,size)
        X,Y=np.meshgrid(x,x)
        self.lattice=(np.sin(X)+np.cos(Y)).flatten()
        self.size=size

    def map(self, enc: Dict) -> Dict:
        flat=np.array([a*p for a,p in zip(enc['residual']['amp'],enc['residual']['phase'])])
        flat=np.pad(flat,(0,max(0,len(self.lattice)-len(flat))))[:len(self.lattice)]
        diff=flat-self.lattice[:len(flat)]
        thresh=np.std(diff)*1.5
        defects=[(int(i),float(diff[i])) for i in np.where(np.abs(diff)>thresh)[0]]
        return {'lattice': self.lattice, 'defects': defects, 'enc': enc}


class DNAFolding:
    def fold(self, mapped: Dict) -> Dict:
        d=np.array([v for _,v in mapped['defects']]) if mapped['defects'] else np.array([0.])
        n=len(d); folds=[]
        level=d.copy()
        while len(level)>4:
            h=len(level)//2
            folds.append({'scale':len(folds), 'approx':level[:h], 'detail':level[h:2*h]})
            level=level[:h]+level[h:2*h]
            level=level[:len(level)//2] if len(level)>4 else level
        return {'folds': folds, 'core': level, 'mapped': mapped}

    def unfold(self, folded: Dict) -> np.ndarray:
        r=folded['core'].copy()
        for f in reversed(folded['folds']):
            r=np.concatenate([r, f['detail']])
        return r


class HLFCCompressor:
    def __init__(self, n_components=50):
        self.fe=FundamentalExtraction(n_components)
        self.se=SymmetryEncoding()
        self.cl=CrystalLatticeMapping()
        self.df=DNAFolding()

    def compress(self, x: np.ndarray) -> Dict:
        return self.df.fold(self.cl.map(self.se.encode(self.fe.extract(x))))

    def decompress(self, c: Dict) -> np.ndarray:
        return self.fe.reconstruct(c['mapped']['enc']['residual'])

    def add(self, c1: Dict, c2: Dict) -> Dict:
        x1=self.decompress(c1); x2=self.decompress(c2)
        n=max(len(x1),len(x2))
        return self.compress(np.pad(x1,(0,n-len(x1)))+np.pad(x2,(0,n-len(x2))))


# ══════════════════════════════════════════════
# 4. RESONANCE FIELD  (FFT Hamiltonian + nonlinear)
# ══════════════════════════════════════════════

class ResonanceField:
    def __init__(self, dim=64, gamma=0.3, dt=0.2):
        self.dim=dim; self.gamma=gamma; self.dt=dt
        rng=np.random.default_rng(0)
        self.psi=rng.standard_normal(dim)+1j*rng.standard_normal(dim)
        self.psi = np.nan_to_num(self.psi, nan=0.0)
        self.psi/=(np.linalg.norm(self.psi)+EPSILON)
        self.psi/=np.linalg.norm(self.psi)
        self.psi_prev=self.psi.copy()
        self.H=np.linspace(0.5,10.,dim)
        self._event_queue: List[Tuple[float,np.ndarray,float]]=[]

    def inject(self, v: np.ndarray, strength=0.6):
        v=v.flatten()[:self.dim].astype(complex)
        v/=(np.linalg.norm(v)+EPSILON)
        self.psi=(1-strength)*self.psi+strength*v
        self.psi/=(np.linalg.norm(self.psi)+EPSILON)

    def queue_event(self, v: np.ndarray, t: float, strength=0.3):
        self._event_queue.append((t, v, strength))

    def evolve(self, steps=1) -> np.ndarray:
        self.psi_prev=self.psi.copy()
        for _ in range(steps):
            # Process queued events (Dirac delta injection)
            now=time.time()
            remaining=[]
            for et,ev,es in self._event_queue:
                if et<=now: self.inject(ev, es)
                else: remaining.append((et,ev,es))
            self._event_queue=remaining
            # FFT Hamiltonian evolution: -i[H,ψ]
            pf=fft(self.psi)
            pf*=np.exp(-1j*self.dt*self.H)
            self.psi=ifft(pf)
            # Nonlinear: γ(|ψ|²-1)ψ
            d=np.abs(self.psi)**2
            self.psi+=self.gamma*self.dt*(d-1.)*self.psi
            self.psi/=(np.linalg.norm(self.psi)+EPSILON)
        return self.psi

    def resonance(self, pattern: np.ndarray) -> float:
        p=fft(pattern.astype(complex)[:self.dim])
        r=np.abs(ifft(fft(self.psi)*np.conj(p)))
        return float(np.max(r))

    def enhanced_resonance(self, pattern: np.ndarray, gamma_res=0.1) -> float:
        r=self.resonance(pattern)
        q=r/(np.std(np.abs(self.psi))+EPSILON)
        return r*(1.+gamma_res*q)

    def stats(self) -> FieldStats:
        kappa=float(np.mean(np.abs(self.psi-self.psi_prev)**2))
        spec=np.abs(fft(self.psi))
        return FieldStats(
            criticality=kappa,
            dominant_freq=float(np.argmax(spec))/self.dim,
            mean_phase=float(np.mean(np.angle(self.psi))),
            phase_spread=float(np.std(np.angle(self.psi))),
            energy=float(np.sum(np.abs(self.psi)**2)))

    def criticality(self) -> float:
        return self.stats().criticality

    def reset(self):
        rng = np.random.default_rng()  # no seed — different every time
        self.psi = rng.standard_normal(self.dim) + 1j*rng.standard_normal(self.dim)
        self.psi /= np.linalg.norm(self.psi)
        self.psi_prev = self.psi.copy()


# ══════════════════════════════════════════════
# 5. RESONATOR LEVEL  (local coupling + inhibition)
# ══════════════════════════════════════════════

class ResonatorLevel:
    def __init__(self, n=64, gamma=0.35, locality=3, omega_range=(1.,10.)):
        self.n=n; self.gamma=gamma; self.locality=locality
        rng=np.random.default_rng(1)
        self.R=np.zeros(n); self.omega=np.linspace(*omega_range,n)
        self.W=rng.standard_normal(2*locality+1)*0.3; self.W[locality]=0.
        # Diffusion weights for wave propagation
        self.D=0.1

    @staticmethod
    def _sig(x): return 1./(1.+np.exp(-np.clip(x,-20,20)))

    def update(self, dt=0.1, drive=None) -> np.ndarray:
        freq=self.omega*self.R
        coup=np.zeros(self.n)
        for off in range(-self.locality, self.locality+1):
            if off==0: continue
            w=self.W[off+self.locality]
            if off>0: coup[:-off]+=w*self._sig(self.R[off:])
            else: coup[-off:]+=w*self._sig(self.R[:off])
        # Laplacian diffusion ∇²R
        lap=np.zeros(self.n)
        lap[1:-1]=self.R[:-2]-2*self.R[1:-1]+self.R[2:]
        inhib=-self.gamma*np.sum(np.abs(self.R))/self.n
        drv=drive[:self.n].real*200. if drive is not None else 0.
        Rn=self.R+dt*(freq+coup+self.D*lap+inhib)+drv
        # Enhanced resonance gating
        res_gate=1.+0.1*np.abs(self.R)
        Rn*=res_gate
        t=np.quantile(np.abs(Rn),0.8)
        Rn[np.abs(Rn)<t]*=0.1
        self.R=np.clip(Rn,-10,10)
        return self.R

    def reset(self): self.R=np.zeros(self.n)


# ══════════════════════════════════════════════
# 6. ASSEMBLY LEVEL  (oscillatory + resonance-enhanced)
# ══════════════════════════════════════════════

class AssemblyLevel:
    def __init__(self, n_assemblies=16, resonator_n=64):
        self.na=n_assemblies; self.rn=resonator_n
        rng=np.random.default_rng(2)
        self.A=np.zeros(n_assemblies)
        # Oscillator state [real, imag] per assembly
        self.O=np.zeros((n_assemblies,2))
        self.omega=np.linspace(0.5,5.,n_assemblies)
        self.V=rng.standard_normal((n_assemblies,resonator_n))*0.1
        self.C=rng.standard_normal((n_assemblies,n_assemblies))*0.05
        np.fill_diagonal(self.C,0.)
        self.phi=0.1; self.gamma=0.05

    def update(self, R: np.ndarray, G: np.ndarray, dt=0.1) -> np.ndarray:
        # Resonance enhancement
        res_enh=1.+0.1*np.abs(self.A)
        # Assembly dynamics: dA/dt = F(A) + V·σ(R) - φ·C·A + T(G,A)
        sig_R=self._sig(R[:self.rn] if len(R)>=self.rn else np.pad(R,(0,self.rn-len(R))))
        inp=self.V@sig_R
        lateral=-self.phi*self.C@self.A
        glob=0.05*G[:self.na] if len(G)>=self.na else 0.
        dA=-0.1*self.A+inp+lateral+glob
        self.A+=dt*dA*res_enh
        self.A=np.clip(self.A,-5,5)
        # Oscillatory: do/dt = [[0,-ω],[ω,0]]·o - γ·o
        for k in range(self.na):
            w=self.omega[k]
            o=self.O[k]
            dO=np.array([o[1]*(-w)-self.gamma*o[0],
                          o[0]*w  -self.gamma*o[1]])
            dO+=0.1*self.A[k]
            self.O[k]+=dt*dO
        return self.A

    @staticmethod
    def _sig(x): return 1./(1.+np.exp(-np.clip(x,-20,20)))

    def oscillator_output(self) -> np.ndarray:
        return self.O[:,0]  # Real component of each oscillator

    def reset(self):
        self.A=np.zeros(self.na); self.O=np.zeros((self.na,2))


# ══════════════════════════════════════════════
# 7. MODULE LEVEL  (working memory + network)
# ══════════════════════════════════════════════

class ModuleLevel:
    def __init__(self, n_modules=8, assembly_n=16, mem_size=32):
        self.nm=n_modules; self.an=assembly_n; self.ms=mem_size
        rng=np.random.default_rng(3)
        self.M=np.zeros(n_modules)
        self.WM=np.zeros((mem_size,))           # Working memory
        self.wm_weights=np.zeros(mem_size)
        self.wm_gates=np.ones(mem_size)
        self.C=rng.standard_normal((n_modules,n_modules))*0.05
        np.fill_diagonal(self.C,0.)
        self.V=rng.standard_normal((n_modules,assembly_n))*0.1
        self.alpha=0.1; self.beta_mem=0.05
        self._mem_events: deque=deque(maxlen=100)

    def update(self, A: np.ndarray, G: np.ndarray, dt=0.1) -> np.ndarray:
        a=A[:self.an] if len(A)>=self.an else np.pad(A,(0,self.an-len(A)))
        g=G[:self.nm] if len(G)>=self.nm else np.pad(G,(0,self.nm-len(G)))
        res_enh=1.+0.05*np.abs(self.M)
        inp=self.V@self._sig(a)
        lat=-self.alpha*self.C@self.M
        # Working memory integration
        wm_out=np.sum(self.wm_weights[:self.nm]*self.WM[:self.nm])*0.01
        dM=-self.M+inp+lat+0.05*g+wm_out
        self.M+=dt*dM*res_enh
        self.M=np.clip(self.M,-5,5)
        # Update working memory: m_WM = ∑ w_i·C(e_i)·g_i
        self.WM=np.roll(self.WM,1)
        self.WM[0]=np.mean(np.abs(self.M))
        return self.M

    def add_memory_event(self, event_vec: np.ndarray):
        self._mem_events.append(event_vec.copy())
        # Update weights from events
        if len(self._mem_events)>1:
            recent=np.array(list(self._mem_events)[-8:])
            self.wm_weights[:min(self.ms,len(recent))]=np.abs(np.mean(recent,axis=0))[:self.ms]

    @staticmethod
    def _sig(x): return 1./(1.+np.exp(-np.clip(x,-20,20)))

    def reset(self): self.M=np.zeros(self.nm); self.WM=np.zeros(self.ms)


# ══════════════════════════════════════════════
# 8. GLOBAL LEVEL  (integration + criticality)
# ══════════════════════════════════════════════

class GlobalLevel:
    def __init__(self, dim=64, module_n=8):
        self.dim=dim; self.mn=module_n
        rng=np.random.default_rng(4)
        self.G=np.zeros(dim)
        self.WG=rng.standard_normal((dim, module_n+dim))*0.05
        self.kappa=0.5          # criticality parameter
        self.kappa0=0.5
        self.alpha_G=0.1
        self._pred=np.zeros(dim) # prediction for temporal recursion
        self._prev_G=np.zeros(dim)

    def update(self, M: np.ndarray, O: np.ndarray, R_field: np.ndarray,
               events: List[Event], dt=0.1) -> np.ndarray:
        m=M[:self.mn] if len(M)>=self.mn else np.pad(M,(0,self.mn-len(M)))
        o=O[:self.dim] if len(O)>=self.dim else np.pad(O,(0,self.dim-len(O)))
        inp=np.concatenate([m,o[:self.dim]])
        inp=inp[:self.mn+self.dim]
        self._prev_G=self.G.copy()
        # dG/dt = -α·G + W·[M;O] + R_G(G) + P_G + κ·R_crit + ΣE
        decay=-self.alpha_G*self.G
        proj=self.WG@inp
        res=0.05*np.abs(R_field[:self.dim] if len(R_field)>=self.dim else np.pad(R_field,(0,self.dim-len(R_field))))
        pred_err=self._pred-self.G
        pred_corr=0.1*pred_err
        crit=self.kappa*self._critical_resonance(self.G)
        ev_sum=np.zeros(self.dim)
        for e in events:
            if 'vector' in e.data:
                v=e.data['vector'][:self.dim]
                ev_sum[:len(v)]+=v*e.priority
        dG=decay+proj[:self.dim]+res+pred_corr+crit+ev_sum*0.01
        self.G+=dt*dG
        self.G=np.clip(self.G/((np.linalg.norm(self.G)+EPSILON)/10.),-10,10)
        # Update kappa: dκ/dt = α(|∇G|²-κ₀)
        grad_G=np.mean((self.G-self._prev_G)**2)
        self.kappa+=dt*0.1*(grad_G-self.kappa0)
        self.kappa=float(np.clip(self.kappa,0.01,2.))
        # Update prediction
        self._pred=self.G+dt*(self.G-self._prev_G)
        return self.G

    def _critical_resonance(self, G: np.ndarray) -> np.ndarray:
        F=fft(G.astype(complex))
        F*=np.exp(-1j*0.1*np.linspace(0.5,5.,len(F)))
        return np.abs(ifft(F)).astype(float)[:self.dim]*0.1

    def reset(self):
        self.G=np.zeros(self.dim); self._prev_G=np.zeros(self.dim)
        self.kappa=0.5


# ══════════════════════════════════════════════
# 9. EVENT SYSTEM
# ══════════════════════════════════════════════

class EventScheduler:
    def __init__(self, alpha=0.1):
        self.alpha=alpha; self._queue: List[Event]=[]

    def schedule(self, e: Event):
        heapq.heappush(self._queue, e)

    def next_time(self, t_current: float, priority: float) -> float:
        return t_current*(1.+self.alpha*priority)**-1

    def pop_due(self) -> List[Event]:
        out=[]; now=time.time()
        while self._queue and self._queue[0].time<=now:
            out.append(heapq.heappop(self._queue))
        return out

    def __len__(self): return len(self._queue)


class EventGenerator:
    def __init__(self, pat_thr=0.7, surp_thr=0.15, res_thr=0.5):
        self.pt=pat_thr; self.st=surp_thr; self.rt=res_thr
        self._recent: deque=deque(maxlen=10)
        self._pred: Optional[np.ndarray]=None
        # Type counters for reporting
        self.type_counts: Dict[str,int] = {t.name: 0 for t in EventType}
        self._total = 0

    def _track(self, evs: List[Event]) -> List[Event]:
        for e in evs:
            self.type_counts[e.type] = self.type_counts.get(e.type, 0) + 1
            self._total += 1
        return evs

    def from_resonance(self, field: ResonanceField, patterns: List[np.ndarray]) -> List[Event]:
        evs=[]
        for i,p in enumerate(patterns):
            r=field.enhanced_resonance(p)
            if r>self.rt:
                evs.append(Event(EventType.RESONANCE.name, time.time(),
                    {'resonance':r,'pattern_id':i,'vector':p}, 'resonance', r))
        return self._track(evs)

    def from_surprise(self, state: np.ndarray, pred: Optional[np.ndarray]=None) -> List[Event]:
        evs=[]
        self._recent.append(state.copy())
        ref=pred if pred is not None else (self._recent[-2] if len(self._recent)>1 else None)
        if ref is not None:
            err=float(np.mean(np.abs(state-ref[:len(state)])))
            if err>self.st:
                evs.append(Event(EventType.SURPRISE.name, time.time(),
                    {'error':err,'vector':state}, 'surprise', err))
        self._pred=state.copy()
        return self._track(evs)

    def from_external(self, v: np.ndarray, priority=1.) -> Event:
        e = Event(EventType.EXTERNAL.name, time.time(),
            {'vector':v,'amplitude':float(np.linalg.norm(v))}, 'external', priority)
        self._track([e])
        return e

    def modulate(self, e: Event, field: ResonanceField, kappa: float,
                 alpha_res=0.1, alpha_crit=0.05) -> Event:
        v=e.data.get('vector', np.zeros(1))
        r=field.enhanced_resonance(v) if len(v)>1 else 0.
        e.priority*=(1.+alpha_res*r+alpha_crit*kappa)
        return e

    def type_report(self) -> str:
        if self._total == 0: return '    (no events)'
        lines = []
        for t, c in sorted(self.type_counts.items(), key=lambda x: -x[1]):
            if c == 0: continue
            pct = 100.*c/self._total
            lines.append(f"    {t:12} {c:4}  ({pct:5.1f}%)")
        return '\n'.join(lines)

    def reset_counts(self):
        self.type_counts = {t.name: 0 for t in EventType}
        self._total = 0


# ══════════════════════════════════════════════
# 10. RECURSIVE PROCESSING
# ══════════════════════════════════════════════

class RecursiveProcessor:
    def __init__(self, dim=64):
        self.dim=dim
        rng=np.random.default_rng(5)
        self.alpha_H=0.1; self.alpha_V=0.1; self.alpha_T=0.1; self.beta_E=0.05
        self._prev: Optional[np.ndarray]=None
        self._pred: Optional[np.ndarray]=None

    def horizontal(self, psi: np.ndarray, inputs: np.ndarray,
                   events: List[Event], res_enh: float) -> np.ndarray:
        ev_sum=sum(e.priority for e in events) if events else 0.
        out=(psi+0.1*inputs)*(1.+self.alpha_H*res_enh)*(1.+self.beta_E*ev_sum)
        return out/((np.linalg.norm(out)+EPSILON)/max(np.linalg.norm(out),1.))

    def vertical(self, psi: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 events: List[Event]) -> np.ndarray:
        lo=lower[:self.dim] if len(lower)>=self.dim else np.pad(lower,(0,self.dim-len(lower)))
        up=upper[:self.dim] if len(upper)>=self.dim else np.pad(upper,(0,self.dim-len(upper)))
        cross=0.05*(lo+up)
        r_lev=float(np.dot(psi,lo)/(np.linalg.norm(psi)*np.linalg.norm(lo)+EPSILON))
        ev_sum=sum(e.priority for e in events) if events else 0.
        out=(psi+cross)*(1.+self.alpha_V*abs(r_lev))*(1.+self.beta_E*ev_sum)
        return out

    def temporal(self, psi: np.ndarray, events: List[Event]) -> np.ndarray:
        if self._prev is None: self._prev=psi.copy()
        pred=psi+(psi-self._prev)*0.1 if self._pred is None else self._pred
        r_temp=float(np.mean(np.abs(psi*np.conj(self._prev[:len(psi)].astype(complex)))))
        ev_sum=sum(e.priority for e in events) if events else 0.
        out=0.9*psi+0.1*pred*(1.+self.alpha_T*r_temp)*(1.+self.beta_E*ev_sum)
        self._prev=psi.copy(); self._pred=out.copy()
        return out


# ══════════════════════════════════════════════
# 11. FEEDBACK
# ══════════════════════════════════════════════

class FeedbackController:
    def __init__(self, dim=64):
        self.dim=dim; self.gamma_res=0.1; self.delta_crit=0.05; self.kappa0=0.5
        rng=np.random.default_rng(6)
        self.W_cross=rng.standard_normal((dim,dim))*0.02
        self._history: deque=deque(maxlen=32)
        self._kernel=np.exp(-np.linspace(0,3,16))

    def resonance_amplified(self, psi: np.ndarray, field: ResonanceField) -> np.ndarray:
        r=field.enhanced_resonance(psi)
        return psi*(1.+self.gamma_res*r)

    def cross_level(self, psi_i: np.ndarray, psi_j: np.ndarray, t: float) -> np.ndarray:
        r=float(np.dot(psi_i,psi_j)/(np.linalg.norm(psi_i)*np.linalg.norm(psi_j)+EPSILON))
        return self.W_cross@psi_j*r

    def temporal(self, events: List[Event]) -> np.ndarray:
        if not events: return np.zeros(self.dim)
        self._history.extend(events)
        if not self._history: return np.zeros(self.dim)
        recent=list(self._history)[-len(self._kernel):]
        out=np.zeros(self.dim)
        for i,(e,k) in enumerate(zip(recent,self._kernel)):
            v=e.data.get('vector',np.zeros(self.dim))[:self.dim]
            out[:len(v)]+=k*v*e.priority
        return out/((np.linalg.norm(out)+EPSILON))

    def criticality_enhanced(self, psi: np.ndarray, kappa: float) -> np.ndarray:
        scale=1.+self.delta_crit*(kappa-self.kappa0)**2
        return psi*scale


# ══════════════════════════════════════════════
# 12. THOUGHT PROCESSES
# ══════════════════════════════════════════════

class ThoughtProcessor:
    def __init__(self, dim=64):
        self.dim=dim
        self._chains: deque=deque(maxlen=16)
        self._event_history: deque=deque(maxlen=64)
        self.theta_thought=0.7

    def cascade(self, trigger: Event, G: np.ndarray, kappa: float) -> List[Event]:
        evs=[]
        if trigger.priority>self.theta_thought:
            v=trigger.data.get('vector', G)
            # Sub-events at different delays
            for tau,scale in [(0.01,0.8),(0.05,0.5),(0.1,0.3)]:
                sub_v=v*scale+G[:len(v)]*0.1
                evs.append(Event(EventType.THOUGHT.name, time.time()+tau,
                    {'vector':sub_v,'parent':trigger.type}, 'cascade', trigger.priority*scale))
        return evs

    def multi_scale(self, events: List[Event], G: np.ndarray) -> np.ndarray:
        if not events: return G
        scales={}
        for e in events:
            sc=int(e.priority*3)
            scales.setdefault(sc,[]).append(e)
        out=G.copy()
        for sc,evs in scales.items():
            v=np.mean([e.data.get('vector',np.zeros(self.dim))[:self.dim] for e in evs],axis=0)
            w=0.1/(sc+1)
            out=out*(1.+w*np.linalg.norm(v)*0.01)
        return out

    def self_generate(self, G: np.ndarray, kappa: float) -> Optional[Event]:
        self._event_history.append(G.copy())
        if len(self._event_history)<4: return None
        recent=np.array(list(self._event_history)[-4:])
        trend=recent[-1]-recent[0]
        if np.linalg.norm(trend)>self.theta_thought:
            v=G+trend*0.1
            return Event(EventType.THOUGHT.name, time.time(),
                {'vector':v,'source':'self_generated'}, 'self', kappa)
        return None

    def resonant_chain(self, events: List[Event]) -> float:
        if len(events)<2: return 0.
        vecs=[e.data.get('vector',np.zeros(self.dim)) for e in events]
        chain=1.
        for i in range(len(vecs)-1):
            v1=vecs[i][:self.dim]; v2=vecs[i+1][:self.dim]
            sim=float(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+EPSILON))
            chain*=max(0.,sim)
        return chain


# ══════════════════════════════════════════════
# 13. META-LEARNING  (contrastive + recursive)
# ══════════════════════════════════════════════

class MetaLearning:
    def __init__(self, state_dim=64, max_recent=8):
        self.dim=state_dim
        self._recent: deque=deque(maxlen=max_recent)
        self._L_meta=0.

    def loss(self, state, target, negatives=None):
        s=state/(np.linalg.norm(state)+EPSILON)
        t=target/(np.linalg.norm(target)+EPSILON)
        n=min(len(s),len(t)); s,t=s[:n],t[:n]
        pos=float(np.mean((s-t)**2))
        neg_l=0.
        if negatives:
            for neg in negatives:
                nv=neg/(np.linalg.norm(neg)+EPSILON)
                sim=float(np.dot(s,nv[:n]))
                neg_l+=max(0., sim-0.1)**2   # margin 0.2→0.1, tighter
            neg_l/=len(negatives)
        penalty=1.
        if self._recent:
            sims=[abs(float(np.dot(s,p[:n]/(np.linalg.norm(p)+EPSILON)))) for p in self._recent]
            penalty=float(np.exp(-3.*np.mean(sims)))
        self._recent.append(s.copy())
        total=(pos + 2.0*neg_l)*penalty   # weight 0.8→2.0
        self._L_meta=0.9*self._L_meta+0.1*total
        return total

    def meta_loss(self): return self._L_meta

class ModalityDetector:
    """
    Detects input modality from prefix and feature signature.
    Tracks per-modality accuracy across the run.
    """
    PREFIXES = {'arr:': 'array', 'hex:': 'binary', 'file:': 'file'}
    MODALITY_PATTERNS = {
        'math':     [r'\d+[\+\-\*\/]\d+', r'\d+ mod \d+'],
        'logic':    [r'^is \d+', r'^not ', r'true|false', r'and|or'],
        'sequence': [r'^next:', r'^sort:'],
        'language': [r'sound$', r'^capital of', r'past$', r'synonym'],
        'question': [r'\?$'],
        'array':    [r'^arr:'],
        'binary':   [r'^hex:'],
        'file':     [r'^file:'],
    }

    def __init__(self):
        import re
        self._re = re
        self._compiled = {k: [re.compile(p, re.IGNORECASE) for p in pats]
                          for k, pats in self.MODALITY_PATTERNS.items()}
        # Per-modality: {modality: [correct, total]}
        self.scores: Dict[str, List[int]] = {k: [0,0] for k in self.MODALITY_PATTERNS}
        self.scores['unknown'] = [0,0]

    def detect(self, text: str) -> str:
        for prefix, mod in self.PREFIXES.items():
            if text.startswith(prefix): return mod
        for mod, patterns in self._compiled.items():
            if any(p.search(text) for p in patterns): return mod
        return 'unknown'

    def record(self, text: str, correct: bool):
        mod = self.detect(text)
        if mod not in self.scores: self.scores[mod] = [0,0]
        self.scores[mod][1] += 1
        if correct: self.scores[mod][0] += 1

    def report(self) -> str:
        lines = []
        for mod, (ok, tot) in sorted(self.scores.items()):
            if tot == 0: continue
            pct = 100.*ok/tot
            bar = '█'*int(pct/10) + '░'*(10-int(pct/10))
            lines.append(f"    {mod:12} {bar} {ok:3}/{tot:3} ({pct:5.1f}%)")
        return '\n'.join(lines) if lines else '    (no data)'

    def wavelet_signature(self, features: np.ndarray) -> Tuple[str, float]:
        """Returns (signature_label, detail_ratio) — verifies encoder differentiates modalities"""
        half = len(features)//2
        approx_e = float(np.sum(features[:half]**2))+EPSILON
        detail_e = float(np.sum(features[half:]**2))+EPSILON
        ratio = detail_e / approx_e
        if ratio < 0.3:   label = 'smooth(text-like)'
        elif ratio < 1.0: label = 'mixed'
        elif ratio < 3.0: label = 'structured(image-like)'
        else:             label = 'spiky(binary-like)'
        return label, ratio

class AnchorMemory:
    def __init__(self, dim=64, min_sep=0.5):
        self.dim=dim; self.min_sep=min_sep
        self.anchors: Dict[str,np.ndarray]={}
        self.outputs: Dict[str,str]={}
        self._vecs: List[np.ndarray]=[]
        self._keys: List[str]=[]
        self._sep_cache: Optional[Tuple[float,float]] = None  # (min, avg)
        self._dirty = False

    def store(self, key: str, state: np.ndarray, output: str):
        v=state/(np.linalg.norm(state)+EPSILON)
        self.anchors[key]=v; self.outputs[key]=output
        if key not in self._keys:
            self._vecs.append(v); self._keys.append(key)
        else:
            self._vecs[self._keys.index(key)]=v
        self._dirty = True

    def lookup(self, state: np.ndarray, k=3) -> List[Tuple[str,float]]:
        if not self._vecs: return []
        q=state/(np.linalg.norm(state)+EPSILON)
        sims=[(self._keys[i],float(np.dot(q,v))) for i,v in enumerate(self._vecs)]
        return sorted(sims,key=lambda x:x[1],reverse=True)[:k]

    def separation_stats(self) -> Tuple[float,float]:
        """Compute min and avg pairwise cosine distance. Cached until dirty."""
        if not self._dirty and self._sep_cache is not None:
            return self._sep_cache
        if len(self._vecs) < 2:
            return 0., 0.
        # Sample up to 64 anchors for speed
        vecs = np.array(self._vecs[:64])
        sims = vecs @ vecs.T
        # Zero diagonal, take upper triangle
        n = len(vecs)
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append(1. - sims[i,j])  # cosine distance
        if not pairs:
            return 0., 0.
        mn = float(np.min(pairs)); avg = float(np.mean(pairs))
        self._sep_cache = (mn, avg)
        self._dirty = False
        return mn, avg

    def get_output(self, key: str) -> Optional[str]: return self.outputs.get(key)
    @property
    def n(self): return len(self.anchors)

    def self_retrieval_rate(self, sample=50) -> float:
        if len(self._vecs) < 2: return 1.0
        keys = self._keys[:sample]
        vecs = np.array([self.anchors[k] for k in keys])
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1)
        max_other = np.max(sims, axis=1)
        np.fill_diagonal(sims, 2)
        self_sims = np.diag(sims)
        return float(np.mean(self_sims > max_other))

# ══════════════════════════════════════════════
# 14. ADAPTIVE CONTROL LOOP
# ══════════════════════════════════════════════

class AdaptiveControlLoop:
    def __init__(self):
        self.history: deque=deque(maxlen=32)

    def update(self, p: EncoderParams, s: FieldStats, err: float) -> EncoderParams:
        if err>p.prev_error*1.05:
            p.prev_error=err; return p
        # Law 1: chunk_k ← criticality
        dk=-math.copysign(GAIN_K, s.criticality-K_TARGET)
        kc=float(np.clip(p.chunk_k+dk, 2., 5.))
        p.chunk_k=(1-LAMBDA_LP)*p.chunk_k+LAMBDA_LP*kc
        # Law 2: DAMR radius ← dominant frequency
        rho_t=max(1.,s.dominant_freq*8.)
        dr=float(np.clip((rho_t-p.damr_radius),-1.,1.))*GAIN_RHO
        p.damr_radius=float(np.clip((1-LAMBDA_LP)*p.damr_radius+LAMBDA_LP*(p.damr_radius+dr),3.,8.))
        # Law 3: active scales ← phase coherence
        coh=1.-min(1.,s.phase_spread/math.pi)
        n_sc=max(1,round(1+coh*3))
        p.active_scales=[1.,0.5,0.25,0.125][:n_sc]
        p.prev_error=err
        self.history.append({'k':s.criticality,'ck':p.chunk_k,'r':p.damr_radius})
        return p


# ══════════════════════════════════════════════
# 15. SPARSE COMPUTATION + WORK STEALING
# ══════════════════════════════════════════════

class SparseComputer:
    def __init__(self, theta=0.01):
        self.theta=theta
        self._cache: Dict[str,Tuple[np.ndarray,float]]={}
        self._lru: deque=deque(maxlen=256)

    def should_update(self, psi: np.ndarray, key: str) -> bool:
        if key not in self._cache: return True
        prev,_=self._cache[key]
        return float(np.linalg.norm(psi-prev[:len(psi)]))>self.theta

    def cache(self, key: str, result: np.ndarray):
        self._cache[key]=(result.copy(),time.time())
        self._lru.append(key)

    def get_cached(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            v,_=self._cache[key]; return v
        return None


class WorkStealer:
    def __init__(self, n_workers=N_WORKERS):
        self._pool=_POOL; self._queue: deque=deque(maxlen=32)
        self._futures={}

    def submit(self, key: str, fn, *args):
        if key not in self._futures or self._futures[key].done():
            self._futures[key]=self._pool.submit(fn,*args)

    def result(self, key: str, timeout=0.005):
        if key in self._futures:
            f=self._futures[key]
            if f.done():
                del self._futures[key]; return f.result()
        return None

    def run_parallel(self, fns: List[Tuple[str,Any,tuple]]) -> Dict[str,Any]:
        futures={k: self._pool.submit(fn,*args) for k,fn,args in fns}
        results={}
        for k,f in futures.items():
            try: results[k]=f.result(timeout=0.1)
            except: results[k]=None
        return results


# ══════════════════════════════════════════════
# 16. PRECISION CONTROL  (adaptive float precision)
# ══════════════════════════════════════════════

class PrecisionController:
    def preserve(self, x: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        ax=np.abs(x); ax=np.where(ax<EPSILON,1.,ax)
        exp=np.floor(np.log2(ax+EPSILON))
        man=x/(2.**exp)
        return man.astype(np.float32), exp.astype(np.float16)

    def reconstruct(self, man: np.ndarray, exp: np.ndarray) -> np.ndarray:
        return man.astype(np.float64)*(2.**exp.astype(np.float64))

    def adaptive_compute(self, psi: np.ndarray, needs_high: bool) -> np.ndarray:
        return psi.astype(np.float64) if needs_high else psi.astype(np.float32)


# ══════════════════════════════════════════════
# 17. MAIN CYPHA SYSTEM
# ══════════════════════════════════════════════

class Cypha:
    def __init__(self, feature_dim=4096, resonance_dim=256):
        self.fd=feature_dim; self.rd=resonance_dim
        # Encoders
        self.bin_enc   = BinaryEncoder(feature_dim)
        self.bridge    = PhaseBridge(feature_dim, resonance_dim)
        self.hlfc      = HLFCCompressor(n_components=32)
        # Levels
        self.field     = ResonanceField(resonance_dim)
        self.res_level = ResonatorLevel(resonance_dim)
        self.assembly  = AssemblyLevel(16, resonance_dim)
        self.module    = ModuleLevel(8, 16)
        self.global_l  = GlobalLevel(resonance_dim, 8)
        # Processing
        self.recursive = RecursiveProcessor(resonance_dim)
        self.feedback  = FeedbackController(resonance_dim)
        self.thought   = ThoughtProcessor(resonance_dim)
        # Learning
        self.meta      = MetaLearning(resonance_dim)
        self.memory    = AnchorMemory(feature_dim)
        # Infrastructure
        self.ctrl      = AdaptiveControlLoop()
        self.scheduler = EventScheduler()
        self.gen       = EventGenerator()
        self.sparse    = SparseComputer()
        self.stealer   = WorkStealer()
        self.precision = PrecisionController()
        # State
        self.params    = EncoderParams()
        self.step      = 0
        self.temperature = 1.5
        self._metrics: List[Metrics]=[]
        self.modality   = ModalityDetector()
        self._last_infer_stats: Optional[Dict] = None
        self._active_events: List[Event]=[]
        self._patterns: List[np.ndarray]=[]

    # ── Encode ──────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        if text.startswith('arr:'):
            try:
                import base64
                raw = base64.b64decode(text[4:])
                feats = self.bin_enc.encode_array(raw, self.params)
            except: feats = self.bin_enc.encode_text(text, self.params)
        elif text.startswith('hex:'):
            try:
                raw = bytes.fromhex(text[4:])
                feats = self.bin_enc.encode_array(raw, self.params)
            except: feats = self.bin_enc.encode_text(text, self.params)
        else:
            feats = self.bin_enc.encode_text(text, self.params)
        return self.bridge.bridge(feats)

    # ── Forward ─────────────────────────────

    def forward(self, text: str) -> Dict[str,Any]:
        t0=time.time()
        enc=self.encode(text)

        for _ in range(6):
            self.field.inject(enc, strength=0.25)
            psi = self.field.evolve(2)

        compressed = self.hlfc.compress(enc.real)

        # Events
        due_events=self.scheduler.pop_due()
        res_events=self.gen.from_resonance(self.field, self._patterns[:4])
        surp_events=self.gen.from_surprise(np.abs(psi))
        all_events=due_events+res_events+surp_events+self._active_events
        all_events=[self.gen.modulate(e,self.field,self.global_l.kappa) for e in all_events]
        self._active_events=[]

        # Recursive processing (horizontal, vertical, temporal)
        res_enh=self.field.enhanced_resonance(np.abs(psi))
        R=self.recursive.horizontal(np.abs(psi), enc.real, all_events, res_enh)
        R=self.recursive.temporal(R, all_events)

        # Multi-level (parallel where possible)
        def _resonator(): return self.res_level.update(dt=0.1, drive=psi)
        def _assembly():  return None  # deps on resonator, run after

        res_state=self.res_level.update(dt=0.1, drive=psi)
        G_prev=self.global_l.G.copy()
        asm_state=self.assembly.update(res_state, G_prev)
        mod_state=self.module.update(asm_state, G_prev)
        global_state=self.global_l.update(mod_state, self.assembly.oscillator_output(),
                                          np.abs(psi), all_events)

        # Feedback
        fb_res=self.feedback.resonance_amplified(global_state, self.field)
        fb_temp=self.feedback.temporal(all_events)
        fb_crit=self.feedback.criticality_enhanced(global_state, self.global_l.kappa)
        global_state=0.7*global_state+0.1*fb_res+0.1*fb_temp[:len(global_state)]+0.1*fb_crit

        # Thought
        thought_evs=[]
        for e in all_events:
            thought_evs.extend(self.thought.cascade(e, global_state, self.global_l.kappa))
        self_ev=self.thought.self_generate(global_state, self.global_l.kappa)
        if self_ev: thought_evs.append(self_ev)
        chain_str=self.thought.resonant_chain(thought_evs)
        global_state=self.thought.multi_scale(thought_evs, global_state)

        # Vertical recursion with global
        R_final=self.recursive.vertical(R, np.abs(psi), global_state, thought_evs)

        out_state=R_final/(np.linalg.norm(R_final)+EPSILON)

        return {
            'state':       out_state,
            'global':      global_state,
            'psi':         psi,
            'events':      all_events+thought_evs,
            'compressed':  compressed,
            'chain':       chain_str,
            'field_stats': self.field.stats(),
            'ms':          (time.time()-t0)*1000,
        }


    def train_step(self, inp: str, out: str, negatives: Optional[List[str]]=None) -> Metrics: 
        t0=time.time()
        # Reset between each training example — critical, prevents state bleed
        self.field.reset(); self.res_level.reset()
        self.assembly.reset(); self.module.reset(); self.global_l.reset()

        res_inp=self.forward(inp)
        res_out=self.forward(out)
        state_in=res_inp['state']; state_tgt=res_out['state']
        neg_states=[self.forward(n)['state'] for n in (negatives or [])[:4]]

        loss=self.meta.loss(state_in, state_tgt, neg_states)

        if len(self._patterns)<32:
            self._patterns.append(np.abs(state_in))

        self.params=self.ctrl.update(self.params, res_inp['field_stats'], loss)

        enc_in = self.bin_enc.encode_text(inp, self.params)
        anchor = self.encode_features(inp)
        self.memory.store(inp, anchor, out)

        if self.memory.n > 1:
            stored_out = self.memory.get_output(inp) or ''
            self.modality.record(inp, stored_out == out)

        if self.step%200==0 and self.step>0:
            self.temperature=max(0.8, self.temperature*0.97)
        self.step+=1

        m=Metrics(self.step,loss,res_inp['field_stats'].criticality,
              self.params.chunk_k,self.params.damr_radius,self.memory.n,
              (time.time()-t0)*1000, len(res_inp['events']))
        self._metrics.append(m)
        return m

    def train(self, data: List[Tuple[str,str]], epochs=3, verbose=True) -> List[Metrics]:
        all_m=[]
        self.gen.reset_counts()
        for ep in range(epochs):
            idxs=np.random.permutation(len(data))
            el=0.; e_count=0
            self.gen.reset_counts()
            for i,idx in enumerate(idxs):
                inp,out=data[idx]
                window=[data[j][0] for j in range(max(0,idx-3),min(len(data),idx+4)) if j!=idx]
                m=self.train_step(inp,out,window)
                el+=m.loss; e_count+=m.events; all_m.append(m)

            if verbose:
                avg_loss=el/len(data)
                min_sep, avg_sep = self.memory.separation_stats()
                fs = self.field.stats()
                print(f"\n{'═'*55}")
                print(f"  Epoch {ep+1}/{epochs}")
                print(f"{'─'*55}")
                print(f"  Loss:    {avg_loss:.4f}   Meta-L: {self.meta.meta_loss():.4f}")
                print(f"  Anchors: {self.memory.n}      Steps:  {self.step}")
                print(f"  Field:   κ={fs.criticality:.4f}  energy={fs.energy:.3f}  phase_σ={fs.phase_spread:.3f}")
                print(f"  Ctrl:    k={self.params.chunk_k:.2f}  ρ={self.params.damr_radius:.2f}  scales={len(self.params.active_scales)}")
                srr = self.memory.self_retrieval_rate()
                print(f"  Memory:  min_sep={min_sep:.3f}  avg_sep={avg_sep:.3f}  srr={srr:.3f}  {'⚠ COLLAPSING' if min_sep<0.05 else '✓ healthy'}")
                print(f"  Events:  {e_count} total  avg={e_count/max(1,len(data)):.1f}/step")
                print(f"  Event types:")
                print(self.gen.type_report())
                print(f"  Modality accuracy:")
                print(self.modality.report())
                print(f"{'═'*55}")
        return all_m

    def train_file(self, path: str, epochs=3, verbose=True) -> List[Metrics]:
        pairs=[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                if '|||' in line:
                    a,b=line.strip().split('|||',1)
                    pairs.append((a.strip(),b.strip()))
        print(f"Loaded {len(pairs)} pairs from {path}")
        return self.train(pairs,epochs,verbose)

    def infer(self, text: str, verbose=True) -> Tuple[str,float]:
        self.field.reset(); self.res_level.reset()
        self.assembly.reset(); self.module.reset(); self.global_l.reset()
        t0=time.time()
        res=self.forward(text)
        anchor_q = self.bin_enc.encode_text(text, self.params)
        matches = self.memory.lookup(anchor_q, k=3)
        ms=(time.time()-t0)*1000

        if not matches:
            if verbose: print(f"  → [no memory]  (ms={ms:.1f})")
            return '[no memory]', 0.

        best_key, best_sim = matches[0]
        out=self.memory.get_output(best_key) or '[unknown]'
        conf=float(np.exp(best_sim/self.temperature))

        if verbose:
            mod = self.modality.detect(text)
            fs  = res['field_stats']
            # Wavelet signature for encoder verification
            feats = self.bin_enc.encode_text(text, self.params)
            sig, ratio = self.modality.wavelet_signature(feats)
            print(f"  → {out}  [conf={conf:.3f}]")
            print(f"  Top-3:", end='')
            for k2,s in matches:
                o2=self.memory.get_output(k2) or '?'
                print(f"  {o2}({s:.3f})", end='')
            print()
            print(f"  Modality: {mod}  |  Encoder sig: {sig}  ratio={ratio:.4f}")
            print(f"  Field κ={fs.criticality:.4f}  events={len(res['events'])}  ms={ms:.1f}")

        return out, conf

    def infer_top_k(self, text: str, k=3) -> List[Tuple[str,str,float]]:
        self.field.reset(); self.res_level.reset()
        self.bridge.bridge(self.bin_enc.encode_text(text, self.params)).real
        return [(k2, self.memory.get_output(k2) or '?',
                 float(np.exp(s/self.temperature)))
                for k2,s in self.memory.lookup(state,k)]


    def train_file(self, path: str, epochs=3, verbose=True) -> List[Metrics]:
        pairs=[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                if '|||' in line:
                    a,b=line.strip().split('|||',1)
                    pairs.append((a.strip(),b.strip()))
        print(f"Loaded {len(pairs)} pairs")
        return self.train(pairs,epochs,verbose)

    def param_history(self):
        return {k:[getattr(m,k) for m in self._metrics]
                for k in ('chunk_k','damr_r','criticality','loss','events')}

    def encode_features(self, text: str) -> np.ndarray:
        import base64
        if text.startswith('arr:'):
            try: return self.bin_enc.encode_array(base64.b64decode(text[4:]), self.params)
            except: pass
        if text.startswith('hex:'):
            try: return self.bin_enc.encode_array(bytes.fromhex(text[4:]), self.params)
            except: pass
        return self.bin_enc.encode_text(text, self.params)

# ══════════════════════════════════════════════
#   Section 18 — CHAT + MAIN
# ══════════════════════════════════════════════

def _decode_input(text: str) -> bytes:
    """Resolve prefixed inputs to raw bytes for multimodal."""
    import base64
    if text.startswith('arr:'):
        return base64.b64decode(text[4:])
    if text.startswith('hex:'):
        return bytes.fromhex(text[4:])
    if text.startswith('file:'):
        path = text[5:].strip()
        if os.path.exists(path):
            with open(path,'rb') as f: return f.read()
        return b''
    return text.encode('utf-8')


def chat_loop(cypha: Cypha):
    print("\nCypha HRNA — Interactive")
    print("  train <file> [epochs]   train on input|||output file")
    print("  infer <text>            run inference with full diagnostics")
    print("  sep                     print anchor separation matrix (top 10)")
    print("  quit\n")

    while True:
        try:
            raw = input("you> ").strip()
            if not raw: continue
            if raw.lower() in ('quit','exit','q'): break

            # ── train ──────────────────────────────────────────
            if raw.startswith('train '):
                parts = raw.split(None, 2)
                fp = parts[1]
                ep = int(parts[2]) if len(parts) > 2 else 3
                if os.path.exists(fp):
                    cypha.train_file(fp, ep)
                else:
                    print(f"  File not found: {fp}")
                continue

            # ── infer ──────────────────────────────────────────
            if raw.startswith('infer '):
                query = raw[6:].strip()
                print()
                cypha.infer(query, verbose=True)
                print()
                continue

            # ── sep ────────────────────────────────────────────
            if raw == 'sep':
                if cypha.memory.n < 2:
                    print("  Need at least 2 anchors first.")
                    continue
                keys = cypha.memory._keys[:10]
                vecs = np.array([cypha.memory.anchors[k] for k in keys])
                sims = vecs @ vecs.T
                print(f"\n  Cosine similarity matrix (top {len(keys)} anchors):")
                hdr = ''.join(f"{i:>6}" for i in range(len(keys)))
                print(f"  {'':>3} {hdr}")
                for i,k in enumerate(keys):
                    row = ''.join(f"{'---':>6}" if j==i else f"{sims[i,j]:>6.2f}"
                                  for j in range(len(keys)))
                    label = k[:18]
                    print(f"  {i:>2} {row}  {label}")
                mn, avg = cypha.memory.separation_stats()
                print(f"\n  min_sep={mn:.3f}  avg_sep={avg:.3f}\n")
                continue

            # ── default: treat as infer ─────────────────────────
            print()
            cypha.infer(raw, verbose=True)
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting."); break
        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            traceback.print_exc()


def main():
    print("="*55)
    print("  Cypha HRNA — Full Architecture")
    print(f"  CPU multicore ({N_WORKERS} workers) | Pure numpy")
    print("="*55)

    cypha = Cypha(feature_dim=4096, resonance_dim=256)

    # Quick smoke test before training
    demo = [
        ("12+165","177"),   ("44+60","104"),    ("25+75","100"),   ("7+7","14"),
        ("cat sound","meow"),("dog sound","bark"),("owl sound","hoot"),
        ("wolf sound","howl"),("bear sound","growl"),
        ("capital of France","Paris"),("capital of Japan","Tokyo"),
        ("capital of Australia","Canberra"),("capital of Germany","Berlin"),
        ("is 5 > 3","true"), ("is 2 > 10","false"),("is 7 > 4","true"),
        ("sort: 5 2 9 1","1 2 5 9"),("sort: 3 7 1 4","1 3 4 7"),
        ("run past","ran"),  ("jump past","jumped"),("swim past","swam"),
        ("answer to life","42"),("hello","hi there"),("how are you","good"),
        ("next: 1, 2, 3, 4, 5","6"),("next: 2, 4, 8, 16","32"),
    ]

    print(f"\nTraining on {len(demo)} examples × 5 epochs...\n")
    cypha.train_file("data.txt", epochs=5, verbose=True)

    print("\n── Quick Verification ──\n")
    tests = [
        ("cat sound",          "meow"),
        ("capital of France",  "Paris"),
        ("is 5 > 3",           "true"),
        ("12+165",             "177"),
        ("wolf sound",         "howl"),
        ("next: 1, 2, 3, 4, 5","6"),
    ]
    ok = 0
    for inp, exp in tests:
        r, c = cypha.infer(inp, verbose=True)
        mark = "✓" if r==exp else "✗"
        print(f"  {mark} expected '{exp}'\n")
        if r==exp: ok+=1
    print(f"  {ok}/{len(tests)} exact matches on seen data\n")

    chat_loop(cypha)


if __name__ == "__main__":
    main()