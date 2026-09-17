import torch

class ResonatorLevel:
    def __init__(self, n=100, dimension=1, diffusion=0.1, q=0.5, 
                 gamma_inhib=0.2, locality_radius=3, device='cpu'):
        self.n = n
        self.dimension = dimension
        self.device = device
        self.diffusion = diffusion
        self.q = q
        self.gamma_inhib = gamma_inhib
        self.locality_radius = locality_radius
        self.R = torch.zeros(n, device=device)
        self.omega = torch.linspace(1.0, 10.0, n, device=device)
        self.local_weights = torch.randn(2 * locality_radius + 1, device=device) * 0.3
        self.local_weights[locality_radius] = 0

    def local_coupling(self, R):
        coupling = torch.zeros_like(R)
        for offset in range(-self.locality_radius, self.locality_radius + 1):
            if offset == 0:
                continue
            weight_idx = offset + self.locality_radius
            weight = self.local_weights[weight_idx]
            if offset > 0:
                coupling[:-offset] += weight * torch.sigmoid(R[offset:])
            else:
                coupling[-offset:] += weight * torch.sigmoid(R[:offset])
        return coupling

    def update(self, dt=0.1, events=None, external_drive=None):
        freq_term = self.omega * self.R
        connect_term = self.local_coupling(self.R)
        diff_term = torch.zeros_like(self.R)
        diff_term[1:-1] = self.diffusion * (self.R[:-2] - 2 * self.R[1:-1] + self.R[2:])
        quality_term = self.q * self.R * torch.sigmoid(self.R)
        total_activity = torch.sum(torch.abs(self.R))
        inhibition = -self.gamma_inhib * total_activity / self.R.numel()
        drive_term = 0.0
        if external_drive is not None:
            drive_term = external_drive[:self.n] * 200.0  # Stronger drive
        R_new = self.R + dt * (freq_term + connect_term + diff_term + quality_term + inhibition) + drive_term
        abs_R = torch.abs(R_new)
        threshold = torch.quantile(abs_R, 0.8)
        mask = abs_R < threshold
        R_new[mask] *= 0.1
        self.R = R_new
        max_val = torch.max(torch.abs(self.R))
        if max_val > 10:
            self.R = 10 * self.R / max_val
        return self.R

class AssemblyLevel:
    def __init__(self, n=20, nr=100, sparsity=0.15, device='cpu'):
        self.n = n
        self.nr = nr
        self.device = device
        self.sparsity = sparsity
        self.A = torch.zeros(n, device=device)
        
        # Identity backbone: each assembly strongly connected to 2-3 specific resonators
        V_dense = torch.zeros(n, nr, device=device)
        connections_per_assembly = nr // n  # 64/32 = 2
        for i in range(n):
            # Primary connections for this assembly
            for j in range(connections_per_assembly):
                idx = (i * connections_per_assembly + j) % nr
                V_dense[i, idx] = 5.0  # Strong primary connection
            # Add random noise everywhere
            V_dense[i] += torch.randn(nr, device=device) * 0.5

        k_res = max(1, int(nr * sparsity))
        V_topk = torch.topk(V_dense.abs(), k_res, dim=1)
        self.V_indices = V_topk.indices
        self.V_values = torch.gather(V_dense, 1, self.V_indices)
        
        C_dense = torch.ones(n, n, device=device) * 0.3
        k_lat = max(1, int(n * 0.3))
        C_topk = torch.topk(C_dense.abs(), k_lat, dim=1)
        self.C_indices = C_topk.indices
        self.C_values = torch.gather(C_dense, 1, self.C_indices)

    def sparse_matmul(self, indices, values, vector):
        result = torch.zeros(indices.shape[0], device=self.device)
        vector_sig = torch.sigmoid(vector)
        for i in range(indices.shape[0]):
            result[i] = (values[i] * vector_sig[indices[i]]).sum()
        return result

    def update(self, R, G=None, dt=0.1, events=None):
        inherent_term = -0.01 * self.A + 0.1 * torch.sigmoid(self.A)
        resonator_term = self.sparse_matmul(self.V_indices, self.V_values, R)
        comp_term = -self.sparse_matmul(self.C_indices, self.C_values, self.A)
        top_term = 0.1 * G if G is not None and G.shape == self.A.shape else torch.zeros_like(self.A)
        self.A += dt * (inherent_term + resonator_term + comp_term + top_term)
        return self.A

class ModuleLevel:
    def __init__(self, n=5, na=20, sparsity=0.2, device='cpu'):
        self.n = n
        self.na = na
        self.device = device
        self.sparsity = sparsity
        self.M = torch.zeros(n, device=device)
        
        # Structured territories for modules too
        conn_dense = torch.zeros(n, na, device=device)
        connections_per_module = na // n  # 20/16 = 1
        for i in range(n):
            for j in range(connections_per_module + 1):  # +1 for better coverage
                idx = (i * connections_per_module + j) % na
                conn_dense[i, idx] = 5.0
            conn_dense[i] += torch.randn(na, device=device) * 0.5

        k_conn = max(1, int(na * sparsity))
        conn_topk = torch.topk(conn_dense.abs(), k_conn, dim=1)
        self.conn_indices = conn_topk.indices
        self.conn_values = torch.gather(conn_dense, 1, self.conn_indices)
        
        inhib_dense = torch.ones(n, n, device=device) * 0.4
        k_inhib = max(1, int(n * 0.4))
        inhib_topk = torch.topk(inhib_dense.abs(), k_inhib, dim=1)
        self.inhib_indices = inhib_topk.indices
        self.inhib_values = torch.gather(inhib_dense, 1, self.inhib_indices)

    def sparse_matmul(self, indices, values, vector):
        result = torch.zeros(indices.shape[0], device=self.device)
        vector_sig = torch.sigmoid(vector)
        for i in range(indices.shape[0]):
            result[i] = (values[i] * vector_sig[indices[i]]).sum()
        return result

    def update(self, A, G=None, dt=0.1, events=None):
        decay = -0.1 * self.M
        intdyn = torch.sigmoid(self.M) * (1.0 - self.M)
        assembly_term = self.sparse_matmul(self.conn_indices, self.conn_values, A)
        inhibit_term = -self.sparse_matmul(self.inhib_indices, self.inhib_values, self.M)
        top_term = 0.1 * G if G is not None and G.shape == self.M.shape else torch.zeros_like(self.M)
        self.M += dt * (decay + intdyn + assembly_term + inhibit_term + top_term)
        return self.M

class GlobalLevel:
    def __init__(self, d=5, nm=5, alpha=0.1, device='cpu'):
        self.d = d
        self.nm = nm
        self.alpha = alpha
        self.device = device
        self.G = torch.zeros(d, device=device)
        self.W = torch.randn(d, nm, device=device) * 0.1
        self.crit = 0.5
    
    def update(self, M, inp=None, dt=0.1, events=None):
        decay = -self.alpha * 0.1 * self.G
        bottomup_term = torch.matmul(self.W, M) if M.shape[0] == self.W.shape[1] else torch.zeros_like(self.G)
        ext_term = 0.2 * inp if inp is not None and inp.shape == self.G.shape else torch.zeros_like(self.G)
        self.G += dt * (decay + bottomup_term + ext_term)
        self.G = torch.tanh(self.G)  # ← Commented out
        return self.G
