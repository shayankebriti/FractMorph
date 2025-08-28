import torch
import math
import torch.nn as nn

class FrFT3DModule(nn.Module):
    def __init__(self, order=0.5, log_output=False, device_option='cuda', use_cache=True, learnable_order=False):
        """
        Initializes the FrFT3DModule with specified parameters.
        
        Parameters:
        - order: The fractional Fourier transform order (default 0.5).
        - log_output: Whether to apply log transformation to the output (default False).
        - device_option: Specifies the computation device ('cuda', 'cpu', or 'volume'; default 'cuda').
        - use_cache: Whether to cache intermediate constant matrices/vectors (default True).
        - learnable_order: If True, 'order' is an nn.Parameter learned during training
        (caching is automatically disabled to avoid stale graphs). If False (default),
        'order' is stored as a non-trainable buffer and caching remains enabled.
        """
        super(FrFT3DModule, self).__init__()
        assert device_option in ('cuda', 'cpu', 'volume'), \
            "device_option must be 'cuda', 'cpu', or 'volume'"
        self.device_option = device_option
        if device_option == 'cuda':
            self.calc_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_option == 'cpu':
            self.calc_device = torch.device('cpu')
        else:
            self.calc_device = None

        self.learnable_order = learnable_order
        if learnable_order:
            self.order = nn.Parameter(torch.tensor(float(order), dtype=torch.float32))
        else:
            self.register_buffer('order', torch.tensor(float(order), dtype=torch.float32))

        self.log_output = log_output
        self.use_cache = use_cache and (not learnable_order)

        self._frft_mat_cache = {}
        self._transform_vec_cache = {}
        self._conv_matrix_cache = {}

    def _resolve_device(self, volume=None):
        if self.calc_device is not None:
            return self.calc_device
        if volume is not None:
            return volume.device
        return self.order.device

    def frft_mat(self, N, a_scalar, device):
        a_key = float(a_scalar)
        cache_key = (int(N), a_key, str(device))

        if self.use_cache and cache_key in self._frft_mat_cache:
            return self._frft_mat_cache[cache_key]

        with torch.no_grad():
            app_ord = 2
            Evec = self.transform_vec(N, app_ord).to(device).to(dtype=torch.complex64)
            even = 1 - (N % 2)
            l = torch.arange(0, N - 1, device=device, dtype=torch.float32)
            l = torch.cat([l, torch.tensor([N - 1 + even], device=device, dtype=torch.float32)], dim=0)
            phase = torch.exp(-1j * math.pi / 2.0 * a_key * l).to(torch.complex64)
            f = torch.diag(phase)  # [N,N] complex
            F = (N ** 0.5) * torch.einsum("ij,jk,ni->nk", f, Evec.T, Evec)
            F = F.to(device).to(torch.complex64).detach()

        if self.use_cache:
            self._frft_mat_cache[cache_key] = F
        return F

    def transform_vec(self, N, app_ord):
        cache_key = (int(N), int(app_ord))
        if self.use_cache and cache_key in self._transform_vec_cache:
            return self._transform_vec_cache[cache_key]

        dev = self._resolve_device() if hasattr(self, "_resolve_device") else None

        with torch.no_grad():
            if N < 3:
                out = torch.eye(N, dtype=torch.float32, device=dev)
                if self.use_cache:
                    self._transform_vec_cache[cache_key] = out
                return out

            app_ord = int(app_ord // 2)
            num_zeros = max(N - 1 - 2 * app_ord, 0)

            s = torch.cat((
                torch.tensor([0, 1], dtype=torch.float32, device=dev),
                torch.zeros(num_zeros, dtype=torch.float32, device=dev),
                torch.tensor([1], dtype=torch.float32, device=dev),
            ))
            if s.numel() != N:
                if s.numel() > N:
                    s = s[:N]
                else:
                    s = torch.cat((s, torch.zeros(N - s.numel(), dtype=torch.float32, device=dev)))

            S = self.conv_matrix(N, s)
            S = S + torch.diag(torch.fft.fft(s).real)

            p, r = N, math.floor(N / 2)
            P = torch.zeros((p, p), dtype=torch.float32, device=dev)
            P[0, 0] = 1.0

            sqrt2_inv = 1.0 / math.sqrt(2.0)
            even = 1 - (p % 2)

            row_idx = torch.arange(1, r - even + 1, device=dev)
            P[row_idx, row_idx] = sqrt2_inv
            P[row_idx, p - row_idx] = sqrt2_inv
            if even:
                P[r, r] = 1.0

            row_idx_odd = torch.arange(r + 1, p, device=dev)
            P[row_idx_odd, row_idx_odd] = -sqrt2_inv
            P[row_idx_odd, p - row_idx_odd] = sqrt2_inv

            CS = torch.einsum("ij,jk,ni->nk", S, P.T, P)

            n2_floor_plus1 = math.floor(N / 2 + 1)
            C2 = CS[:n2_floor_plus1, :n2_floor_plus1]
            S2 = CS[n2_floor_plus1:N, n2_floor_plus1:N]

            ec, vc = torch.linalg.eig(C2); ec, vc = ec.real, vc.real
            es, vs = torch.linalg.eig(S2); es, vs = es.real, vs.real

            qvc = torch.vstack((
                vc,
                torch.zeros((math.ceil(N / 2 - 1), math.floor(N / 2 + 1)), dtype=torch.float32, device=dev),
            ))
            SC2 = P @ qvc

            qvs = torch.vstack((
                torch.zeros((math.floor(N / 2 + 1), math.ceil(N / 2 - 1)), dtype=torch.float32, device=dev),
                vs,
            ))
            SS2 = P @ qvs

            idx_c = torch.argsort(-ec)
            idx_s = torch.argsort(-es)
            SC2 = SC2[:, idx_c]
            SS2 = SS2[:, idx_s]

            if N % 2 == 0:
                S2C2 = torch.zeros((N, N + 1), dtype=torch.float32, device=dev)
                SS2 = torch.hstack((SS2, torch.zeros((SS2.shape[0], 1), dtype=torch.float32, device=dev)))
                S2C2[:, list(range(0, N + 1, 2))] = SC2
                S2C2[:, list(range(1, N, 2))] = SS2
                S2C2 = S2C2[:, torch.arange(S2C2.size(1), device=dev) != N - 1]
            else:
                S2C2 = torch.zeros((N, N), dtype=torch.float32, device=dev)
                S2C2[:, list(range(0, N + 1, 2))] = SC2
                S2C2[:, list(range(1, N, 2))] = SS2

            S2C2 = S2C2.detach()

        if self.use_cache:
            self._transform_vec_cache[cache_key] = S2C2
        return S2C2

    def conv_matrix(self, N, s):
        cache_key = (int(N), tuple(s.detach().cpu().tolist()))
        if self.use_cache and cache_key in self._conv_matrix_cache:
            return self._conv_matrix_cache[cache_key]

        with torch.no_grad():
            M = torch.stack([torch.roll(s, shifts=i) for i in range(N)], dim=1).detach()

        if self.use_cache:
            self._conv_matrix_cache[cache_key] = M
        return M

    def FrFT3D(self, volume):
        """
        Applies the forward 3D fractional Fourier transform (FrFT).
        
        Parameters:
        - volume: torch.Tensor
            Input tensor of shape (B, C, D, H, W).
        
        Returns:
        - torch.Tensor
            - If log_output=False: complex output of shape (B, C, D, H, W).
            - If log_output=True: real tensor of shape (B, 2*C, D, H, W), where the
              first C channels are log1p(|out|) and the next C channels are the phase
              angle of out.
        """
        device = self._resolve_device(volume)
        self.order.data = self.order.data.to(device)

        B, C, D, H, W = volume.shape
        d_test = self.frft_mat(D, self.order, device)
        h_test = self.frft_mat(H, self.order, device)
        w_test = self.frft_mat(W, self.order, device)

        vol = volume.to(device).to(dtype=torch.complex64)
        vol = torch.fft.fftshift(vol, dim=(2,3,4))
        out_d = torch.einsum('ij,bcjhw->bcihw', d_test, vol)
        out_h = torch.einsum('ij,bcdjw->bcd iw', h_test, out_d)
        out_w = torch.einsum('ij,bcdhi->bcdhj', w_test, out_h)
        out = torch.fft.fftshift(out_w, dim=(2,3,4))

        if self.log_output:
            mag   = torch.log1p(torch.abs(out))
            phase = torch.angle(out)
            return torch.cat([mag, phase], dim=1)

        return out

    def IFrFT3D(self, volume):
        """
        Applies the inverse 3D fractional Fourier transform (IFrFT).
        
        Parameters:
        - volume: torch.Tensor
            Input tensor of shape (B, C, D, H, W).
        
        Returns:
        - torch.Tensor
            Reconstructed tensor of shape (B, C, D, H, W).
        """
        device = self._resolve_device(volume)
        self.order.data = self.order.data.to(device)

        B, C, D, H, W = volume.shape
        d_test = self.frft_mat(D, -self.order, device)
        h_test = self.frft_mat(H, -self.order, device)
        w_test = self.frft_mat(W, -self.order, device)

        vol = volume.to(device)
        vol = torch.fft.fftshift(vol, dim=(2,3,4))
        out_d = torch.einsum('ij,bcjhw->bcihw', d_test, vol)
        out_h = torch.einsum('ij,bcdjw->bcd iw', h_test, out_d)
        out_w = torch.einsum('ij,bcdhi->bcdhj', w_test, out_h)
        out = torch.fft.fftshift(out_w, dim=(2,3,4))
        return out
