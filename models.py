"""Custom feature extractors and policy classes."""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import pennylane as qml
except ImportError:  # pragma: no cover - only hit when quantum deps are absent
    qml = None
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _split_feature_dims(total_dim: int, n_parts: int) -> list[int]:
    if total_dim <= 0:
        raise ValueError('Invalid argument.')
    if n_parts <= 0:
        raise ValueError('Invalid argument.')
    base = total_dim // n_parts
    remainder = total_dim % n_parts
    dims = [base + (1 if i >= n_parts - remainder and remainder > 0 else 0) for i in range(n_parts)]
    if any(dim <= 0 for dim in dims):
        raise ValueError('Invalid argument.')
    return dims


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError('Invalid argument.')
    return math.log(math.expm1(value))


class MlpBottleneckExtractor(BaseFeaturesExtractor):
    """MLP bottleneck feature extractor."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, bottleneck_dim: int = 7):
        self.obs_dim = int(observation_space.shape[0])
        self.bottleneck_dim = int(bottleneck_dim)
        super().__init__(observation_space, features_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.obs_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.trunk(observations)


class LinearBottleneckExtractor(BaseFeaturesExtractor):
    """
    Parameter-matched linear bottleneck:
        obs_dim -> bottleneck_dim -> features_dim
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, bottleneck_dim: int = 7):
        self.obs_dim = int(observation_space.shape[0])
        self.bottleneck_dim = int(bottleneck_dim)
        super().__init__(observation_space, features_dim)
        self.encoder = nn.Linear(self.obs_dim, self.bottleneck_dim)
        self.out = nn.Linear(self.bottleneck_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.out(self.encoder(observations))


class TanhBottleneckExtractor(BaseFeaturesExtractor):
    """
    Tanh bottleneck:
        obs_dim -> bottleneck_dim -> tanh -> features_dim
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, bottleneck_dim: int = 7):
        self.obs_dim = int(observation_space.shape[0])
        self.bottleneck_dim = int(bottleneck_dim)
        super().__init__(observation_space, features_dim)
        self.encoder = nn.Linear(self.obs_dim, self.bottleneck_dim)
        self.out = nn.Linear(self.bottleneck_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.encoder(observations)))


class ClippedLatentBottleneckExtractor(BaseFeaturesExtractor):
    """
    Clipped latent bottleneck:
        obs_dim -> bottleneck_dim -> clip[-1, 1] -> features_dim
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, bottleneck_dim: int = 7):
        self.obs_dim = int(observation_space.shape[0])
        self.bottleneck_dim = int(bottleneck_dim)
        super().__init__(observation_space, features_dim)
        self.encoder = nn.Linear(self.obs_dim, self.bottleneck_dim)
        self.out = nn.Linear(self.bottleneck_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.out(torch.clamp(self.encoder(observations), min=-1.0, max=1.0))


class LayerNormBottleneckExtractor(BaseFeaturesExtractor):
    """
    LayerNorm bottleneck:
        obs_dim -> bottleneck_dim -> LayerNorm -> features_dim
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, bottleneck_dim: int = 7):
        self.obs_dim = int(observation_space.shape[0])
        self.bottleneck_dim = int(bottleneck_dim)
        super().__init__(observation_space, features_dim)
        self.encoder = nn.Linear(self.obs_dim, self.bottleneck_dim)
        self.norm = nn.LayerNorm(self.bottleneck_dim)
        self.out = nn.Linear(self.bottleneck_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.encoder(observations)))


class SpectralNormBottleneckExtractor(BaseFeaturesExtractor):
    """
    Spectral-normalized bottleneck encoder.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, bottleneck_dim: int = 7):
        self.obs_dim = int(observation_space.shape[0])
        self.bottleneck_dim = int(bottleneck_dim)
        super().__init__(observation_space, features_dim)
        self.encoder = nn.utils.spectral_norm(nn.Linear(self.obs_dim, self.bottleneck_dim))
        self.out = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, features_dim))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.out(F.relu(self.encoder(observations)))


class FourierFeatureExtractor(BaseFeaturesExtractor):
    """Fourier feature extractor."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, n_freq: int = 7):
        self.obs_dim = int(observation_space.shape[0])
        self.n_freq = int(n_freq)
        super().__init__(observation_space, features_dim)
        self.freq_proj = nn.Linear(self.obs_dim, self.n_freq)
        self.out = nn.Linear(self.n_freq * 2, features_dim)
        self.activation = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        phase = self.freq_proj(observations)
        features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
        return self.activation(self.out(features))


class WaveletExtractor(BaseFeaturesExtractor):
    """Haar wavelet feature extractor."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
        init_threshold: float = 0.1,
    ):
        self.obs_dim = int(observation_space.shape[0])
        self.padded_dim = self.obs_dim if self.obs_dim % 2 == 0 else self.obs_dim + 1
        self.coeff_dim = self.padded_dim // 2
        super().__init__(observation_space, features_dim)

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.low = nn.Conv1d(1, 1, kernel_size=2, stride=2, bias=False)
        self.high = nn.Conv1d(1, 1, kernel_size=2, stride=2, bias=False)
        with torch.no_grad():
            self.low.weight.copy_(torch.tensor([[[inv_sqrt2, inv_sqrt2]]], dtype=torch.float32))
            self.high.weight.copy_(torch.tensor([[[inv_sqrt2, -inv_sqrt2]]], dtype=torch.float32))
        self.low.weight.requires_grad_(False)
        self.high.weight.requires_grad_(False)
        self.lam = nn.Parameter(torch.tensor(float(init_threshold), dtype=torch.float32))
        self.out = nn.Linear(self.padded_dim, features_dim)

    def _soft_threshold(self, coeffs: torch.Tensor) -> torch.Tensor:
        threshold = F.softplus(self.lam)
        return torch.sign(coeffs) * F.relu(torch.abs(coeffs) - threshold)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.unsqueeze(1)
        if self.obs_dim % 2 != 0:
            x = F.pad(x, (0, 1))
        c_a = self.low(x)
        c_d = self.high(x)
        denoised_detail = self._soft_threshold(c_d)
        coeffs = torch.cat((c_a.flatten(start_dim=1), denoised_detail.flatten(start_dim=1)), dim=1)
        return F.relu(self.out(coeffs))


class KalmanExtractor(BaseFeaturesExtractor):
    """Diagonal Kalman-style feature extractor."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
        init_q: float = 1.0,
        init_r: float = 1.0,
    ):
        self.obs_dim = int(observation_space.shape[0])
        super().__init__(observation_space, features_dim)

        left_dim, right_dim = _split_feature_dims(features_dim, 2)
        self.log_q = nn.Parameter(torch.full((self.obs_dim,), _inverse_softplus(float(init_q)), dtype=torch.float32))
        self.log_r = nn.Parameter(torch.full((self.obs_dim,), _inverse_softplus(float(init_r)), dtype=torch.float32))
        self.f_proj = nn.Linear(self.obs_dim, left_dim)
        self.i_proj = nn.Linear(self.obs_dim, right_dim)
        self.out = nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        q = F.softplus(self.log_q)
        r = F.softplus(self.log_r)
        gain = q / (q + r + 1e-8)
        filtered = gain * observations
        innovation = (1.0 - gain) * observations
        hidden = torch.cat(
            (
                self.f_proj(filtered),
                self.i_proj(innovation),
            ),
            dim=1,
        )
        hidden = F.relu(hidden)
        return F.relu(self.out(hidden))


class QuantumLayer(nn.Module):
    """Variational PQC layer used by the FPQC-SAC feature bottleneck.

    The layer maps a real-valued vector to qubit rotations, applies trainable
    variational rotations, and returns Pauli-Z expectation values. By Born's
    rule, each expectation is the difference between the probabilities of
    measuring ``|0>`` and ``|1>`` on that wire, so the output is bounded in
    ``[-1, 1]`` and can be consumed by the downstream SAC MLP.

    Args:
        n_qubits: Number of wires in the circuit and output expectation values.
        n_layers: Number of variational rotation/entanglement layers.
        device: Optional PyTorch device used for the quantum submodule.
        use_entanglement: If false, disables CNOT gates for the No-CNOT
            ablation and keeps only local single-qubit rotations.
        entanglement_topology: ``"ring"`` uses PennyLane's ring-style
            ``BasicEntanglerLayers``; ``"line"`` applies nearest-neighbor CNOTs
            without connecting the last qubit back to the first.
        embedding_type: ``"angle"`` uses angle embedding; ``"amplitude"`` uses
            normalized amplitude embedding.
        freeze_pqc_params: If true, freezes trainable PQC rotations for the
            Frozen-PQC ablation while still training surrounding neural layers.

    Input:
        Tensor of shape ``(batch_size, n_qubits)`` for angle embedding, or
        ``(batch_size, 2 ** n_qubits)`` for amplitude embedding.

    Output:
        Tensor of shape ``(batch_size, n_qubits)`` containing Pauli-Z
        expectation values.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: Optional[torch.device] = None,
        use_entanglement: bool = True,
        entanglement_topology: str = "ring",
        embedding_type: str = "angle",
        freeze_pqc_params: bool = False,
    ):
        super().__init__()
        if qml is None:
            raise ImportError('Optional dependency is unavailable.')
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_entanglement = bool(use_entanglement)
        self.entanglement_topology = str(entanglement_topology).lower()
        self.embedding_type = str(embedding_type).lower()
        self.freeze_pqc_params = bool(freeze_pqc_params)
        if self.entanglement_topology not in {"ring", "line"}:
            raise ValueError('Invalid argument.')
        if self.embedding_type not in {"angle", "amplitude"}:
            raise ValueError('Invalid argument.')

        if not self.use_entanglement and self.embedding_type == "angle":
            self.weights = nn.Parameter(torch.empty(self.n_layers, self.n_qubits))
            nn.init.uniform_(self.weights, a=0.0, b=2.0 * math.pi)
            if self.freeze_pqc_params:
                self.weights.requires_grad_(False)
            if device is not None:
                self.to(device)
            return

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            if self.embedding_type == "angle":
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            else:
                qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            if not self.use_entanglement:
                # No-CNOT ablation: local rotations only, no inter-qubit
                # entangling operation. This isolates the effect of CNOTs.
                for layer_weights in weights:
                    for wire, angle in enumerate(layer_weights):
                        qml.RX(angle, wires=wire)
            elif self.entanglement_topology == "ring":
                # Ring topology entangles neighboring qubits and connects the
                # last wire back to the first, matching the main FPQC setting.
                qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            else:
                # Line topology keeps nearest-neighbor CNOTs but removes the
                # wraparound CNOT, separating topology from parameter count.
                for layer_weights in weights:
                    for wire, angle in enumerate(layer_weights):
                        qml.RX(angle, wires=wire)
                    for wire in range(self.n_qubits - 1):
                        qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        weight_shapes = {"weights": (self.n_layers, self.n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        if self.freeze_pqc_params:
            for param in self.qlayer.parameters():
                param.requires_grad_(False)

        if device is not None:
            self.to(device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.use_entanglement and self.embedding_type == "angle":
            # AngleEmbedding uses RX by default. Without CNOTs, all RX rotations commute,
            # so <Z> after all layers is exactly cos(input_angle + sum(layer_angles)).
            return torch.cos(h + self.weights.sum(dim=0))
        return self.qlayer(h)


class QuantumFeatureExtractor(BaseFeaturesExtractor):
    """Parameterized quantum-circuit feature extractor."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
        n_qubits: int = 4,
        n_layers: int = 2,
        quantum_device: str = "cpu",
        use_entanglement: bool = True,
        entanglement_topology: str = "ring",
        embedding_type: str = "angle",
        freeze_pqc_params: bool = False,
    ):
        
        self.obs_dim = int(observation_space.shape[0])
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.quantum_device = quantum_device
        self.use_entanglement = bool(use_entanglement)
        self.entanglement_topology = str(entanglement_topology).lower()
        self.embedding_type = str(embedding_type).lower()
        self.freeze_pqc_params = bool(freeze_pqc_params)
        if self.embedding_type not in {"angle", "amplitude"}:
            raise ValueError('Invalid argument.')
        super().__init__(observation_space, features_dim)

        quantum_input_dim = n_qubits if self.embedding_type == "angle" else 2 ** n_qubits
        self.pre_net = nn.Sequential(
            nn.Linear(self.obs_dim, quantum_input_dim),
            nn.ReLU(),
        )
        self.quantum = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=torch.device(quantum_device),
            use_entanglement=self.use_entanglement,
            entanglement_topology=self.entanglement_topology,
            embedding_type=self.embedding_type,
            freeze_pqc_params=self.freeze_pqc_params,
        )

        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        h = self.pre_net(x)
        original_device = h.device
        h_cpu = h.to(torch.device(self.quantum_device))
        x_quantum = self.quantum(h_cpu)
        x = x_quantum.to(original_device)
        x = self.post_net(x)
        return x


class QuantumActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy using the quantum feature extractor."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        features_dim: int = 64,
        n_qubits: int = 4,
        n_layers: int = 2,
        **kwargs,
    ):
        quantum_device = kwargs.pop("quantum_device", "cpu")
        enable_post_net = kwargs.pop("enable_post_net", True)
        policy_kwargs: Dict = kwargs.pop("policy_kwargs", {})
        quantum_device = policy_kwargs.pop("quantum_device", quantum_device)
        enable_post_net = policy_kwargs.pop("enable_post_net", enable_post_net)
        n_layers = policy_kwargs.pop("n_layers", n_layers)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.pop("features_extractor_class", None)
        if "features_extractor_class" not in policy_kwargs:
            policy_kwargs["features_extractor_class"] = QuantumFeatureExtractor
        if "features_extractor_kwargs" not in policy_kwargs:
            policy_kwargs["features_extractor_kwargs"] = {
                "features_dim": features_dim,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "quantum_device": quantum_device,
            }
        else:
            if "quantum_device" not in policy_kwargs["features_extractor_kwargs"]:
                policy_kwargs["features_extractor_kwargs"]["quantum_device"] = quantum_device
            if "n_layers" not in policy_kwargs["features_extractor_kwargs"]:
                policy_kwargs["features_extractor_kwargs"]["n_layers"] = n_layers

        fe_kw = policy_kwargs.get("features_extractor_kwargs")
        if isinstance(fe_kw, dict):
            fe_kw.pop("enable_post_net", None)

        for key in ("enable_post_net", "policy_kwargs", "n_qubits", "n_layers", "quantum_device"):
            kwargs.pop(key, None)
            policy_kwargs.pop(key, None)

        net_arch = kwargs.get("net_arch", dict(pi=[64, 64], vf=[64, 64]))
        if isinstance(net_arch, list) and len(net_arch) == 1 and isinstance(net_arch[0], dict):
            net_arch = net_arch[0]
        kwargs["net_arch"] = net_arch

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args,
            **kwargs,
            **policy_kwargs,
        )


