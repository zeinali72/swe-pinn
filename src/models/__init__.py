"""Neural network architectures for PINN and DeepONet models."""
from src.models.layers import (
    Normalize, FourierFeatures, NTKDense, DGMLayer, apply_output_scaling,
)
from src.models.pinn import FourierPINN, MLP, DGMNetwork
from src.models.deeponet import DeepONet, FourierDeepONet
from src.models.ntk import NTK_MLP, FourierNTK_MLP
from src.models.factory import init_model, init_deeponet_model
