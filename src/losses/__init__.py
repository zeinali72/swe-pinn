"""Loss functions for physics-informed training."""
from src.losses.pde import compute_pde_loss, compute_neg_h_loss, compute_ic_loss
from src.losses.boundary import (
    loss_boundary_dirichlet_h,
    loss_boundary_dirichlet_hu,
    loss_boundary_dirichlet_hv,
    loss_slip_wall_generalized,
    loss_boundary_wall_vertical,
    loss_boundary_wall_horizontal,
    loss_boundary_neumann_outflow_x,
)
from src.losses.data_loss import compute_data_loss
from src.losses.composite import compute_bc_loss, compute_building_bc_loss, total_loss
