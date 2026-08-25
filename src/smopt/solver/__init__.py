"""Solvers for optimization over the Stiefel manifold."""

from .pencf import PenCF
from .slpg import SLPG, SLPG_l21, SLPG_smooth


__all__: list[str] = ["SLPG", "PenCF", "SLPG_l21", "SLPG_smooth"]
