"""Taylor-model bounds on batches of boxes, in PyTorch (taylor.py) and CUDA (cuda.py), and a branch and bound over
boxes (search.py) for proving properties with them. See README.md in the repository."""
from .search import branch_and_bound, split
from .taylor import Dual, Settings, TaylorModel, evaluate, jacobian, network

__all__ = ["TaylorModel", "Settings", "Dual", "evaluate", "jacobian", "network", "branch_and_bound", "split"]
__version__ = "0.1.0"
