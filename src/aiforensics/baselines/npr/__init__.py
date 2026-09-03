"""NPR external baseline package (Task 10).

The public adapter is importable without Torch, Torchvision, OpenCV, SciPy, or
the external NPR checkout; heavy dependencies stay lazy inside the runtime.
"""

from aiforensics.baselines.npr.adapter import NPRAdapter

__all__ = ["NPRAdapter"]
