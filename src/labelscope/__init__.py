"""labelscope — a diagnostic scope for Vesuvius Challenge surface-label datasets.

The Vesuvius Challenge Open Problems post names label quality as one of the main
unwrapping bottlenecks: labels "may wiggle, may drift slightly off the true surface,
may avoid the most ambiguous regions".  labelscope turns each of those three
sentences into a number you can compute on a dataset you already have.
"""

__version__ = "0.1.0"

from labelscope.io import VolumePair, probe_volume, read_volume  # noqa: F401
