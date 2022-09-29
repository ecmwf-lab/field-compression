import os

from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Pkg

Pkg.activate(os.path.dirname(os.path.dirname(__file__)))

from julia import BitInformation

__all__ = ["BitInformation"]
