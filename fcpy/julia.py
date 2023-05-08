# (C) Copyright 2022 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os

from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Pkg

Pkg.activate(os.path.dirname(os.path.dirname(__file__)))

from julia import BitInformation

__all__ = ["BitInformation"]
