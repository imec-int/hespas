# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
"Entrypoint for the simulator"

import logging

from .chakra_gen.chakra_trace_generator import main

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
