# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import os
from src.hespas.chakra_gen.chakra_trace_generator import create_chakra_traces

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def test_resnet50_roofline_a100():
    create_chakra_traces(f"{CONFIG_DIR}/configs/config_roofline_a100_resnet50.json")

