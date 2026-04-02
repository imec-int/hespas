#!/usr/bin/env python
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

import os
import csv
import pprint
import statistics

sources = {}
workloads = []
avg_func = statistics.mean

file = os.path.join(os.path.dirname(__file__), "llama3_train.csv")
with open(file, newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        system = row["system"]
        source = row["source"]
        workload = row["workload"]
        runtime = float(row["runtime_s"])

        if source not in sources:
            sources[source] = {}
        if system not in sources[source]:
            sources[source][system] = {}
        if workload not in workloads:
            workloads.append(workload)
        sources[source][system][workload] = runtime

per_workload_speedups = {}
per_source_speedups = {}
speedup_names = []
for source, system_dict in sources.items():
    if source not in per_source_speedups:
        per_source_speedups[source] = {}
    if source not in per_workload_speedups:
        per_workload_speedups[source] = {}

    systems = [x for x in system_dict.keys()]
    for workload in workloads:
        if workload not in per_workload_speedups[source]:
            per_workload_speedups[source][workload] = {}
        for from_system in systems:
            for to_system in systems:
                if from_system == to_system:
                    continue
                speedup_name = "{}-{}".format(from_system, to_system)
                if speedup_name not in speedup_names:
                    speedup_names.append(speedup_name)
                per_workload_speedups[source][workload][speedup_name] = 1 / (sources[source][to_system][workload] / sources[source][from_system][workload])
    for speedup_name in speedup_names:
        per_source_speedups[source][speedup_name] = round(avg_func([x[speedup_name] for x in per_workload_speedups[source].values()]), 3)

pprint.pprint(per_workload_speedups)
pprint.pprint({k: {i: j for i, j in v.items() if j > 1} for k, v in per_source_speedups.items()})

