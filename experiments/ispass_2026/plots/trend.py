#!/usr/bin/env python
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

import os
import csv
import pprint
import statistics

systems = {}

file = os.path.join(os.path.dirname(__file__), "llama3_train.csv")
with open(file, newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        system = row["system"]
        source = row["source"]
        workload = row["workload"]
        runtime = float(row["runtime_s"])

        if system not in systems:
            systems[system] = {}
        if source not in systems[system]:
            systems[system][source] = {}
        systems[system][source][workload] = runtime

reference_diffs = {}
other_diffs = {}
final_diffs = {}
for system, source_dict in systems.items():
    if system not in reference_diffs:
        reference_diffs[system] = {}
    if system not in other_diffs:
        other_diffs[system] = {}
    if system not in final_diffs:
        final_diffs[system] = {}

    workload_tuples = [(k, v) for k, v in source_dict["reference"].items()]
    for i in range(1, len(workload_tuples)):
        reference_diffs[system]["{}-{}".format(workload_tuples[i][0], workload_tuples[i-1][0])] = (workload_tuples[i][1] / workload_tuples[i-1][1])
    for source in [x for x in source_dict.keys() if x != "reference"]:
        if source not in other_diffs[system]:
            other_diffs[system][source] = {}
        workload_tuples = [(k, v) for k, v in source_dict[source].items()]
        for i in range(1, len(workload_tuples)):
            diff_name = "{}-{}".format(workload_tuples[i][0], workload_tuples[i-1][0])
            other_diffs[system][source][diff_name] = abs(((workload_tuples[i][1] / workload_tuples[i-1][1]) / reference_diffs[system][diff_name]) - 1)
        final_diffs[system][source] = round(100 * statistics.median(other_diffs[system][source].values()), 3)

pprint.pprint(final_diffs)

