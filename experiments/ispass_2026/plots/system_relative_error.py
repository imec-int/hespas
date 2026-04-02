#!/usr/bin/env python
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

import os
import csv
import pprint
import statistics

workloads = {}
sources = []

file = os.path.join(os.path.dirname(__file__), "llama3_train.csv")
with open(file, newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        system = row["system"]
        source = row["source"]
        workload = row["workload"]
        runtime = float(row["runtime_s"])

        if workload not in workloads:
            workloads[workload] = {}
        if source not in workloads[workload]:
            workloads[workload][source] = {}
        workloads[workload][source][system] = runtime
        if source not in sources and source != "reference":
            sources.append(source)

reference_diffs = {}
per_change_diffs = {}
per_workload_diffs = {}
overall_diffs = {}
for workload, source_dict in workloads.items():
    if workload not in reference_diffs:
        reference_diffs[workload] = {}
    if workload not in per_change_diffs:
        per_change_diffs[workload] = {}
    if workload not in per_workload_diffs:
        per_workload_diffs[workload] = {}

    system_tuples = [(k, v) for k, v in source_dict["reference"].items()]
    for from_tuple in system_tuples:
        for to_tuple in system_tuples:
            if from_tuple == to_tuple:
                continue
            reference_diffs[workload]["{}-{}".format(to_tuple[0], from_tuple[0])] = (to_tuple[1] / from_tuple[1])
    for source in [x for x in source_dict.keys() if x != "reference"]:
        if source not in per_change_diffs[workload]:
            per_change_diffs[workload][source] = {}
        system_tuples = [(k, v) for k, v in source_dict[source].items()]
        for from_tuple in system_tuples:
            for to_tuple in system_tuples:
                if from_tuple == to_tuple:
                    continue
                diff_name = "{}-{}".format(to_tuple[0], from_tuple[0])
                per_change_diffs[workload][source][diff_name] = abs(((to_tuple[1] / from_tuple[1]) / reference_diffs[workload][diff_name]) - 1)
        per_workload_diffs[workload][source] = round(100 * statistics.median(per_change_diffs[workload][source].values()), 3)
for source in sources:
    overall_diffs[source] = round(statistics.median([x[source] for x in [x for x in per_workload_diffs.values()]]), 3)

print("All changes diffs:")
pprint.pprint(per_change_diffs)
print("\nPer-workload diffs:")
pprint.pprint(per_workload_diffs)
print("\nOverall diffs:")
pprint.pprint(overall_diffs)

