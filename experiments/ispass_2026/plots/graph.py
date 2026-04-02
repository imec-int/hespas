#!/usr/bin/env python
# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

import sys
import csv
import math
import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

imec_colours = [
    "#99BDE4",
    "#2D6C85",
    "#929497",
    "#52BDC1",
    "#3F98BD",
    "#3E98BD",
    "#1582BD",
    "#36337C",
    "#3C3C3B",
]
cur_imec_colour = 0
hatches = [
    '/',
    '\\',
    '+',
    'x',
    'o',
    '*',
    'O',
    '.',
    '-',
    '|',
]
cur_hatch = 0
renames_tuples = (
("reference", "Ground Truth"),
("profiling", "Profiling"),
("cocossim", "COCOSSim"),
("roofline", "Analytical"),
("ONNXim", "ONNXim"),
)
renames = dict(renames_tuples)
seen_sources = []
sources_colours = {}
sources_hatches = {}
for _, rename in renames_tuples:
    sources_colours[rename] = imec_colours[cur_imec_colour]
    cur_imec_colour += 1
    sources_hatches[rename] = hatches[cur_hatch]
    cur_hatch += 1
seen_workloads = []
plots = {}
def source_sort(t):
    s = t[0]
    pos = 0
    for i in renames_tuples:
        if i[1] in s:
            break
        pos += 10
    return pos

with open(sys.argv[1], newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        system = row["system"].strip().replace("_4GPU", "")
        # Normalize system names (e.g. A100_SXM_40GB → A100_40GB)
        system = re.sub(r'_SXM(?=_)', '', system)
        source = renames[row["source"].strip()]
        workload = row["workload"].strip()
        # Normalize workload names: strip -opt suffix and device/mode suffixes
        workload = re.sub(r'-opt$', '', workload)
        workload = re.sub(r'_(train|inference)(_step)?_\d+_devices$', '', workload)
        workload = re.sub(r'_(train|inference)$', '', workload)
        runtime = row["runtime_s"].strip()
        if system not in plots:
            plots[system] = {}
        if workload not in plots[system]:
            plots[system][workload] = {}
        if workload not in seen_workloads:
            seen_workloads.append(workload)
        if source not in plots[system][workload]:
            plots[system][workload][source] = []
        if source not in seen_sources:
            seen_sources.append(source)
        plots[system][workload][source].append(runtime)

# Average duplicate entries that collapsed due to normalization
for system in plots:
    for workload in plots[system]:
        for source in plots[system][workload]:
            vals = plots[system][workload][source]
            if len(vals) > 1:
                avg = sum(float(v) for v in vals) / len(vals)
                plots[system][workload][source] = [str(avg)]

width = 0.25
between_spacing = .1
x_poses = []
printed_legend = False

base_fontsize = 23 - 2 * len(plots)
plt.rcParams.update({'font.size': base_fontsize})
plt.rcParams.update({
    "font.family": "serif",
})
fig = plt.figure()
gs = fig.add_gridspec(len(plots), hspace=0)
#fig, axs = plt.subplots(len(plots), 1, layout='constrained')
axs = gs.subplots(sharex=True)
width_adjust = 1
height_adjust = 1
top_left_legend = False
legend_rows = 1
log_y = False
if len(sys.argv) > 2:
    width_adjust = float(sys.argv[2])
if len(sys.argv) > 3:
    height_adjust = float(sys.argv[3])
if len(sys.argv) > 4:
    top_left_legend = True if sys.argv[4].lower() == "true" else False
    legend_rows = 2 if top_left_legend is True else legend_rows
if len(sys.argv) > 5:
    legend_rows = int(sys.argv[5])
if len(sys.argv) > 6:
    log_y = True if sys.argv[6].lower() == "true" else False
if len(sys.argv) > 7:
    between_spacing = float(sys.argv[7])
plot_width = (18.5-2*len(plots))*width_adjust
plot_height = (6.5+len(plots))*height_adjust
fig.set_size_inches(plot_width, plot_height)
for i in range(0, len(seen_workloads)):
    x_poses.append(i * len(seen_sources) * (width + between_spacing))

abs_error = True
legend_handles = []
legend_labels = []
line_label = "Error {}%".format("(abs) " if abs_error else "")
line_colour = "#90288D"
line_marker = "o"
line_style = "None"
marker_size = base_fontsize - 4
reference_name = renames["reference"]
dot_padding = .15 * ((len(plots)/plot_height)*2.5)
first_ax = None
for figno, (system, workload_dict) in enumerate(plots.items()):
    if len(plots) > 1:
        ax = axs[figno]
    else:
        ax = axs
    if first_ax is None:
        first_ax = ax
    ax2 = ax.twinx()
    mult = 0
    workloads = tuple(workload_dict.keys())

    colours = {}
    first = True
    added_line = False
    fig_sources = []
    line_x_poses = []
    line_y_vals = []
    ignored_poses = []
    y_max = 0
    y_min = 10000000
    y2_max = 0
    for x_pos_i, (workload, source_dict) in enumerate(workload_dict.items()):
        x_pos = x_poses[x_pos_i] - ((len(source_dict) * width) * .5) + width * .5 + (len(seen_sources) - len(source_dict)) * width * .5
        runtimes = []
        offset = 0
        reference_value = float(source_dict[reference_name][0]) if reference_name in source_dict else None
        for source, runtime in sorted(source_dict.items(), key=source_sort):
            if source not in fig_sources:
                fig_sources.append(source)
            if isinstance(runtime, list) and len(runtime) == 1:
                runtime = runtime[0]
            else:
                raise ValueError("how did you mess up lists again")
            runtime = float(runtime) if len(runtime) > 0 else 0
            if reference_value and reference_value > 0:
                reference_error = ((runtime/reference_value) - 1) * 100
            else:
                reference_error = 0
            y_max = max(runtime, y_max)
            y2_max = max(reference_error, y2_max)
            y_min = min(runtime, y_min) if runtime > 0 else y_min
            if abs_error:
                reference_error = abs(reference_error)
            this_x_pos = x_pos + offset
            line_x_poses.append(this_x_pos)
            line_y_vals.append(reference_error)
            if source == reference_name or runtime <= 0 or reference_value is None:
                ignored_poses.append(this_x_pos)
            rect = ax.bar(this_x_pos, runtime, width, label=source if source not in colours else "", color=sources_colours.get(source), edgecolor="black", hatch=sources_hatches.get(source))
            if first is True:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            offset += width
        first = False

    num_ticks = 5
    ax_new_ticks = []

    if log_y is False:
        ax_tick_max = y_max + y_max * .1
        ax.set_ylim(0, ax_tick_max)
        ax_inc = ax_tick_max / num_ticks
        ax_new_ticks = [x * ax_inc for x in range(1, num_ticks)]
    else:
        ax.set_yscale('log')
        ax_tick_max_pow = int(round(math.log(y_max)/math.log(10))) + 1
        ax_min_pow = int(round(math.log(y_min)/math.log(10)))
        ax.set_ylim(10 ** (ax_min_pow-1), 10 ** ax_tick_max_pow)
        ax_new_ticks = [10 ** x for x in range(ax_min_pow, ax_tick_max_pow)]

    ax2_tick_max = 100 if y2_max < 100 else (round((y2_max * 1.5) / 50) * 50)
    ax2.set_ylim(0, ax2_tick_max)
    actual_line_x_poses = []
    actual_line_y_vals = []
    for x, y in zip(line_x_poses, line_y_vals):
        if x in ignored_poses:
            continue
        ax2.text(x, y+(ax2_tick_max*dot_padding), str(int(y)), ha="center", bbox=dict(boxstyle="round,pad=0.1", ec="black", fc="white",))
        actual_line_x_poses.append(x)
        actual_line_y_vals.append(y)
    line = ax2.plot(actual_line_x_poses, actual_line_y_vals, markersize=marker_size, markerfacecolor=line_colour, markeredgewidth=1, markeredgecolor="black", color=line_colour, marker=line_marker, label=line_label, linestyle=line_style)
    if added_line is False:
        line_legend_handles, line_legend_labels = ax2.get_legend_handles_labels()
        legend_handles.append(mlines.Line2D([], [], markersize=marker_size, markerfacecolor=line_colour, color=line_colour, markeredgewidth=1, markeredgecolor="black", marker=line_marker, label=line_label, linestyle=line_style))
        legend_labels.append(line_label)
        added_line = True
    ax.set_ylabel(system + "\ntime (s)", fontweight='bold', size=base_fontsize+2)
    ax2.set_ylabel(line_label, fontweight='bold', size=base_fontsize+2)
    ax2_inc = ax2_tick_max / num_ticks
    ax2_new_ticks = [x * ax2_inc for x in range(1, num_ticks)]
    if log_y is False:
        ax.set_yticks(ax_new_ticks, labels=[round(x, 2) for x in ax_new_ticks])
    else:
        ax.set_yticks(ax_new_ticks, labels=[x for x in ax_new_ticks])
    ax2.set_yticks(ax2_new_ticks, labels=[round(x, 2) for x in ax2_new_ticks])
    if figno == len(plots)-1:
        ax.set_xticks(x_poses, labels=workloads, rotation=15, size=base_fontsize+2)
    else:
        ax.set_xticks(x_poses, labels=[""] * len(x_poses), size=base_fontsize+2)

fig.align_labels()
if not top_left_legend:
    legend_y = 1.07 + ((.94-1.07)/(10.5-3.75)) * (plot_height-3.75) + (legend_rows-1) * .12
    fig.legend(legend_handles, legend_labels, loc='upper center', ncols=len(legend_labels) // legend_rows, bbox_to_anchor=(0.5, legend_y))
else:
    ax.legend(legend_handles, legend_labels, loc='upper left', ncols=len(legend_labels) // legend_rows)
plt.subplots_adjust(hspace=0)

plt.savefig("{}.svg".format(sys.argv[1].replace(".csv","")), bbox_inches='tight', dpi=100)

