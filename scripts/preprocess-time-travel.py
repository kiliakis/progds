#!/usr/bin/python
import argparse
import numpy as np
import pandas as pd
import sys
import os
import csv
# import yaml
# import re
# import bisect
from common import utils as ut

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
this_filename = sys.argv[0].split('/')[-1]


# Correlation plot with warp activity and total stalls.
parser = argparse.ArgumentParser(
    description='Evaluate the potential of all stocks.',
    usage='{} infile -o outfile'.format(this_filename))

parser.add_argument('-d', '--datadir', action='store', type=str,
                    help='The input stock data dir.')

parser.add_argument('-s', '--stocks', action='store', type=int,
                    help='Number of stocks to check.')

# parser.add_argument('-v', '--value', action='store', type=int,
#                     help='Min money value.')

parser.add_argument('-o', '--outdir', action='store', type=str,
                    default='./preprocess/', help='The directory to store the preprocess data.')

parser.add_argument('-m', '--methods', nargs='+', choices=['maxr', 'dp', 'subd'],
                    default=['maxr', 'dp', 'subd'], help='Potential heuristics to use, maxr: maxrange, dp: exhaustive, subd: subdollar')

# parser.add_argument('-yg', '--yamlglobal', type=str,
#                     default='{}/common/config.yml'.format(
#                         this_directory),
#                     help='The global yaml config file.')

# parser.add_argument('-s', '--show', action='store_true',
#                     help='Show the plots or save only.')
# parser.add_argument('-selected', '--selected-kernels', action='store_true',
#                     help='Use the selected kernels.')

if __name__ == '__main__':
    # read cmd line options
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # read yaml config file
    # globyc = yaml.load(open(args.yamlglobal, 'r'),
    #                    Loader=yaml.FullLoader)
    # locyc = yaml.load(open(args.yamllocal, 'r'), Loader=yaml.FullLoader)
    header = ['stock', 'date', 'value']
    for method in args.methods:
        stock_potential = ut.parse_stock_potential(
            args.datadir, method=method, numstocks=args.stocks)
        with open(os.path.join(args.outdir, '{}.csv'.format(method)), 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(header)
            writer.writerows(stock_potential)

    # print(stock_potential[:10])

    # # First I read all the input files
    # # Baseline data
    # retdic = read_and_format_input(locyc, globyc, vars(args))
    # basedic = retdic['basedic']
    # loogdic = retdic['loogdic']
    # all_kernels = retdic['all_kernels']

    # if args.selected_kernels:
    #     selected_kernels = globyc['selected_kernels']
    #     diff = set(selected_kernels) - set(all_kernels)
    #     if (len(diff) > 0):
    #         print('WARNING: These kernels were not found')
    #         print(diff)
    #     all_kernels = list(set(selected_kernels) & set(all_kernels))

    # smetric = locyc['sortmetric']
    # pmetric = locyc['plotmetric']

    # figdic = {}
    # # continue here, check that it works as expected
    # mixeddic = calc_mixed_metrics_with_scan(basedic, loogdic, locyc['mixedmetrics'],
    #                                         formulas = globyc['mixed_formulas'],
    #                                         constants = locyc['constants'],
    #                                         knobs_to_keep=locyc['knobs_to_keep'])

    # # vals = [list(m2.values())[0] for m in mixeddic.values()
    # #         for m2 in m.values()]
    # # kernels = [list(m.keys())[0] for m in mixeddic.values()]
    # for k in all_kernels:
    #     for m in basedic.keys():
    #         for knob in basedic[m][k].keys():
    #             if knob not in locyc['knobs_to_keep']:
    #                 continue
    #             name = locyc['knobs_to_keep'][knob]
    #             if name not in figdic:
    #                 figdic[name] = {'kernels':[]}

    #             if k not in figdic[name]['kernels']:
    #                 figdic[name]['kernels'].append(k)

    #             if m not in figdic[name]:
    #                 figdic[name][m] = []

    #             figdic[name][m].append(basedic[m][k][knob])
    #     for m in mixeddic.keys():
    #         for knob in mixeddic[m][k].keys():
    #             if knob not in locyc['knobs_to_keep']:
    #                 continue
    #             name = locyc['knobs_to_keep'][knob]

    #             if name not in figdic:
    #                 figdic[name] = {'kernels':[]}

    #             if k not in figdic[name]['kernels']:
    #                 figdic[name]['kernels'].append(k)

    #             if m not in figdic[name]:
    #                 figdic[name][m] = []

    #             figdic[name][m].append(mixeddic[m][k][knob])

    # # I need to process the data and divide it into a list with #percentiles elements,
    # # and each element #aggregate_pos elements, this means I need to first sort the
    # # kernels according to the normalized IPC
    # for knob in figdic.keys():
    #     idx = np.argsort(figdic[knob][smetric])
    #     nans = np.zeros(len(idx), dtype=bool)
    #     for m in figdic[knob].keys():
    #         figdic[knob][m] = np.array(figdic[knob][m])[idx]
    #         # if m != 'kernels':
    #         #     nans = np.logical_or(nans, np.isnan(figdic[knob][m]))
    #     for m in figdic[knob].keys():
    #         figdic[knob][m] = figdic[knob][m][~nans]
    #     print('Knob: {}, Kernels remained: {}'.format(
    #         knob, len(idx)-np.sum(nans)))

    # # sys.exit()
    # # Ok kernels have been sorted now, nans have been removed
    # # cmap = cm.get_cmap(figconf['percentiles']['colormap'])
    # # colors = [cmap(x) for x in np.linspace(0., 1., bins)]
    # for knob, vals in figdic.items():
    #     figconf = locyc['figure']
    #     fig, ax_arr = plt.subplots(**figconf['subplots'])

    #     # todo here use index
    #     ax = ax_arr
    #     plt.sca(ax)

    #     step = 1.0
    #     pos = 0
    #     width = 0.95*step
    #     # bins = figconf['percentiles']['number']
    #     xticks = [[], []]
    #     labels = set()
    #     for k in vals['kernels']:
    #         bottom = 0
    #         for m in locyc['plotmetric']:
    #             y = 100 * float(vals[m])
    #             label = None
    #             if m not in labels:
    #                 label = globyc['metric_shorts'].get(m, m)
    #                 labels.add(label)

    #             plt.bar(pos, y, width=width, edgecolor='black',
    #                 linewidth=1, color=locyc['colors'][m],
    #                 hatch=locyc['hatches'][m], label=label)

    #             if 'saved' not in m:
    #                 bottom += y

    #     # There should be only one knob
    #     # extract the right subset of the array
    #     # subIndexes = np.array_split(np.arange(len(vals[pmetric])), bins)
    #     # title = 'Kernels: {}'.format(len(vals[pmetric]))
    #     for i, subIndex in enumerate(subIndexes):
    #         y = np.mean(vals[pmetric][subIndex])
    #         x = np.mean(vals[smetric][subIndex])
    #         plt.bar(pos, y, width=width, edgecolor='0',
    #                 linewidth=1, color=colors[i],
    #                 hatch=None, label=None)
    #         xticks[0].append(pos)
    #         xticks[1].append(x)
    #         pos += step
    #     # two more, one with IPC <= 1 and one with IPC > 1
    #     plt.axvline(pos - 0.25*step, ymax=0.8,  ls='--', c='black')
    #     pos += 0.5 * step
    #     bless = 0
    #     bmore = 0
    #     idx_less = vals[smetric] <= 1
    #     idx_more = vals[smetric] > 1
    #     y = np.mean(vals[pmetric][idx_less])
    #     plt.bar(pos, y, width=width, bottom=bless,
    #             edgecolor='0', linewidth=1, color=colors[0],
    #             hatch=None, label=None)
    #     bless += y

    #     y = np.mean(vals[pmetric][idx_more])
    #     plt.bar(pos+step, y, width=width, bottom=bmore,
    #             edgecolor='0', linewidth=1, color=colors[-1],
    #             hatch=None, label=None)
    #     bmore += y

    #     xticks[0].append(pos)
    #     xticks[1].append(np.mean(vals[smetric][idx_less]))
    #     plt.text(pos, bless, '<1', fontsize=10, ha='center', va='bottom')

    #     xticks[0].append(pos+step)
    #     xticks[1].append(np.mean(vals[smetric][idx_more]))
    #     plt.text(pos+step, bmore, '>1', fontsize=10, ha='center', va='bottom')

    # plt.grid(axis='y', zorder=0, alpha=0.5)
    # # xticks = xtickspos.astype('str')
    # # xticks[-1] += '+'
    # xticks[1] = np.round(xticks[1], 2)

    # # xticks[0] = np.array(xticks[0])[np.linspace(
    # #     0, len(xticks[0])-1, 5, dtype=int)]
    # # xticks[1] = np.round(np.linspace(xticks[1][0], xticks[1][-1], 5), 2)
    # plt.xticks(xticks[0], xticks[1], **figconf['ticks']['x'])
    # # plt.xlim(xticks[0][0]+(pos-width)/2 - (pos+width)/2,
    # #          xticks[0][-1]+(pos-width)/2 + (pos+width)/2)
    # plt.ylabel(figconf['ylabel'], **figconf['label'])
    # plt.xlabel(figconf['xlabel'], **figconf['label'])
    # ax.tick_params(**figconf['tick_params'])
    # # plt.legend(**figconf['legend'])
    # plt.tight_layout()
    # plt.subplots_adjust(**figconf['subplots_adjust'])
    # plt.ylim(figconf['ylim'][perwarp])
    # # plt.yticks(figconf['yticks'], **figconf['ticks']['y'])
    # # plt.title(title, **figconf['title'])
    # plt.text(s=title, transform=ax.transAxes, **figconf['text'])

    # outfiles = ['{}/{}-{}-kernels{}_bins{}.jpeg'.format(args.outdir,
    #                                                     this_filename.replace('.py', ''), perwarp, len(all_kernels), bins),
    #             '{}/{}-{}-kernels{}_bins{}.pdf'.format(args.outdir,
    #                                                    this_filename.replace('.py', ''), perwarp, len(all_kernels),  bins)]
    # for outfile in outfiles:
    #     save_and_crop(fig, outfile, dpi=600, bbox_inches='tight')
    # if args.show:
    #     plt.show()
    # plt.close()
