import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import OrderedDict
from os.path import dirname
import math
from colour import Color
import numpy as np
from torch_geometric.utils import to_networkx
import os

def set_node_attr(g, types):
    node_attr = dict()
    if not types is None:
        for i in range(g.num_nodes):
            node_attr[i] = dict()
            nlabel = int(np.where(g.x[i].numpy() != 0)[0])
            ntype = types[nlabel]
            node_attr[i]['label'] = nlabel
            node_attr[i]['type'] = ntype
    else:
        for i in range(g.num_nodes):
            node_attr[i] = dict()
            node_attr[i]['label'] = None
            node_attr[i]['type'] = None
    return node_attr




def vis_small(q=None, gs=None, info_dict=None, types = None):
    plt.figure(figsize=(8, 3))
    _info_dict_preprocess(info_dict)
    nx_q = to_networkx(q, to_undirected=True)
    
    nx.set_node_attributes(nx_q, set_node_attr(q, types))
    nx_gs = []
    for g in gs:
        nx_g = to_networkx(g, to_undirected=True)
        nx.set_node_attributes(nx_g, set_node_attr(g, types))
        nx_gs.append(nx_g)

    # get num
    graph_num = 1 + len(nx_gs)   
    plot_m, plot_n = _calc_subplot_size_small(graph_num)  

    # draw query graph
    # info_dict['each_graph_text_font_size'] = 9
    ax = plt.subplot(plot_m, plot_n, 1)

    draw_graph_small(nx_q, info_dict)

    draw_extra(0, ax, info_dict,
               _list_safe_get(info_dict['each_graph_text_list'], 0, ""))

    # draw graph candidates
    # info_dict['each_graph_text_font_size'] = 12
    for i in range(len(nx_gs)):
        ax = plt.subplot(plot_m, plot_n, i + 2)
        draw_graph_small(nx_gs[i], info_dict)
        draw_extra(i, ax, info_dict,
                   _list_safe_get(info_dict['each_graph_text_list'], i + 1, ""))

    # plot setting
    # plt.tight_layout()
    left = 0.01  # the left side of the subplots of the figure
    right = 0.99  # the right side of the subplots of the figure
    top = 1 - info_dict['top_space']  # the top of the subplots of the figure
    bottom = \
        info_dict['bottom_space']  # the bottom of the subplots of the figure
    wspace = \
        info_dict['wbetween_space']  # the amount of width reserved for blank space between subplots
    hspace = \
        info_dict['hbetween_space']  # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                        wspace=wspace, hspace=hspace)

    # save / display
    _save_figs(info_dict)



def _get_line_width(g):
    lw = 5.0 * np.exp(-0.05 * g.number_of_edges())
    return lw


def _get_edge_width(g, info_dict):
    ew = info_dict.get('edge_weight_default', 1.0)
    ew = ew * np.exp(-0.0015 * g.number_of_edges())
    return info_dict.get('edge_weights', [ew] * len(g.edges()))


def draw_graph_small(g, info_dict):
    if g is None:
        return
    if g.number_of_nodes() > 1000:
        print('Graph to plot too large with {} nodes! skip...'.format(
            g.number_of_nodes()))
        return
    pos = _sorted_dict(graphviz_layout(g))
    color_values = _get_node_colors(g, info_dict)
    node_labels = _sorted_dict(nx.get_node_attributes(g, info_dict['node_label_type']))
    for key, value in node_labels.items():
        # Labels are not shown, but if the ids want to be plotted, then they are shown.
        if not info_dict['show_labels']:
            node_labels[key] = ''
    # print(pos)
    nx.draw_networkx(g, pos, nodelist=pos.keys(),
                     node_color=color_values, with_labels=True,
                     node_size=_get_node_size(g, info_dict),
                     labels=node_labels,
                     font_size=info_dict['draw_node_label_font_size'],
                     linewidths=_get_line_width(g), width=_get_edge_width(g, info_dict))

    if info_dict['draw_edge_label_enable'] == True:
        edge_labels = nx.get_edge_attributes(g, info_dict['edge_label_name'])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_size=info_dict[
                                         'draw_edge_label_font_size'])



def _get_node_size(g, info_dict):
    ns = info_dict['draw_node_size']
    return ns * np.exp(-0.02 * g.number_of_nodes())

def _sorted_dict(d):
    rtn = OrderedDict()
    for k in sorted(d.keys()):
        rtn[k] = d[k]
    return rtn

def _get_node_colors(g, info_dict):
    if info_dict['node_label_name'] is not None:
        color_values = []
        node_color_labels = _sorted_dict(nx.get_node_attributes(g, info_dict['node_label_name']))
        for node_label in node_color_labels.values():
            color = info_dict['draw_node_color_map'].get(node_label, None)
            color_values.append(color)
    else:
        # color_values = ['lightskyblue'] * g.number_of_nodes()
        color_values = ['#CCCCCC'] * g.number_of_nodes()
    # print(color_values)
    return color_values


def _info_dict_preprocess(info_dict):
    info_dict.setdefault('draw_node_size', 10)
    info_dict.setdefault('draw_node_label_enable', True)
    info_dict.setdefault('node_label_name', '')
    info_dict.setdefault('draw_node_label_font_size', 6)

    info_dict.setdefault('draw_edge_label_enable', False)
    info_dict.setdefault('edge_label_name', '')
    info_dict.setdefault('draw_edge_label_font_size', 6)

    info_dict.setdefault('each_graph_text_font_size', "")
    info_dict.setdefault('each_graph_text_pos', [0.5, 0.8])

    info_dict.setdefault('plot_dpi', 200)
    info_dict.setdefault('plot_save_path', "")

    info_dict.setdefault('top_space', 0.08)
    info_dict.setdefault('bottom_space', 0)
    info_dict.setdefault('hbetween_space', 0.5)
    info_dict.setdefault('wbetween_space', 0.01)

def _calc_subplot_size_small(area):
    w = int(area)
    return [2, math.ceil(w / 2)]


def draw_extra(i, ax, info_dict, text):
    left = _list_safe_get(info_dict['each_graph_text_pos'], 0, 0.5)
    bottom = _list_safe_get(info_dict['each_graph_text_pos'], 1, 0.8)
    # print(left, bottom)
    ax.title.set_position([left, bottom])
    ax.set_title(text, fontsize=info_dict['each_graph_text_font_size'])
    plt.axis('off')

def _list_safe_get(l, index, default):
    try:
        return l[index]
    except IndexError:
        return default


def _save_figs(info_dict):
    save_path = info_dict['plot_save_path_pdf']
    print(save_path)
    if not save_path:
        # print('plt.show')
        plt.show()
    else:
        for full_path in [info_dict['plot_save_path_pdf']]:
            if not full_path:
                continue
            # print('Saving query vis plot to {}'.format(sp))
            if not os.path.exists(dirname(full_path)):
                os.makedirs(dirname(full_path))
            plt.savefig(full_path, dpi=info_dict['plot_dpi'])
    plt.close()

