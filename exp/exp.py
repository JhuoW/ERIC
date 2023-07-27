from re import S
from statistics import mode
import numpy as np
from rich import print
from model.GSC import GSC
from argparse import ArgumentParser
from utils.utils import *
from utils.vis import vis_small
import seaborn as sb
import matplotlib
import matplotlib.colors as mcolors
from torch_geometric.data import Batch
from tqdm import tqdm
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from utils.color import *
matplotlib.use('Agg')
matplotlib.rc('font', **{'family': 'serif', 'size': 22})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
TRUE_MODEL = 'astar'


@torch.no_grad()
def evaluate(testing_graphs, training_graphs, model, dataset: DatasetLocal, config):
    model.eval()

    ground_truth                   = np.empty((len(testing_graphs), len(training_graphs)))
    ground_truth_ged               = np.empty((len(testing_graphs), len(training_graphs)))
    prediction_mat                 = np.empty((len(testing_graphs), len(training_graphs)))

    num_test_pairs                 = len(testing_graphs) * len(training_graphs)
    t                              = tqdm(total=num_test_pairs)

    for i,g in enumerate(testing_graphs):
        source_batch               = Batch.from_data_list([g] * len(training_graphs))
        target_batch               = Batch.from_data_list(training_graphs)
        data                       = dataset.transform_batch((source_batch, target_batch), config)
        target                     = data["target"]
        # target = data["norm_ged"]

        ground_truth[i]            = target
        target_ged                 = data["target_ged"]
        ground_truth_ged[i]        = target_ged
        prediction, loss_cl        = model(data)
        prediction_mat[i]          = prediction.cpu().detach().numpy()

        t.update(len(training_graphs))

    prediction_mat_unexp = np.log(prediction_mat) 
    return prediction_mat, prediction_mat_unexp


def get_true_result(all_graphs, testing_graphs, trainval_graphs, sim_or_dist = 'dist'):
    ged_matrix                                  = trainval_graphs.ged
    nged_matrix                                 = trainval_graphs.norm_ged
    ground_truth_ged                            = np.empty((len(testing_graphs), len(trainval_graphs)))   # test graph 和 train-val graph 之间的距离
    ground_truth_nged                           = np.empty((len(testing_graphs), len(trainval_graphs))) 
    for test_id, test_g in  enumerate(testing_graphs):
        for tv_id, tv_g in  enumerate(trainval_graphs):
            ground_truth_ged[test_id][tv_id]    = ged_matrix[test_g['i'], tv_g['i']]
            ground_truth_nged[test_id][tv_id]   = nged_matrix[test_g['i'], tv_g['i']]
    print(ground_truth_ged.shape)
    sort_id_mat                                 = np.argsort(ground_truth_ged,  kind = 'mergesort')
    sort_id_mat_normed                          = np.argsort(ground_truth_nged, kind = 'mergesort')
    if sim_or_dist == 'sim':
        sort_id_mat                             = sort_id_mat[:, ::-1]
        sort_id_mat_normed                      = sort_id_mat_normed[:, ::-1]
    return ground_truth_ged, ground_truth_nged, sort_id_mat, sort_id_mat_normed

def get_pred_result(pred_mat, sim_or_dist = 'sim'):
    # pred_mat 为相似度矩阵   (140, 560)  = sim_mat
    sort_id_mat = np.argsort(pred_mat, kind='mergesort')[:, ::-1]   # 每行按照相似度从大到小排序
    return sort_id_mat


def copy_unnormalized_sim_mat(sim_mat, testing_graphs, trainval_graphs):
    rtn = np.copy(sim_mat)
    m, n = sim_mat.shape
    for i in range(m):
        for j in range(n):
            rtn[i][j] = unnormalized_dist_sim(rtn[i][j], testing_graphs[i], trainval_graphs[j])
    return rtn

def unnormalized_dist_sim(d, g1, g2):
    g1_size = g1.num_nodes
    g2_size = g2.num_nodes

    return torch.log(d) * (g1_size + g2_size) / 2

def draw_emb_hist_heat(args, testing_graphs, trainval_graphs, model: GSC, extra_dir=None, plot_max_num=np.inf):
    extra_dir = osp.join(args.extra_dir, args.dataset, 'mne')
    all_graphs = testing_graphs + trainval_graphs
    emb_layers_dicts = model.collect_embeddings(all_graphs)
    ground_truth_ged , ground_truth_nged, sort_id_mat, sort_id_mat_normed = get_true_result(all_graphs, testing_graphs, trainval_graphs)
    colors = ['Oranges', 'Blues', 'Greens']
    for gnn_id in sorted(emb_layers_dicts.keys()):
        nel = emb_layers_dicts[gnn_id]
        cmap_color = colors[gnn_id % len(colors)]
        draw_emb_hist_heat_helper(gnn_id, nel, cmap_color, args.dataset, testing_graphs, trainval_graphs, sort_id_mat_normed, ground_truth_nged, True, plot_max_num, extra_dir)


def draw_emb_hist_heat_helper(gnn_id, nel, cmap_color, dataset_name,
                              test_graphs, trainval_graphs, ids, ground_truth_nged, ds_norm,
                              plot_max_num, extra_dir):

    plt_cnt = 0
    for i, tv in tqdm(enumerate(test_graphs)):
        sort = [1, 10, 50, 100, len(trainval_graphs)]
        gids = [int(ids[i][:sort[0]]), int(ids[i][sort[1]]), int(ids[i][sort[2]]), int(ids[i][sort[3]]) ,int(ids[i][-1:])]
        for s, j in enumerate(gids):
            d = ground_truth_nged[i][j]
            query_nel_idx = int(tv['i'])                   
            match_nel_idx = int(j)        
            result = cosine_similarity(nel[int(query_nel_idx)].cpu().detach().numpy(),  nel[int(match_nel_idx)].cpu().detach().numpy())
            if s == 0:
                vmax = result.max()
            plt.figure()
            sb_plot = sb.heatmap(result, fmt='d', cmap=cmap_color, vmax = vmax)
            fig = sb_plot.get_figure()
            dir = '{}/{}'.format(extra_dir, 'heatmap')
            fn  = '{}_{}_{}_sort{}_gcn{}'.format(i, j, d, sort[s],gnn_id)
            plt_cnt += save_fig(fig, dir, fn, print_path=False)
            if extra_dir:
                plt_cnt += save_fig(fig, extra_dir + '/heatmap', fn, print_path=False)
            plt.close()
            result_array = []
            for m in range(len(result)):  
                for n in range(len(result[m])):
                    result_array.append(result[m][n])
            plt.figure()
            sb_plot = sb.distplot(result_array, bins=16, color='r',
                                                kde=False, rug=False, hist=True, )
            sb_plot.set(xlabel='Similarity', ylabel='Num of Nodes')
            fig = sb_plot.get_figure()
            dir = '{}/{}'.format(extra_dir, 'histogram')
            fn = '{}_{}_{}_gcn{}'.format(i, j, d, gnn_id)                                    
            plt_cnt += save_fig(fig, dir, fn, print_path=False)
            if extra_dir:
                plt_cnt += save_fig(fig, extra_dir + '/histogram', fn, print_path=False)
            plt.close()
        if plt_cnt > plot_max_num:
            print('Saved {} node embeddings mne plots for gcn{}'.format(plt_cnt, gnn_id))
            return
    print('Saved {} node embeddings mne plots for gcn{}'.format(plt_cnt, gnn_id))            


def visualize_embeddings_gradual(args, testing_graphs, trainval_graphs, model: GSC, plot_max_num=np.inf,  concise=False , sort = True, perplexity = None):
    extra_dir = osp.join(args.extra_dir, args.dataset, 'emb_vis_gradual')
    all_graphs = testing_graphs + trainval_graphs
    graph_emb_layers = model.collect_graph_embeddings(all_graphs)
    ground_truth_ged , ground_truth_nged, sort_id_mat, sort_id_mat_normed = get_true_result(all_graphs, testing_graphs, trainval_graphs)
    for gnn_id in graph_emb_layers.keys():
        if gnn_id == 3:
            print(gnn_id)
        if sort:
            tsne = TSNE(n_components=2, perplexity=60)  # , method='barnes_hut'
        else:
            if perplexity is None:
                tsne = TSNE(n_components=2)
            else:
                tsne = TSNE(n_components=2, perplexity=perplexity)
        g_emb_gnn_i = graph_emb_layers[gnn_id]
        vec = []
        ids = []
        for i, v in g_emb_gnn_i.items():   
            vec.append(v.cpu().detach().numpy())
            ids.append(i)
        vec = np.array(vec)
        embs = tsne.fit_transform(vec)
        plt_cnt = 0
        if not concise:
            print('TSNE embeddings: {} --> {} to plot'.format(
                vec.shape, embs.shape))
        # plot_embeddings_for_gs_by_glabel(embs, testing_graphs, trainval_graphs, extra_dir)        
        plot_embeddings_for_each_query(embs, ids, plt_cnt, ground_truth_nged, sort_id_mat_normed,extra_dir,
                                    plot_max_num, gnn_id ,trainval_graphs, testing_graphs,concise, sort, perplexity)


def plot_embeddings_for_each_query(embs, ids, plt_cnt,ground_truth_nged, sort_id_mat_normed, extra_dir,
                                   plot_max_num, gnn_id, trainval_graphs, testing_graphs, concise, sort, perplexity):
    emb_dict = dict()
    for k in range(embs.shape[0]):
        emb_dict[ids[k]] = embs[k]
    m = np.shape(sort_id_mat_normed)[0]
    n = np.shape(sort_id_mat_normed)[1]
    if sort:
        plot_what = 'emb_vis_query'
    else:
        if perplexity is None:
            plot_what = 'emb_vis_query_unsort'
        else:
            plot_what = 'emb_vis_query_unsort_{}'.format(perplexity)
    
    for i in tqdm(range(m)):  
        axis_x_red = []
        axis_y_red = []
        axis_x_blue = []
        axis_y_blue = []
        axis_x_query = []
        axis_y_query = []
        red_number = []
        blue_number = []
        for j in range(n):   
            if ranking(i, j, sort_id_mat_normed, ground_truth_nged) < n / 2:  
                red_number.append((j, ground_truth_nged[i][j]))   
            else:
                blue_number.append((j, ground_truth_nged[i][j]))
        sorted(red_number, key=lambda x: x[1])
        sorted(blue_number, key=lambda x: x[1], reverse=True)
        for j in range(len(red_number)):  
            axis_x_red.append(emb_dict[int(trainval_graphs[red_number[j][0]]['i'])][0])
            axis_y_red.append(emb_dict[int(trainval_graphs[red_number[j][0]]['i'])][1])
        for j in range(len(blue_number)):
            axis_x_blue.append(emb_dict[int(trainval_graphs[blue_number[j][0]]['i'])][0])
            axis_y_blue.append(emb_dict[int(trainval_graphs[blue_number[j][0]]['i'])][1])
        axis_x_query.append(emb_dict[int(testing_graphs[i]['i'])][0])
        axis_y_query.append(emb_dict[int(testing_graphs[i]['i'])][1])
        # Plot.
        plt.figure()
        if sort:
            plt.scatter(axis_x_blue, axis_y_blue, s=20,
                        c=sorted(range(len(axis_x_blue)), reverse=False),
                        marker='s', cmap=plt.cm.get_cmap('Blues'))
            plt.scatter(axis_x_red, axis_y_red, s=20,
                        c=sorted(range(len(axis_x_red)), reverse=False),
                        marker='o', cmap=plt.cm.get_cmap('Reds'))
            plt.scatter(axis_x_query, axis_y_query, s=400, c='green',
                        marker='P', alpha=0.8)
        else:
            plt.scatter(axis_x_blue, axis_y_blue, s=20,
                        color = 'blue',
                        marker='s')
            plt.scatter(axis_x_red, axis_y_red, s=20,
                        c='red',
                        marker='o')
            plt.scatter(axis_x_query, axis_y_query, s=400, c='green',
                        marker='P')
        plt.axis('off')
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        fn = str(i)
        plt_cnt += save_fig(plt, '{}/{}/gnn_{}'.format(extra_dir, plot_what, gnn_id), fn)
        # if extra_dir:
        #     plt_cnt += save_fig(plt, '{}/{}/gnn_{}/query_'.format(extra_dir, plot_what, gnn_id), fn)
        plt.close()
        if plt_cnt > plot_max_num:
            if not concise:
                print('Saved {} embedding visualization plots'.format(plt_cnt))
            return
    if not concise:
        print('Saved {} embedding visualization plots'.format(plt_cnt))


def ranking(qid, gid, sort_id_mat_normed, ground_truth_nged, one_based=True):

    # Assume self is ground truth.
    sort_id_mat = sort_id_mat_normed

    finds = np.where(sort_id_mat[qid] == gid)  
    assert (len(finds) == 1 and len(finds[0]) == 1)
    fid = finds[0][0]   
    dist_sim_mat = ground_truth_nged
    while fid > 0:
        cid = sort_id_mat[qid][fid]      
        pid = sort_id_mat[qid][fid - 1]  
        if dist_sim_mat[qid][pid] == dist_sim_mat[qid][cid]: 
            fid -= 1
        else:
            break
    if one_based:
        fid += 1
    return fid

def plot_embeddings_for_gs_by_glabel(embs, test_gs, train_gs, extra_dir):
    points = []
    for i, g in enumerate(train_gs + test_gs):
        emb = embs[i]  
        assert (emb.shape == (2,))
        points.append({'x': emb[0], 'y': emb[1], 'glabel': g.graph['glabel']})
    plot_embeddings_for_points(points, 'all_gs', dir, extra_dir)
    plot_embeddings_for_points(points[0:len(train_gs)], 'train_gs', dir, extra_dir)
    plot_embeddings_for_points(points[len(train_gs):], 'test_gs', dir, extra_dir)


def plot_embeddings_for_points(points, what_gs,extra_dir):
    plot_what = 'emb_vis_glabel'
    markers = ('D', 's', 'P', 'v', '^', '*', '<', '>', 'p', 'o')
    colors = ('red', 'blue', 'green')
    cmaps = plt.cm.get_cmap('tab20')
    plt.figure()
    for p in points:
        x = p['x']
        y = p['y']
        glabel = p['glabel']
        # glabel = np.random.randint(1, 16)
        assert (type(glabel) is int and glabel >= 0)
        if glabel < 3:
            c = colors[glabel]
        else:
            c = cmaps(glabel)
        m = markers[glabel % len(markers)]
        plt.scatter(x, y, c=c, marker=m)
    plt.axis('off')
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    fn = what_gs
    save_fig(plt, '{}/{}'.format(dir, plot_what), fn)
    if extra_dir:
        save_fig(plt, '{}/{}'.format(extra_dir, plot_what), fn)
    plt.close()





def classification_mat(thresh_pos, thresh_neg,ground_truth_nged, norm = True):  
    snorm = ('norm' if norm else 'unnorm')
    vname = 'classif_mat_{}_{}_{}'.format(thresh_pos, thresh_neg, snorm)
    dist_mat = ground_truth_nged
    label_mat, num_poses, num_negs, _, _ = get_classification_labels_from_dist_mat(dist_mat, thresh_pos, thresh_neg)
    rtn = (label_mat, num_poses, num_negs)
    return rtn

def visualize_embeddings_binary(args, testing_graphs, trainval_graphs, model: GSC, norm = True, perplexity = None):
    extra_dir = osp.join(args.extra_dir, args.dataset, 'emb_vis_binary')
    if args.dataset == 'AIDS700nef' or args.dataset == 'LINUX' or args.dataset == 'IMDBMulti':
        thresh_pos = 0.95
        thresh_neg = 0.95
        thresh_pos_sim = 0.5
        thresh_neg_sim = 0.5
    all_graphs = testing_graphs + trainval_graphs
    graph_emb_layers = model.collect_graph_embeddings(all_graphs)
    ground_truth_ged , ground_truth_nged, sort_id_mat, sort_id_mat_normed = get_true_result(all_graphs, testing_graphs, trainval_graphs)
    label_mat, _, _ = classification_mat(thresh_pos, thresh_neg, ground_truth_nged, norm)
    for gnn_id in graph_emb_layers.keys():
        if gnn_id == 3:
            print(gnn_id)

        if perplexity is None:
            tsne = TSNE(n_components=2)
        else:
            tsne = TSNE(n_components=2, perplexity=perplexity)
        g_emb_gnn_i = graph_emb_layers[gnn_id]
        vec = []
        ids = []
        for i, v in g_emb_gnn_i.items():   
            vec.append(v.cpu().detach().numpy())
            ids.append(i)
        vec = np.array(vec)
        embs = tsne.fit_transform(vec)
        create_dir_if_not_exists(extra_dir)
        m = np.shape(label_mat)[0]
        n = np.shape(label_mat)[1]
        plt_cnt = 0
        print('TSNE embeddings: {} --> {} to plot'.format(
            vec.shape, embs.shape))


        if perplexity is None:
            plot_what = 'emb_vis_binary'
        else:
            plot_what = 'emb_vis_binary_{}'.format(perplexity)
    
        emb_dict = dict()
        for k in range(embs.shape[0]):
            emb_dict[ids[k]] = embs[k]

        for j in range(m):
            axis_x_red = []
            axis_y_red = []
            axis_x_blue = []
            axis_y_blue = []
            axis_x_query = []
            axis_y_query = []
            for i in range(n):
                if label_mat[j][i] == -1:
                    axis_x_blue.append(emb_dict[int(trainval_graphs[i]['i'])][0])
                    axis_y_blue.append(emb_dict[int(trainval_graphs[i]['i'])][1])
                else:
                    axis_x_red.append(emb_dict[int(trainval_graphs[i]['i'])][0])
                    axis_y_red.append(emb_dict[int(trainval_graphs[i]['i'])][1])
            axis_x_query.append(emb_dict[int(testing_graphs[j]['i'])][0])
            axis_y_query.append(emb_dict[int(testing_graphs[j]['i'])][1])

            plt.figure()
            plt.scatter(axis_x_red, axis_y_red, s=20, c='red', marker='o')
            plt.scatter(axis_x_blue, axis_y_blue, s=20, c='blue',
                        marker='o')
            plt.scatter(axis_x_query, axis_y_query, s=300, c='green', marker='X', alpha=0.7)
            plt.axis('off')
            cur_axes = plt.gca()
            cur_axes.axes.get_xaxis().set_visible(False)
            cur_axes.axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            fn = str(j)
            # plt.savefig(dir + '/' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
            plt_cnt += save_fig(plt, '{}/{}/gnn_{}'.format(extra_dir, plot_what, gnn_id), fn)
            plt_cnt += 1
            plt.close()
        print('Saved {} embedding visualization plots'.format(plt_cnt))


def get_text_label_for_ranking(ds_metric, qid, gid, norm, is_query, dataset,
                               gids_groundtruth, plot_gids, normed_matrix, unnormed_matrix):
    # norm = True
    rtn = ''

    if ds_metric == 'ged':  # here
        if norm:
            ds_label = 'nGED'
        else:
            ds_label = 'GED'
    elif ds_metric == 'glet':
        ds_label = 'glet'
    elif ds_metric == 'mcs':
        if norm:
            ds_label = 'nMCS'
        else:
            ds_label = 'MCS'
    else:
        raise NotImplementedError()
    if is_query:  
        if ds_metric == 'mcs':
            rtn += '{} by\n'.format(ds_label)  # nGED by \n A*
        else:
            if 'AIDS' in dataset or dataset == 'LINUX':
                rtn += '{} by\nA*'.format(ds_label)
            else:
                # rtn += '{} by Beam-\nHungarian-VJ'.format(ds_label)
                rtn += '{} by \nB-H-V'.format(ds_label)
    else:  # 
        ds_str = 'ged'
        # ds_str, ged_sim = r.dist_sim(qid, gid, norm)
        ged_sim        = normed_matrix[qid][gid]  # q 和g 的预测nged
        if ds_str == 'ged':
            ged_str = get_ged_select_norm_str(unnormed_matrix, normed_matrix, qid, gid, norm)
            if gid != gids_groundtruth[5]:
                rtn += '\n {}'.format(ged_str.split('(')[0])
            else:
                rtn += '\n ...   {}   ...'.format(ged_str.split('(')[0])
        else:
            rtn += '\n {:.2f}'.format(ged_sim)  # in case it is a numpy.float64, use float()
    if plot_gids:
        rtn += '\nid: {}'.format(gid)
    return rtn

def get_ged_select_norm_str(unnormed_matrix,normed_matrix, qid, gid, norm):
    # ged = r.dist_sim(qid, gid, norm=False)[1]
    ged = unnormed_matrix[qid][gid]   
    # norm_ged = r.dist_sim(qid, gid, norm=True)[1]
    norm_ged  =  normed_matrix[qid][gid]
    if norm:
        return '{:.2f}({})'.format(norm_ged, int(ged))
    else:
        return '{}({:.2f})'.format(int(ged), norm_ged)

def format_ds(ds):
    return '{:.2f}'.format(ds)

def get_norm_str(norm):
    if norm is None:
        return ''
    elif norm:
        return '_norm'
    else:
        return '_nonorm'

def set_save_paths_for_vis(info_dict, extra_dir, fn, plt_cnt):
    info_dict['plot_save_path_pdf'] = '{}/{}.{}'.format(extra_dir, fn, 'pdf')
    plt_cnt += 1
    return info_dict, plt_cnt

def draw_ranking(args, testing_graphs, trainval_graphs, pred_mat , pred_mat_unexp, existing_mappings = None, plot_gids=False, verbose=True, plot_max_num=np.inf, model_path = None, color = True, plot_node_ids = True):
    plot_what = 'ranking'
    concise = True
    c = None
    if color:
        c = 'color'
    else:
        c = 'gray'
    extra_dir = osp.join(args.extra_dir, args.dataset, plot_what, model_path, c)
    # em = self._get_node_mappings(data)
    # plot_node_ids= args.dataset != 'webeasy' and em
    types = trainval_graphs.types if args.dataset == 'AIDS700nef' else None
    if args.dataset == 'AIDS700nef':
        if color:
            color_map = get_color_map(trainval_graphs + testing_graphs, trainval_graphs)
        else:
            color_map = get_color_map(trainval_graphs + testing_graphs, trainval_graphs, use_color=color)
    else:
        color_map = None

    info_dict = {
        # draw node config
        'draw_node_size': 20,
        'draw_node_label_enable': True,
        'show_labels': plot_node_ids,
        'node_label_type': 'label' if plot_node_ids else 'type',
        'node_label_name': 'type' if args.dataset == 'AIDS700nef' else None,
        'draw_node_label_font_size': 6,
        'draw_node_color_map': color_map ,
        # draw edge config
        'draw_edge_label_enable': False,
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 10,
        'each_graph_text_pos': [0.5, 1.05],
        # graph padding: value range: [0, 1]
        'top_space': 0.20 if concise else 0.26,  # out of whole graph
        'bottom_space': 0.05,
        'hbetween_space': 0.6 if concise else 1,  # out of the subgraph
        'wbetween_space': 0,
        # plot config
        'plot_dpi': 200,
        'plot_save_path_eps': '',
        'plot_save_path_png': '',
        'plot_save_path_pdf': ''
    }
    plt_cnt = 0
    all_graphs = testing_graphs + trainval_graphs
    ground_truth_ged , ground_truth_nged, sort_id_mat, sort_id_mat_normed = get_true_result(all_graphs, testing_graphs, trainval_graphs)
    pred_ids_rank = get_pred_result(pred_mat)  # sim from big to small 
    for i in range(len(testing_graphs)):
        q = testing_graphs[i]
        middle_idx = len(trainval_graphs) // 2
        if len(trainval_graphs) < 5:
            print('Too few train gs {}'.format(len(trainval_graphs)))
            return        
        # Choose the top 6 matches, the overall middle match, and the worst match.
        selected_ids = list(range(5))  
        selected_ids.extend([middle_idx, -2, -1])   
        # Get the selected graphs from the groundtruth and the model.
        gids_groundtruth = np.array(sort_id_mat_normed[i][selected_ids])   
        gids_rank = np.array(pred_ids_rank[i][selected_ids])              
        # Top row graphs are only the groundtruth outputs.
        gs_groundtruth = [trainval_graphs[j] for j in gids_groundtruth]  # 
        # Bottom row graphs are the query graph + model ranking.
        gs_rank = [testing_graphs[i]] 
        gs_rank = gs_rank + [trainval_graphs[j] for j in gids_rank]  
        gs = gs_groundtruth + gs_rank  

        # Create the plot labels.
        text = []
        # First label is the name of the groundtruth algorithm, rest are scores for the graphs.
        text += [get_text_label_for_ranking('ged', i, i, True, True, args.dataset, gids_groundtruth, plot_gids, ground_truth_nged, ground_truth_ged)]
        text += [get_text_label_for_ranking(
                            'ged', i, j, True, False, args.dataset, gids_groundtruth, plot_gids, ground_truth_nged, ground_truth_ged)
                            for j in gids_groundtruth]
        # Start bottom row labels, just ranked from 1 to N with some fancy formatting.
        granu_label = 'similarity'.title() if verbose else 'Rank'
        text.append("{} \n {} by {}".format(granu_label, 'Rank', args.model_name))
        for j in range(len(gids_rank)):
            # ds = format_ds(pred_r.pred_ds(i, gids_rank[j], ds_norm))
            ds = -1 * np.log(pred_mat[i][gids_rank[j]])
            optional = '\n{:.2f}'.format(ds) if verbose else ''
            if j == len(gids_rank) - 3:
                rtn = '\n ...   {}   ...{}'.format(int(len(trainval_graphs) / 2), optional)
            elif j == len(gids_rank) - 2:
                rtn = '\n {}{}'.format(int(len(trainval_graphs)-1), optional)
            elif j == len(gids_rank) - 1:
                rtn = '\n {}{}'.format(int(len(trainval_graphs)), optional)
            else:
                rtn = '\n {}{}'.format(str(j + 1), optional)
            # rtn = '\n {}: {:.2f}'.format('sim', pred_r.sim_mat_[i][j])
            text.append(rtn)       
        # Perform the visualization.
        info_dict['each_graph_text_list'] = text
        fn = '{}_{}_{}{}'.format(
            plot_what, 'astar', int(testing_graphs[i]['i']), get_norm_str(True))  # ranking_astar_i_norm
        info_dict, plt_cnt = set_save_paths_for_vis(
            info_dict, extra_dir, fn, plt_cnt)    

        vis_small(q, gs, info_dict, types)
        if plt_cnt > plot_max_num:
            print('Saved {} query demo plots'.format(plt_cnt))
            return
    print('Saved {} query demo plots'.format(plt_cnt))


def get_color_map(gs, trainval_graphs, use_color = True):
    fl = len(FAVORITE_COLORS)
    rtn = {}
    ntypes = defaultdict(int) 
    types = trainval_graphs.types
    for g in gs:  
        for node in range(g.num_nodes):   
            node_type_idx = int(np.where(g.x[node].numpy() != 0)[0])
            node_type_name = types[node_type_idx]
            ntypes[node_type_name] += 1

    secondary = {}
    for i, (ntype, cnt) in enumerate(sorted(ntypes.items(), key=lambda x: x[1],
                                            reverse=True)):
        if ntype is None:
            color = None
            rtn[ntype] = color
        elif i >= fl:
            cmaps = plt.cm.get_cmap('hsv')
            color = cmaps((i - fl) / (len(ntypes) - fl))
            secondary[ntype] = color if use_color else mcolors.to_rgba('#CCCCCC')
        else:
            color = mcolors.to_rgba(FAVORITE_COLORS[i])[:3]
            rtn[ntype] = color if use_color else mcolors.to_rgba('#CCCCCC')
    if secondary:
        rtn.update((secondary))
    return rtn