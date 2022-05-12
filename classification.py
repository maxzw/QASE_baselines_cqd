"""Custom evaluation functions used for binary classification."""
import logging
from typing import Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

from cqd.base import CQD
from models import KGReasoning

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }


def get_class_metrics(
    distances: np.ndarray, 
    easy_answers: np.ndarray, 
    hard_answers: np.ndarray, 
    threshold: float
) -> Tuple[float, float, float, float]:

    epsilon = 1e-6
    
    selection_mask = (distances < threshold)

    tp = np.sum(np.where(selection_mask & ~easy_answers, hard_answers, False), axis=1)
    fp = np.sum(np.where(selection_mask & ~easy_answers, ~hard_answers, False), axis=1)
    tn = np.sum(np.where(~selection_mask & ~easy_answers, ~hard_answers, False), axis=1)
    fn = np.sum(np.where(~selection_mask & ~easy_answers, hard_answers, False), axis=1)

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)

    weights = 1 / hard_answers.sum(axis=1)

    # Return the average of the metrics for the whole batch
    avg_acc = np.average(accuracy, weights=weights).item()
    avg_prec = np.average(precision, weights=weights).item()
    avg_rec = np.average(recall, weights=weights).item()
    avg_f1 = np.average(f1, weights=weights).item()

    return avg_acc, avg_prec, avg_rec, avg_f1


def find_best_threshold(
    distances: np.ndarray, 
    easy_answers: np.ndarray,
    hard_answers: np.ndarray, 
    struct_str: str = None, 
    num_steps: int = 50,
    model_name: str = None,
    dataset_name: str = None,
    save_path = None
    ) -> Tuple[float, float, float, float, float]:

    # track precision and recall
    precisions = []
    recalls = []

    pos_dists = np.where(easy_answers, distances, 0) # find thresholds based on valid easy answers
    pos_dists[pos_dists==0] = np.nan
    pos_dists_mean = np.nanmean(pos_dists)
    
    if model_name == "CQD":
        pbounds = {'threshold': (np.quantile(pos_dists_mean, 0.2), np.quantile(pos_dists_mean, 0.8))}
    else:
        pos_dists_std = np.nanstd(pos_dists) * 3
        pbounds = {'threshold': (pos_dists_mean - pos_dists_std, pos_dists_mean + pos_dists_std)}

    def objective(threshold):
        accuracy, precision, recall, f1 = get_class_metrics(distances, easy_answers, hard_answers, threshold)
        precisions.append(precision)
        recalls.append(recall)
        return f1
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=20,
        n_iter=num_steps,

    )

    if (model_name is not None) and (dataset_name is not None) and (struct_str is not None) and (save_path is not None):
        # save current optimization metrics (2) if needed
        plt.figure(2, figsize=(10,10))
        x = np.array([step['params']['threshold'] for step in optimizer.res])
        y = np.array([step['target'] for step in optimizer.res])
        x_order = np.argsort(x)
        x = x[x_order]
        y = y[x_order]
        r = np.array(recalls)[x_order]
        p = np.array(precisions)[x_order]
        plt.plot(x, y, '-', label="f1")
        plt.plot(x, r, '-', label="recall")
        plt.plot(x, p, '-', label="precision")
        plt.legend()
        plt.xlabel("Distance threshold")
        plt.ylabel("Score")
        plt.title(f"{model_name}_{dataset_name}_{struct_str}")
        plt.savefig(f"{save_path}/{model_name}_{dataset_name}_{struct_str}.png", facecolor='w', bbox_inches='tight')
        plt.clf()

    if struct_str is not None:
        # save figure to collective f1 plot (1) if needed
        plt.figure(1)
        x = np.array([step['params']['threshold'] for step in optimizer.res])
        y = np.array([step['target'] for step in optimizer.res])
        x_order = np.argsort(x)
        x = x[x_order]
        y = y[x_order]
        plt.plot(x, y, 'x-', label=struct_str)

    best_threshold = optimizer.max['params']['threshold']
    best_accuracy, best_precision, best_recall, best_f1 = get_class_metrics(distances, easy_answers, hard_answers, best_threshold)

    return best_threshold, best_accuracy, best_precision, best_recall, best_f1


def find_val_thresholds(model, easy_answers, hard_answers, args, test_dataloader):
    model.eval()

    model_name = None
    if args.geo == 'vec':
        model_name = 'GQE'
    elif args.geo == 'beta':
        model_name = 'BetaE'
    elif args.geo == 'box':
        model_name = 'Q2B'
    elif args.geo == 'cqd':
        model_name = 'CQD'
    dataset_name = str(args.data_path).split("/")[-1].split('-')[0]
    
    # tracking thresholds and scores
    thresholds = {}
    metrics = {}

    step = 0
    total_steps = len(test_dataloader)

    # track queries, distances and answers
    all_query_stuctures = []
    all_distances = torch.empty((0, model.nentity))
    all_easy_answers_mask = torch.empty((0, model.nentity))
    all_hard_answers_mask = torch.empty((0, model.nentity))
    
    requires_grad = isinstance(model, CQD) and model.method == 'continuous'

    with torch.set_grad_enabled(requires_grad):
        for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
            
            # get distances
            batch_queries_dict = defaultdict(list)
            batch_idxs_dict = defaultdict(list)
            for i, query in enumerate(queries):
                batch_queries_dict[query_structures[i]].append(query)
                batch_idxs_dict[query_structures[i]].append(i)
            for query_structure in batch_queries_dict:
                if args.cuda:
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                else:
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
            if args.cuda:
                negative_sample = negative_sample.cuda()
            # negative logit, size: (batch_size, num_ents)
            _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [queries_unflatten[i] for i in idxs]
            query_structures = [str(query_structures[i]) for i in idxs]

            # get answers
            easy_answer_ent_ids = [list(easy_answers[query]) for query in queries_unflatten] # size: (batch_size, VARIABLE)
            hard_answer_ent_ids = [list(hard_answers[query]) for query in queries_unflatten] # size: (batch_size, VARIABLE)
            easy_answers_mask = torch.zeros((len(query_structures), model.nentity)) # initialize with zeros, size: (batch_size, num_ents)
            hard_answers_mask = torch.zeros((len(query_structures), model.nentity))
            for i, (easy_ent_ids, hard_ent_ids) in enumerate(zip(easy_answer_ent_ids, hard_answer_ent_ids)):
                easy_answers_mask[i, easy_ent_ids] = 1
                hard_answers_mask[i, hard_ent_ids] = 1

            # add to tracking
            all_query_stuctures.extend(query_structures)
            all_distances = torch.cat((all_distances, negative_logit.cpu()), dim=0)
            all_easy_answers_mask = torch.cat((all_easy_answers_mask, easy_answers_mask.cpu()), dim=0)
            all_hard_answers_mask = torch.cat((all_hard_answers_mask, hard_answers_mask.cpu()), dim=0)

            if step % 10 == 0:
                logging.info('Gathering predictions of batches... (%d/%d) ' % (step, total_steps))
            step += 1

    # IMPORTANT: reset to raw distances
    if isinstance(model, KGReasoning):
        all_distances = model.gamma.cpu() - all_distances

    # Define plot
    plt.figure(1, figsize=(10,10))

    # find best threshold for each query structure
    for struct in set(all_query_stuctures):
        logging.info(f"Finding best threshold for structure: {struct} / {query_name_dict[eval(struct)]}")

        # select data for current structure
        struct_idx = np.array(np.where(np.array(all_query_stuctures) == struct)[0])
        str_distances = all_distances[struct_idx, :]
        str_easy_answers_mask = all_easy_answers_mask[struct_idx, :]
        str_hard_answers_mask = all_hard_answers_mask[struct_idx, :]

        # find best threshold and metrics
        threshold, accuracy, precision, recall, f1 = find_best_threshold(
            str_distances.numpy(),
            str_easy_answers_mask.bool().numpy(),
            str_hard_answers_mask.bool().numpy(),
            struct_str=query_name_dict[eval(struct)],
            num_steps=100,
            model_name=model_name,
            dataset_name=dataset_name,
            save_path=args.save_path
        )

        logging.info(f"Threshold: {threshold:.4f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # save threshold and metrics
        thresholds[struct] = threshold
        metrics[struct] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Save figure
    plt.figure(1)
    if (model_name is not None) and (dataset_name is not None):
        plt.title("Opt_{}_{}".format(model_name, dataset_name))
    else:
        plt.title("Optimization results")
    plt.xlabel('Distance threshold')
    plt.ylabel('f1-score')
    plt.legend()
    plt.savefig(args.save_path + "/threshold_search.png", facecolor='w', bbox_inches='tight')
    plt.clf()

    return thresholds, metrics


def evaluate_with_thresholds(model, easy_answers, hard_answers, args, test_dataloader, thresholds):
    model.eval()
    
    # tracking thresholds and scores
    metrics = {}

    step = 0
    total_steps = len(test_dataloader)

    # track queries, distances and answers
    all_query_stuctures = []
    all_distances = torch.empty((0, model.nentity))
    all_easy_answers_mask = torch.empty((0, model.nentity))
    all_hard_answers_mask = torch.empty((0, model.nentity))

    requires_grad = isinstance(model, CQD) and model.method == 'continuous'

    with torch.set_grad_enabled(requires_grad):
        for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
            
            # get distances
            batch_queries_dict = defaultdict(list)
            batch_idxs_dict = defaultdict(list)
            for i, query in enumerate(queries):
                batch_queries_dict[query_structures[i]].append(query)
                batch_idxs_dict[query_structures[i]].append(i)
            for query_structure in batch_queries_dict:
                if args.cuda:
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                else:
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
            if args.cuda:
                negative_sample = negative_sample.cuda()
            # negative logit, size: (batch_size, num_ents)
            _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [queries_unflatten[i] for i in idxs]
            query_structures = [str(query_structures[i]) for i in idxs]

            # get answers
            easy_answer_ent_ids = [list(easy_answers[query]) for query in queries_unflatten] # size: (batch_size, VARIABLE)
            hard_answer_ent_ids = [list(hard_answers[query]) for query in queries_unflatten] # size: (batch_size, VARIABLE)
            easy_answers_mask = torch.zeros((len(query_structures), model.nentity)) # initialize with zeros, size: (batch_size, num_ents)
            hard_answers_mask = torch.zeros((len(query_structures), model.nentity))
            for i, (easy_ent_ids, hard_ent_ids) in enumerate(zip(easy_answer_ent_ids, hard_answer_ent_ids)):
                easy_answers_mask[i, easy_ent_ids] = 1
                hard_answers_mask[i, hard_ent_ids] = 1

            # add to tracking
            all_query_stuctures.extend(query_structures)
            all_distances = torch.cat((all_distances, negative_logit.cpu()), dim=0)
            all_easy_answers_mask = torch.cat((all_easy_answers_mask, easy_answers_mask.cpu()), dim=0)
            all_hard_answers_mask = torch.cat((all_hard_answers_mask, hard_answers_mask.cpu()), dim=0)

            if step % 10 == 0:
                logging.info('Gathering predictions of batches... (%d/%d)' % (step, total_steps))
            step += 1

    # IMPORTANT: reset to raw distances
    if isinstance(model, KGReasoning):
        all_distances = model.gamma.cpu() - all_distances

    struct_sizes = {}
    # find best threshold for each query structure
    for struct in set(all_query_stuctures):
        logging.info(f"Calculating metrics for structure: {struct}")

        # select data for current structure
        struct_idx = torch.tensor(np.where(np.array(all_query_stuctures) == struct)[0])
        str_distances = all_distances[struct_idx, :]
        str_easy_answers_mask = all_easy_answers_mask[struct_idx, :]
        str_hard_answers_mask = all_hard_answers_mask[struct_idx, :]

        # track size of current structure for weighted metrics
        struct_sizes[struct] = len(struct_idx)

        # find best threshold and metrics
        accuracy, precision, recall, f1 = get_class_metrics(
            str_distances.numpy(), 
            str_easy_answers_mask.bool().numpy(),
            str_hard_answers_mask.bool().numpy(),
            thresholds[struct]
        )

        # save threshold and metrics
        metrics[struct] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    metrics['macro'] = {
        'accuracy': np.mean([metrics[struct]['accuracy'] for struct in metrics]),
        'precision': np.mean([metrics[struct]['precision'] for struct in metrics]),
        'recall': np.mean([metrics[struct]['recall'] for struct in metrics]),
        'f1': np.mean([metrics[struct]['f1'] for struct in metrics])
    }

    metrics['weighted'] = {
        'accuracy': np.average(
            [metrics[struct]['accuracy'] for struct in metrics if struct is not 'macro'],
            weights=[struct_sizes[struct] for struct in metrics if struct is not 'macro']
        ),
        'precision': np.average(
            [metrics[struct]['precision'] for struct in metrics if struct is not 'macro'],
            weights=[struct_sizes[struct] for struct in metrics if struct is not 'macro']
        ),
        'recall': np.average(
            [metrics[struct]['recall'] for struct in metrics if struct is not 'macro'],
            weights=[struct_sizes[struct] for struct in metrics if struct is not 'macro']
        ),
        'f1': np.average(
            [metrics[struct]['f1'] for struct in metrics if struct is not 'macro'],
            weights=[struct_sizes[struct] for struct in metrics if struct is not 'macro']
        )
    }

    return metrics
