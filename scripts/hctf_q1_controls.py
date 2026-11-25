"""Independent candidate sources and post-decision algorithmic task proxies.

Routing measures an algorithm on a drawing, not human performance. Community
recovery uses a declared reference partition, not assumed real-world truth.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ALPHAS = (0., .0625, .125, .25, .5, .75, 1.)
SIMPLE_SOURCES = ('random', 'stress_gradient', 'community_gradient',
                  'weighted_smacof_5', 'weighted_smacof_25', 'weighted_smacof_100', 'kk')
TASK_METRICS = ('routing_success_rate', 'routing_efficiency',
                'community_recovery_ari', 'community_recovery_nmi')


def graph_from_edges(num_nodes, edge_index):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    values = edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    graph.add_edges_from((int(a), int(b)) for a, b in values.T if a != b)
    return graph


def task_queries(graph, seed, maximum=128):
    """Uniform ordered reachable pairs without replacement, fixed before layouts."""
    pairs = [(a, b) for a in graph for b in sorted(nx.node_connected_component(graph, a)) if a != b]
    rng = np.random.default_rng(seed)
    if len(pairs) > maximum:
        indices = np.sort(rng.choice(len(pairs), maximum, replace=False))
        pairs = [pairs[index] for index in indices]
    return [(a, b, nx.shortest_path_length(graph, a, b)) for a, b in pairs]


def task_proxy_metrics(pos, graph, labels, queries, seed):
    positions = np.asarray(pos, dtype=float)
    successes, efficiencies = [], []
    for source, target, optimum in queries:
        current, visited, steps = source, {source}, 0
        while current != target and steps < len(graph):
            neighbors = sorted(graph[current])
            if not neighbors:
                break
            distances = [(float(np.linalg.norm(positions[node]-positions[target])), node) for node in neighbors]
            distance, following = min(distances)
            # Strict geometric descent avoids cycles. Index breaks exact ties.
            if following in visited or distance >= np.linalg.norm(positions[current]-positions[target])-1e-12:
                break
            current = following
            visited.add(current)
            steps += 1
        successful = current == target
        successes.append(float(successful))
        efficiencies.append(float(optimum / steps) if successful and steps else 0.)
    if not queries:
        raise ValueError('Routing requires at least one reachable ordered pair')
    labels = np.asarray(labels)
    count = len(np.unique(labels))
    recovered = KMeans(n_clusters=count, n_init=10, random_state=seed).fit_predict(positions)
    return {'routing_success_rate': float(np.mean(successes)),
            'routing_efficiency': float(np.mean(efficiencies)),
            'community_recovery_ari': float(adjusted_rand_score(labels, recovered)),
            'community_recovery_nmi': float(normalized_mutual_info_score(labels, recovered))}


def weighted_smacof(initial, graph, steps):
    """Fixed-step 1/d² weighted stress majorization on connected graphs.

    No test-based stopping or best-iterate choice. Singular translation mode
    is removed with a pseudoinverse and zero-mean updates.
    """
    distances = np.asarray(nx.floyd_warshall_numpy(graph, nodelist=list(range(len(graph)))), dtype=float)
    if not np.isfinite(distances).all():
        raise ValueError('Weighted SMACOF expects a connected graph')
    weights = np.zeros_like(distances)
    valid = distances > 0
    weights[valid] = 1. / distances[valid]**2
    laplacian = np.diag(weights.sum(axis=1))-weights
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    result = np.asarray(initial, dtype=float).copy()
    for _ in range(steps):
        length = np.linalg.norm(result[:, None]-result[None, :], axis=2)
        coefficients = weights*distances / np.maximum(length, 1e-12)
        matrix = -coefficients
        np.fill_diagonal(matrix, 0.)
        np.fill_diagonal(matrix, -matrix.sum(axis=1))
        result = inverse @ matrix @ result
        result -= result.mean(axis=0)
    return result


def simple_candidate(source, baseline, graph, labels, seed, train, record):
    initial = np.asarray(baseline, dtype=float)
    if source.startswith('weighted_smacof_'):
        return weighted_smacof(initial, graph, int(source.rsplit('_', 1)[1]))
    if source == 'kk':
        values = nx.kamada_kawai_layout(graph, pos={i: initial[i] for i in graph})
        return np.asarray([values[i] for i in graph])
    lengths = [np.linalg.norm(initial[a]-initial[b]) for a, b in graph.edges]
    scale = max(float(np.median(lengths)), 1e-8)
    if source == 'random':
        noise = np.random.default_rng(seed).normal(size=initial.shape)
        noise -= noise.mean(axis=0)
        return initial + .05*scale*noise
    if source not in ('stress_gradient', 'community_gradient'):
        raise ValueError(source)
    with torch.enable_grad():
        coords = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([coords], lr=.02*scale)
        for _ in range(5):
            optimizer.zero_grad()
            if source == 'stress_gradient':
                loss = train.local_deepgd_stress_tensor(coords, record['perm_index'].cpu(), record['apsp_attr'].cpu())
            else:
                groups = [coords[np.flatnonzero(np.asarray(labels)==label)] for label in np.unique(labels)]
                centers = torch.stack([group.mean(dim=0) for group in groups])
                compactness = torch.stack([(group-center).square().sum(dim=1).mean() for group, center in zip(groups, centers)]).mean()
                separation = torch.pdist(centers).square().mean() if len(groups)>1 else coords.sum()*0.
                loss = compactness/(scale*scale)-.1*separation/(scale*scale)
            loss.backward()
            optimizer.step()
        return coords.detach().numpy()


def choose_safe(baseline_pos, baseline_metrics, proposal, record, labels, evaluator):
    """Independent identical selector per source; task endpoints never enter."""
    candidate, _ = evaluator.measure_layout(proposal, record, labels, include_holdout=False)
    aligned = evaluator.train.mtpf_procrustes_align(candidate, baseline_pos)
    weights = evaluator.objective_metric_weights('')
    best_pos, best_metrics, best_alpha, best_score = baseline_pos, baseline_metrics, 0., 0.
    for alpha in ALPHAS[1:]:
        coords, metrics = evaluator.measure_layout(baseline_pos+alpha*(aligned-baseline_pos), record, labels, include_holdout=False)
        if not evaluator.is_componentwise_safe(baseline_metrics, metrics, relative_tolerance=0.):
            continue
        gains = [evaluator.improvement(baseline_metrics, metrics, name) for name in evaluator.METRICS]
        score = sum(weights[name]*gain for name, gain in zip(evaluator.METRICS, gains))
        if any(gain>1e-10 for gain in gains) and (score>best_score+1e-12 or (abs(score-best_score)<=1e-12 and alpha>best_alpha)):
            best_pos, best_metrics, best_alpha, best_score = coords, metrics, alpha, score
    return best_pos, best_metrics, best_alpha
