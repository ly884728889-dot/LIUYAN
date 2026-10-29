"""Train-only GT orientation supervision, extracted from the frozen generator."""
import numpy as np
import sys
reference_direction = sys.modules[__name__]

def skeletonize(mask):
    """Zhang-Suen thinning, with a zero border and no optional dependencies."""
    image = np.pad(np.asarray(mask, bool), 1)
    while True:
        removed = 0
        for step in (0, 1):
            p = (image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:], image[2:, 2:],
                 image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2], image[:-2, :-2])
            neighbors = sum(v.astype(np.uint8) for v in p)
            transitions = sum((~p[i] & p[(i+1) % 8]).astype(np.uint8) for i in range(8))
            if step == 0:
                triplets = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                triplets = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            erase = image[1:-1, 1:-1] & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & triplets
            removed += int(erase.sum())
            image[1:-1, 1:-1][erase] = False
        if not removed:
            return image[1:-1, 1:-1]

def _skeleton_graph(skeleton):
    """Build an 8-neighbour graph but remove diagonal corner shortcuts."""
    points = [tuple(v) for v in np.argwhere(skeleton)]
    present = set(points)
    graph = {point: [] for point in points}
    for y, x in points:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                neighbor = y + dy, x + dx
                if neighbor not in present:
                    continue
                # A right-angle staircase must not acquire a third diagonal edge.
                if dy and dx and ((y, x + dx) in present or (y + dy, x) in present):
                    continue
                graph[(y, x)].append(neighbor)
    return graph

def _components(graph, excluded):
    pending, output = set(graph).difference(excluded), []
    while pending:
        seed = pending.pop()
        component, queue = {seed}, [seed]
        for node in queue:
            for neighbor in graph[node]:
                if neighbor in pending and neighbor not in excluded:
                    pending.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        output.append(component)
    return output

def _arc_neighborhood(graph, component, seed, radius):
    """Geodesic neighbourhood that remains on one branch."""
    distance, queue = {seed: 0}, [seed]
    for node in queue:
        if distance[node] == radius:
            continue
        for neighbor in graph[node]:
            if neighbor in component and neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return np.asarray(list(distance), np.float64)

def batched_tangents(neighborhoods, min_coherence):
    """Same float64 reductions on equal-length neighborhoods, preserving order."""
    output = [None] * len(neighborhoods)
    groups = {}
    for index, points in enumerate(neighborhoods):
        if len(points) >= 3:
            groups.setdefault(len(points), []).append(index)
    for indices in groups.values():
        points = np.stack([neighborhoods[index] for index in indices])
        centered = points - points.mean(axis=1)[:, None, :]
        yy = np.mean(centered[:, :, 0] ** 2, axis=1)
        xx = np.mean(centered[:, :, 1] ** 2, axis=1)
        xy = np.mean(centered[:, :, 0] * centered[:, :, 1], axis=1)
        coherence = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2) / np.maximum(xx + yy, 1e-12)
        theta = (.5 * np.arctan2(2 * xy, xx - yy)) % np.pi
        for local, index in enumerate(indices):
            if not coherence[local] < min_coherence:
                output[index] = theta[local]
    return output

def orientation_gt(mask, stride=16, bins=8, radius=5, min_coherence=.3, *, return_skeleton=False):
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    if h % stride or w % stride:
        raise ValueError('Mask dimensions must be divisible by stride')
    skel = reference_direction.skeletonize(mask)
    graph = reference_direction._skeleton_graph(skel)
    junction_nodes = {node for node, neighbours in graph.items() if len(neighbours) >= 3}
    junction = np.zeros_like(skel)
    for y, x in junction_nodes:
        junction[y, x] = True
    components = reference_direction._components(graph, junction_nodes)
    component_of = {node: component for component in components for node in component}
    histogram = np.zeros((bins, h // stride, w // stride), np.float32)
    tangent = np.full((h, w), np.nan, np.float32)
    points, neighborhoods, is_incident = [], [], []
    for component in components:
        if len(component) < 3:
            continue
        for point in component:
            points.append(point)
            neighborhoods.append(reference_direction._arc_neighborhood(graph, component, point, radius))
            is_incident.append(False)
    for y, x in junction_nodes:
        for neighbour in graph[(y, x)]:
            component = component_of.get(neighbour)
            if component is None:
                continue
            points.append((y, x))
            neighborhoods.append(reference_direction._arc_neighborhood(graph, component, neighbour, radius))
            is_incident.append(True)
    angles = batched_tangents(neighborhoods, min_coherence)
    incident_votes = 0
    for (y, x), theta, incident in zip(points, angles, is_incident):
        if theta is None:
            continue
        if incident:
            incident_votes += 1
        else:
            tangent[y, x] = theta
        # Preserve reference vote order and two float32 additions per vote.
        position = theta * bins / np.pi
        lower = int(np.floor(position)) % bins
        fraction = position - np.floor(position)
        histogram[lower, y // stride, x // stride] += 1 - fraction
        histogram[(lower + 1) % bins, y // stride, x // stride] += fraction
    mass = histogram.sum(0)
    valid = mass > 0
    histogram /= np.maximum(mass[None], 1)
    histogram[:, ~valid] = 1 / bins
    result = (histogram, valid.astype(np.float32), dict(
        skeleton_pixels=int(skel.sum()), usable_tangents=int(np.isfinite(tangent).sum()),
        junction_pixels=int(junction.sum()), branches=len(components),
        junction_incident_votes=incident_votes, valid_tokens=int(valid.sum())), tangent)
    return (*result, skel) if return_skeleton else result
