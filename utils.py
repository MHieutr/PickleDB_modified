import numpy as np
from multiprocessing import Pool, cpu_count

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def cosine_distance(a, b):
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1 - similarity

def _compute_distance(args):
    key, value, embedding, metric = args
    value = np.array(value, dtype= np.float64)
    if metric == 'euclidean':
        dist = euclidean_distance(embedding, value)
    elif metric == 'cosine':
        dist = cosine_distance(embedding, value)
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")
    return (key, dist)

def compute_distances_parallel(embeddings, input_embedding, metric):
    num_workers = min(cpu_count(), len(embeddings)) 
    args = [(key, value, input_embedding, metric) for key, value in embeddings.items()]

    with Pool(num_workers) as pool:
        distances = pool.map(_compute_distance, args)
    return distances


def find_nearest_neighbors(embeddings, input_embedding, n, metric='euclidean'):
    input_embedding = np.array(input_embedding, dtype=np.float64)
    distances = compute_distances_parallel(embeddings, input_embedding, metric)
    distances.sort(key=lambda x: x[1])  
    return distances[:n]


def find_nearest_neighbors_simple(embeddings, input_embedding, n, metric='euclidean'):
    input_embedding = np.array(input_embedding, dtype=np.float64)
    distances = []

    for key, value in embeddings.items():
        value = np.array(value, dtype=np.float64)
        if metric == 'euclidean':
            dist = euclidean_distance(input_embedding, value)
        elif metric == 'cosine':
            dist = cosine_distance(input_embedding, value)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")
        distances.append((key, dist))
    
    distances.sort(key=lambda x: x[1])
    return distances[:n]


