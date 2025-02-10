from pickledb import PickleDB
import time
import random
import os

def generate_fake_data(num_entries=100000, embedding_length=512):
    fake_data = {}
    
    for i in range(num_entries):
        name = f"Name_{i}"
        embedding = [random.uniform(-1, 1) for _ in range(embedding_length)]
        fake_data[name] = embedding
    
    return fake_data

if __name__ == '__main__':
    db = PickleDB('hieu2.db') # 50000 samples
    # db = PickleDB('hieu2.db') # 100000 samples
    # fake_data = generate_fake_data()

    # for name, embedding in fake_data.items():
    #     db.set(name, embedding)
    
    # db.save()

    input_embedding = [random.uniform(-1, 1) for _ in range(512)]
    n = 2 
    metric = 'cosine'  

    # start_time = time.perf_counter()
    # nearest = db.find_nearest_embeddings_nump(input_embedding, n, metric)
    # multi_time = time.perf_counter() - start_time

    # print("Nearest neighbors:")
    # for name, dist in nearest:
    #     print(f"Name: {name}, Distance: {dist}")

    # print(f"Time taken with numpy vectorize: {multi_time:.6f} seconds")
    # print("-"*15)
    
    start_time = time.perf_counter()
    nearest = db.find_nearest_embeddings(input_embedding, n, metric)
    simple_time = time.perf_counter() - start_time

    print("Nearest neighbors:")
    for name, dist in nearest:
        print(f"Name: {name}, Distance: {dist}")

    print(f"Time taken simple with 2 threads: {simple_time:.6f} seconds")

    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    start_time = time.perf_counter()
    nearest = db.find_nearest_embeddings(input_embedding, n, metric)
    simple_time = time.perf_counter() - start_time

    print("Nearest neighbors:")
    for name, dist in nearest:
        print(f"Name: {name}, Distance: {dist}")

    print(f"Time taken simple with 4 threads: {simple_time:.6f} seconds")


