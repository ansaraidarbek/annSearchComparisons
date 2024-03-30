import os
# print(os.getcwd())
# print(os.path.dirname(os.path.abspath(__file__)))
fullPath = os.path.dirname(os.path.abspath(__file__))
exec(open(fullPath + "/database.py").read())
exec(open(fullPath + "/helperFunctions.py").read())
exec(open(fullPath + "/1_ANNOY/main.py").read())
exec(open(fullPath + "/2_SCLEARN/main.py").read())
exec(open(fullPath + "/3_MRPT/main.py").read())
exec(open(fullPath + "/4_HNSW/main.py").read())
exec(open(fullPath + "/5_FAISS/main.py").read())
# exec(open(fullPath + "/6_DATASCETCH/main.py").read())
exec(open(fullPath + "/7_PYNNDESCENT/main.py").read())
exec(open(fullPath + "/8_SCIPPY/main.py").read())
# exec(open(fullPath + "/9_NMSLIB/main.py").read())

naming = ['mnist-784']
methods = ['annoy', 'sclearn_ballTree', 'sclearn_kdTree', 'mrpt', 'hnswlib', 'faiss_hnsw', 'faiss_ivf', 'pynndescent', 'scipy']
example = ['scipy']
results = []
metric = 'euclidean'
runs = 1
queries = 1000
for name in naming:
    results.append(annoy_run(name, metric, runs, queries))
    results.append(sclearn_run(name, metric, runs, queries, 'ball_tree'))
    results.append(sclearn_run(name, metric, runs, queries, 'kd_tree'))
    results.append(mrpt_run(name, metric, runs, queries))
    results.append(hnsw_run(name, metric, runs, queries))
    results.append(faiss_run(name, metric, runs, queries, 'hnsw'))
    results.append(faiss_run(name, metric, runs, queries, 'ivf'))
    results.append(pynndescent_run(name, metric, runs, queries))
    # results.append(scipy_run(name, metric, runs, queries))
    results.append(nmslib_run(name, metric, runs, queries))

if (len(example) == len(results)):
    for i in range(len(results)):
        print(methods[i], results[i])
