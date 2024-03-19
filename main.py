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
exec(open(fullPath + "/9_NMSLIB/main.py").read())

naming = ['glove-25', 'glove-50', 'glove-100', 'glove-200', 'nytimes-256', 'sift-128', 'lastfm-64']
problematic = ['deepImage-96', 'gist-960']
methods = ['annoy', 'sclearn_ballTree', 'sclearn_kdTree', 'mrpt', 'hnswlib', 'faiss_hnsw', 'faiss_ivf', 'pynndescent', 'scipy', 'nmslib']
datasets_list = ['deepImage-96', 'fashionMnist-784', 'gist-960', 'glove-25', 'glove-50', 'glove-100', 'glove-200', 'mnist-784', 'nytimes-256', 'sift-128', 'lastfm-64']
results = []
metric = 'euclidean'
example = ['mnist-784', 'fashionMnist-784']
runs = 10
queries = 1000
for name in naming:
    result = []
    result.append(annoy_run(name, metric, runs, queries))
    result.append(sclearn_run(name, metric, runs, queries, 'ball_tree'))
    result.append(sclearn_run(name, metric, runs, queries, 'kd_tree'))
    result.append(mrpt_run(name, metric, runs, queries))
    result.append(hnsw_run(name, metric, runs, queries))
    result.append(faiss_run(name, metric, runs, queries, 'hnsw'))
    result.append(faiss_run(name, metric, runs, queries, 'ivf'))
    result.append(pynndescent_run(name, metric, runs, queries))
    result.append(scipy_run(name, metric, runs, queries))
    result.append(nmslib_run(name, metric, runs, queries))
    results.append(result)

for i in range(len(naming)):
    print(naming[i] + "start ---------------------------------------")
    for j in range(len(results[i])):
        if i < len(methods) :
            print(methods[i], end=', ')
        print(results[i][j])
    print(naming[i] + "end ---------------------------------------")
