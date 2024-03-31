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

naming = ['mnist-784', 'fashionMnist-784', 'nytimes-256', 'lastfm-64']
problematic = ['deepImage-96', 'gist-960']
methods = ['annoy', 'sclearn_ballTree', 'sclearn_kdTree', 'mrpt', 'hnswlib', 'faiss_hnsw', 'faiss_ivf', 'pynndescent', 'scipy', 'nmslib']
datasets_list = ['deepImage-96', 'fashionMnist-784', 'gist-960', 'glove-25', 'glove-50', 'glove-100', 'glove-200', 'mnist-784', 'nytimes-256', 'sift-128', 'lastfm-64']
results = []
metric = 'euclidean'
runs = 1
queries = 1000
for name in naming:
    print(name + "-----------------------------------")
    result = []
    annoy = annoy_run(name, metric, runs, queries)
    print(annoy)
    result.append(annoy)
    sclearnBallTree = sclearn_run(name, metric, runs, queries, 'ball_tree')
    print(sclearnBallTree)
    result.append(sclearnBallTree)
    sclearnKdTree = sclearn_run(name, metric, runs, queries, 'kd_tree')
    print(sclearnKdTree)
    result.append(sclearnKdTree)
    mrpt = mrpt_run(name, metric, runs, queries)
    print(mrpt)
    result.append(mrpt)
    hnswLib = hnsw_run(name, metric, runs, queries)
    print(hnswLib)
    result.append(hnswLib)
    faissHnsw = faiss_run(name, metric, runs, queries, 'hnsw')
    print(faissHnsw)
    result.append(faissHnsw)
    faissIvf = faiss_run(name, metric, runs, queries, 'ivf')
    print(faissIvf)
    result.append(faissIvf)
    pyNNDescent = pynndescent_run(name, metric, runs, queries)
    print(pyNNDescent)
    result.append(pyNNDescent)
    scippy = scipy_run(name, metric, runs, queries)
    print(scippy)
    result.append(scippy)
    nmslibNhsw = nmslib_run(name, metric, runs, queries)
    print(nmslibNhsw)
    result.append(nmslibNhsw)
    results.append(result)
    for j in range(len(result)):
        if j < len(methods) :
            print(methods[j], end=', ')
        print(result[j])

for i in range(len(naming)):
    print(naming[i] + "start ---------------------------------------")
    for j in range(len(results[i])):
        if j < len(methods) :
            print(methods[j], end=', ')
        print(results[i][j])
    print(naming[i] + "end ---------------------------------------")
