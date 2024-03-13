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

naming = [
    'mnist-784-euclidean'
]

for name in naming:
    annoy_run(name)
    sclearn_run(name)
    mrpt_run(name)
    hnsw_run(name)
    faiss_run(name)
    pynndescent_run(name)
    scipy_run(name)
    nmslib_run(name)
