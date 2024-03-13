exec(open("./database.py").read())
exec(open("./helperFunctions.py").read())
exec(open("./1_ANNOY/main.py").read())
exec(open("./2_SCLEARN/main.py").read())
exec(open("./3_MRPT/main.py").read())
exec(open("./4_HNSW/main.py").read())
exec(open("./5_FAISS/main.py").read())
# exec(open("./6_DATASCETCH/main.py").read())
exec(open("./7_PYNNDESCENT/main.py").read())
exec(open("./8_SCIPPY/main.py").read())
exec(open("./9_NMSLIB/main.py").read())

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
