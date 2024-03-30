naming = ['first', 'second', 'third']
methods = ['first', 'second', 'third']
results = [[1,2,3], [1,2,3], [1,2,3]]

for i in range(len(naming)):
    print(naming[i] + "start ---------------------------------------")
    for j in range(len(results[i])):
        if i < len(methods) :
            print(methods[i], end=', ')
        print(results[i][j])
    print(naming[i] + "end ---------------------------------------")