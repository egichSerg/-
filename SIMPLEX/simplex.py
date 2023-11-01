import numpy as np

#add fictive variables to matrix and make fictive basis
def toBasis(mtx):
    mtx = np.append(np.zeros(shape=(1, mtx.shape[1])), mtx, axis=0)
    for i in range(1, mtx.shape[0]):
        mtx = np.append(mtx, [[1 if j == i or j == 0 else 0] for j in range(mtx.shape[0])], axis=1)
        mtx[0] -= mtx[i]

    return mtx

def hasNegatives(lst):
    for i in lst:
        if i < 0:
            return True 
    return False

def minNegativeElementIndex(lst):
    cur_negative = 0
    for i in range(len(lst)):
        if lst[i] < lst[cur_negative]:
            cur_negative = i
    return cur_negative

#mtx - matrix
#bci - basis column index
def lexigraphicMin(mtx, bci):
    col = mtx[1:, 0] / mtx[1:, bci]
    similar_rows = np.array([np.where(col == element)[0].tolist() for element in np.unique(col) if element >= 0 and np.abs(element) != np.inf])

    if len(similar_rows) == 0: #if empty, full lexigraphic analysis
        similar_rows = np.arange(1, mtx.shape[0])
    else:
        similar_rows = min(similar_rows, key=lambda x : col[x[0]]) + 1

    if (len(similar_rows) == 1):
        print(similar_rows[0])
        return similar_rows[0]
    
    for column in range(1, mtx.shape[1]):
        elems = mtx[1:, column]
        if not np.all(elems == elems[0]):
            minRow = similar_rows[np.argmin(elems)]
            return minRow

def simplexStep(mtx):
    basisColumn = minNegativeElementIndex(mtx[0])
    if basisColumn == 0:
        return mtx
    
    basisRow = lexigraphicMin(mtx, basisColumn)
    
    for i in range(mtx.shape[0]):
        if i == basisRow or mtx[basisRow][basisColumn] == 0:
            continue

        rowFactor = mtx[i][basisColumn] / mtx[basisRow][basisColumn]
        mtx[i] -= mtx[basisRow] * rowFactor

    return mtx

def removeNegativesFirstCol(mtx):
    negativeRows = np.where(mtx[:, 0] < 0)
    for row in negativeRows:
        mtx[row] *= -1
    return mtx

def main():
    matrix = np.array([
    [1,  1,  1,  1,  0,  0,  0,  0],
    [5, 10, -1,  2, -1,  1, -1,  0],
    [7, 12,  1, 12,  1, -1,  0,  1]
    ])

    # matrix = np.array([
    # [1,  1,  1,  1,  0,  0,  0,  0],
    # [-5, 10, -1,  2, -1,  1, -1,  0],
    # [-5, 10,  1, 12,  1, -1,  0,  1]
    # ])

    matrix = toBasis(matrix) #at first we need fictive basis
    print(matrix.T)
    matrix_prev = matrix[0]
    iter = 0
    while(hasNegatives(matrix[0])):
        matrix = simplexStep(matrix)
        matrix = removeNegativesFirstCol(matrix)
        print(iter)
        iter += 1
        if matrix_prev.tolist() == matrix[0].tolist():
            break
        matrix_prev = matrix[0]
    
    # for row in matrix:
    #     for elem in row:
    #         print(round(elem, 2), end='\t')
    #     print('\n',end='')

    print(matrix.T)

if __name__ == "__main__":
    main()