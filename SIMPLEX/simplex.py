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
    col = mtx[:, 0] / mtx[:, bci]
    similar_rows = min([np.where(col == element)[0].tolist() for element in np.unique(col)], key=lambda x : col[x[0]])

    if (len(similar_rows) == 1):
        return similar_rows[0]
    
    for column in range(1, mtx.shape[1]):
        elems = [mtx[row][column] for row in similar_rows]
        if not np.all(elems == elems[0]):
            minRow = similar_rows[np.argmin(elems)]
            return minRow

def simplexStep(mtx):
    basisColumn = minNegativeElementIndex(mtx[0])
    if basisColumn == 0:
        return mtx
    
    basisRow = lexigraphicMin(mtx, basisColumn)
    
    for i in range(mtx.shape[0]):
        if i == basisRow:
            continue

        rowFactor = mtx[i][basisColumn] / mtx[basisRow][basisColumn]
        mtx[i] -= mtx[basisRow] * rowFactor

    return mtx


def main():
    matrix = np.array([
    [1,  1,  1,  1,  0,  0,  0,  0],
    [5, 10, -1,  2, -1,  1, -1,  0],
    [7, 12,  1, 12,  1, -1,  0,  1]
    ])

    # matrix = np.array([
    # [1,  1,  1,  1,  0,  0,  0,  0],
    # [5, 10, -1,  2, -1,  1, -1,  0],
    # [5, 10,  1, 12,  1, -1,  0,  1]
    # ])

    matrix = toBasis(matrix) #at first we need fictive basis
    while(hasNegatives(matrix[0])):
        matrix = simplexStep(matrix)
        print(matrix[0])
    print(matrix)

if __name__ == "__main__":
    main()