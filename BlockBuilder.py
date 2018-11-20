import numpy

def buildCell(img,cellSize):
    cells = []
    imgSize = numpy.asarray(img.shape,dtype=float)
    cellNum = numpy.ceil(imgSize/cellSize).astype(numpy.int)
    imgSize = imgSize.astype(numpy.int)
    for i in range(cellNum[0]):
        row = []
        for j in range(cellNum[1]):
            acell = numpy.zeros((1,10))
            for m in range(i*cellSize,min(imgSize[0],(i+1)*cellSize)):
                tmpC = img[m,j*cellSize:min(imgSize[1],(j+1)*cellSize)].astype(numpy.int64)
                h = numpy.bincount(tmpC,minlength=10)
                acell = acell+h
            row.append(acell)
        cells.append(row)
                
               
    return cells,cellNum

def make_blocks(block_size, cells,cellNum):
    aFeatureLen = block_size*block_size*10
    featureDim = (cellNum[0]-block_size+1)*(cellNum[0]-block_size+1)
    block = numpy.zeros((featureDim,aFeatureLen),dtype=numpy.float)
    count = 0
    for i in range(cellNum[0]-block_size+1):
        for j in range(cellNum[0]-block_size+1):
            single = numpy.zeros((1,aFeatureLen))
            
            for m in range(i,min(cellNum[0],i+block_size)):
                for n in range(j,min(cellNum[1],j+block_size)):
                    idx1 = (m-i)*10*block_size+(n-j)*10
                    idx2 = idx1+10
                    single[0][idx1:idx2] = cells[m][n]
            block[count,:] = single
            count = count+1
                
    return block

def normalize_L2_Hys(block, threshold):
    epsilon = 0.00001
    norm = numpy.sqrt(numpy.sum(numpy.power(block,2),axis=1) + epsilon)

    block_aux = block/norm[:,None]
    block_aux[block_aux > threshold] = threshold

    norm = numpy.sqrt(numpy.sum(numpy.power(block_aux,2),axis=1) + epsilon)
    
    return block_aux/norm[:,None]

def buildHist(img,cellSize=10,blockSize=2,clip = 0.3):
    cells,cellNum = buildCell(img,cellSize)
    block = make_blocks(blockSize,cells,cellNum)
    block = normalize_L2_Hys(block,clip)
    return block.flatten()