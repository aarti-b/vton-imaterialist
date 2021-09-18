import numpy
import imageio
import glob
import sys
import os
import random

height = 0
width = 0

dstPath = "./convert_imaterialist"

if not os.path.exists(dstPath):
    os.makedirs(dstPath)

testLabelPath = dstPath+"/vton-test-labels-idx3-ubyte"   #for imaterialist dataset use imat19 inplace of vton
testImagePath = dstPath+"/vton-test-images-idx3-ubyte"
trainLabelPath = dstPath+"/vton-train-labels-idx3-ubyte"
trainImagePath = dstPath+"/vton-train-images-idx3-ubyte"

def get_path_labels_images(folder, number=0):
    # Make a list of lists of files for each label
    filelist = []
    labelfilelist=[]
    #path to images stored
    for file in os.listdir('path-to-training-cloth'):
        if (file.endswith('.jpg')):
            fullname = os.path.join('path-to-training-cloth', file)
            if (os.path.getsize(fullname) > 0):
                filelist.append(fullname)
            else:
                print('file ' + fullname + ' is empty')

        if (file.endswith('.jpg')):
            fullname = os.path.join('path-to-training-clothmask', file)
            if (os.path.getsize(fullname) > 0):
                labelfilelist.append(fullname)
            else:
                print('file ' + fullname + ' is empty')

    zipped= list(zip(filelist, labelfilelist))

    return zipped

def images_labels_arrays(labelsImagesFiles, ratio):

    global height, width
    images = []
    labels = []
    
    imShape = imageio.imread(labelsImagesFiles[0][0]).shape
    lblShape = imageio.imread(labelsImagesFiles[0][1]).shape
    print('image shape',imShape)
    print('label shape',lblShape)
    if len(imShape) > 2:
        height, width, img_channels = imShape
    else:
        height, width = imShape
        img_channels = 1

    if len(lblShape) > 2:
        height, width, lbl_channels = lblShape
    else:
        height, width = lblShape
        lbl_channels = 1
    for i in range(0, len(labelsImagesFiles)):
        # display progress, since this can take a while
        if (i % 100 == 0):
            sys.stdout.write("\r%d%% complete" %
                             ((i * 100) / len(labelsImagesFiles)))
            sys.stdout.flush()

        filename = labelsImagesFiles[i][0]
        labelfilename=labelsImagesFiles[i][1]
        try:
            image = imageio.imread(filename)
            label=imageio.imread(labelfilename)
            images.append(image)
            labels.append(label)
        except:
            # If this happens we won't have the requested number
            print("\nCan't read image file " + filename)
    print(len(images))
    if ratio == 'train':
        ratio = 0
    elif ratio == 'test':
        ratio = 1
    else:
        ratio = float(ratio) / 100
    count = len(images)
    trainNum = int(count * (1 - ratio))
    testNum = int(count - trainNum)
    print(trainNum)
    print(testNum)
    if img_channels > 1:
        trainImagedata = numpy.zeros(
            (trainNum, height, width, img_channels), dtype=numpy.uint8)
        testImagedata = numpy.zeros(
            (testNum, height, width, img_channels), dtype=numpy.uint8)
        #print(testImagedata)
    else:
        trainImagedata = numpy.zeros(
            (trainNum, height, width), dtype=numpy.uint8)
        testImagedata = numpy.zeros(
            (testNum, height, width), dtype=numpy.uint8)

    if lbl_channels > 1:
        trainLabeldata = numpy.zeros(
            (trainNum, height, width, lbl_channels), dtype=numpy.uint8)
        testLabeldata = numpy.zeros(
            (testNum, height, width, lbl_channels), dtype=numpy.uint8)
    else:
        trainLabeldata = numpy.zeros(
            (trainNum, height, width), dtype=numpy.uint8)
        testLabeldata = numpy.zeros(
            (testNum, height, width), dtype=numpy.uint8)

    for i in range(trainNum):

        trainImagedata[i] = images[i]
        trainLabeldata[i] = labels[i]

    for i in range(0, testNum):
        #print(images[trainNum + i].shape)
        testImagedata[i] = images[trainNum + i]
        testLabeldata[i] = labels[trainNum + i]
    print("\n")
    return trainImagedata, trainLabeldata, testImagedata, testLabeldata

def write_labeldata(labeldata, outputfile):
    global height, width
    header = numpy.array([0x0803, len(labeldata), height, width], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(labeldata.tobytes())

def write_imagedata(imagedata, outputfile):
    global height, width
    header = numpy.array([0x0803, len(imagedata), height, width], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(imagedata.tobytes())

def main(argv):
    #global idxLabelPath, idxImagePath

    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    if len(argv) is 3:
        labelsImagesFiles= get_path_labels_images(argv[1])
    elif len(argv) is 4:
        labelsImagesFiles= get_path_labels_images(argv[1], int(argv[3]))
    random.shuffle(labelsImagesFiles)

    trainImagedata, trainLabeldata, testImagedata, testLabeldata = images_labels_arrays(
        labelsImagesFiles, argv[2])
    print(testImagedata.shape)


    if argv[2] == 'train':
        write_labeldata(trainLabeldata, trainLabelPath)
        write_imagedata(trainImagedata, trainImagePath)
    elif argv[2] == 'test':
        write_labeldata(testLabeldata, testLabelPath)
        write_imagedata(testImagedata, testImagePath)
    else:
        write_labeldata(trainLabeldata, trainLabelPath)
        write_imagedata(trainImagedata, trainImagePath)
        write_labeldata(testLabeldata, testLabelPath)
        write_imagedata(testImagedata, testImagePath)


if __name__ == '__main__':
    main(sys.argv)

#reference - https://github.com/Arlen0615/Convert-own-data-to-MNIST-format/tree/47ee4d06ea9af120b276aa5d46d6b8204111a9c1
