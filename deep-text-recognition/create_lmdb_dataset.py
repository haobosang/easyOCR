""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
import csv

import fire
import os
import lmdb
import cv2

import numpy as np
from PIL import Image


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath="./output", gtFile="./output/gt.txt", outputPath="./output", checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    # print('Created dataset with %d samples' % nSamples)
    print('[ PM ] Created dataset to ./result with %d samples' % nSamples)


def readDataset(path="./lmdb/training/MJ/"):
    env = lmdb.open(path, readonly=True)

    # 打开事务
    with env.begin() as txn:
        # 打开游标
        cursor = txn.cursor()
        # 遍历数据库
        for key, value in cursor:
            # 解码键和值
            key = key.decode('utf-8')
            value = np.frombuffer(value, dtype=np.uint8)
            # 处理数据
            print(key, value)


def InvoiceDataset(root="~/pm"):
    """
    这个模块用于实现单张图片文本分块，用于训练模型前的数据加载。

    作者：潘锰
    日期：2023-05-18

    :param root project root dir
    """
    gts_list = []
    fieldnames = ["image_name", "left", "top", "right", "bottom", "text"]
    with open("{}/fakeLabel/label.csv".format(root), "r", newline="", encoding="Shift_JIS") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        for i, row in enumerate(reader):
            if i > 0:
                gts_list.append(row)  # row 为字典
        f.close()

    nSamples = len(gts_list)

    label_list = []
    for index in range(nSamples):
        data = dict(gts_list[index])
        image_name = data["image_name"]
        left = int(data["left"])
        top = int(data["top"])
        right = int(data["right"])
        bottom = int(data["bottom"])
        text = data["text"]

        img = Image.open("{}/fakeImage/{}.png".format(root, image_name)).convert('RGB')
        img_block = img.crop((left, top, right, bottom))

        img_block.save("./output/images/image_{:05d}.png".format(index))
        new_img_name = "images/image_{:05d}.png".format(index)
        print("{}\t{}".format(new_img_name, text))
        label_list.append(np.array("{}\t{}".format(new_img_name, text)))

    np.savetxt("./output/gt.txt", np.array(label_list), delimiter=' ', fmt='%s')

    print("[ PM ] cropped from ../fakeImage/ to ./output/images/*.png !")
    print("[ PM ] created ./output/gt.txt !")


if __name__ == '__main__':
    # InvoiceDataset("..")
    fire.Fire(createDataset)
    # fire.Fire(readDataset)
    
    # for file in $(ls | grep 'word_'); do
    # mv "$file" "${file/word_/image_00}"
    # done
