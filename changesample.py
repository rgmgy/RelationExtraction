# -*- coding: UTF-8 -*-
# 选出相关症状的句子，调换两个实体在句子头部的，调换两个实体在句子中的位子，将改变后的<entity1,entity2,newrelation,sentence>存到数组，最后把数组写入一个txt，合并两个txt。
# 选出相关症状的句子，en2 id小于en1 ID的改变为相关症状2，调换en1 和en2 的位置，保存到新的训练文件
def changepos():

    print('reading relation to id')
    relation2id = {}
    f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

    print('reading train data...')
    f = open('./origin_data/RETrainData.txt', 'r', encoding='utf-8')
    newf = open('./origin_data/newtrain', 'w', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split()
        print(content)
        if len(content) != 4:
            raise ValueError('the length of content is wrong')

        # get entity name
        en1 = content[0]
        en2 = content[1]
        sentence = content[3]

        if content[2] == '相关症状' or content[2] == '致病原因':

            id1 = sentence.index(en1)
            id2 = sentence.index(en2)
            if content[2] == '相关症状':
                if id1 > id2:
                    trelation = '症状疾病'
                    newf.write(en2 + '\t' + en1 + '\t' + trelation + ' ' + sentence + '\n')
                else:
                    trelation = '疾病症状'
                    newf.write(en1 + '\t' + en2 + '\t' + trelation + ' ' + sentence + '\n')
            else:

                if id1 > id2:
                    trelation = '疾病病因'
                    newf.write(en2 + '\t' + en1 + '\t' + trelation + ' ' + sentence + '\n')
                else:
                    trelation = '病因疾病'
                    newf.write(en1 + '\t' + en2 + '\t' + trelation + ' ' + sentence + '\n')
        else:
            newf.write(en1 + '\t' + en2 + '\t' + content[2] + ' ' + sentence + '\n')
        f.close()

#疾病症状，症状疾病，间接关系，歧义，不明确
def delete_blank():
    print('reading train data...')
    f = open('./origin_data/RETRainDataNew', 'r', encoding='utf-8')
    newf = open('./origin_data/tempfile', 'w', encoding='utf-8')
    mLines = f.readlines()
    Length = len(mLines)
    last = mLines[-1]
    print(last)
    f.close()
    print(Length)
    curL = 1
    f = open('./origin_data/RETRainDataNew', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        #print(content)
        if content == '':
            break
        else:
            #print(content)
            tcontent = content.strip().split()
            print(tcontent)
            if tcontent[2] == '疾病病因' or tcontent[2] == '病因疾病' or tcontent[2] == '间接关系因':
                newf.write(content)
            curL += 1

    f.close()

delete_blank()



















