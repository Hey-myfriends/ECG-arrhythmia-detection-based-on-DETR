import numpy as np
import wfdb

rootpath = "d:\\Desktop\\ECG分类研究/mit-bih-arrhythmia-database-1.0.0/"
# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data):
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord(rootpath + number, channel_names=['MLII'])  #源文件都放在ecg_data这个文件夹中了
    data = record.p_signal.flatten()
    #data=np.array(data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(rootpath + number, 'atr')
    Rlocation = annotation.sample  #对应位置
    Rclass = annotation.symbol  #对应标签
    print(set(Rclass))

    X_data.append(data)

    return

def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    for n in numberSet:
        getDataSet(n, dataSet)
    return dataSet

def main():
    dataSet = loadData()
    dataSet = np.array(dataSet)
    print(dataSet.shape)
    print("data ok!!!")

if __name__ == '__main__':
    main()
