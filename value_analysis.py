import numpy as np
import matplotlib.pyplot as plt
import xlwt


if __name__ == '__main__':
    interValues_train = np.load('training_set_neuron_outputs.npy')
    interValues_test = np.load('test_set_neuron_outputs.npy')
    labels_train = np.load('training_set_labels.npy')
    weights_fc2 = np.load('weights_fc2.npy')
    weights_fc3 = np.load('weights_fc3.npy')
    # vc = []
    # for i in range(10):
    #     vc.append(1)
    #     vc[i] = []
    #
    # for i in range(interValues_train.shape[0]):
    #     if labels_train[i] == 0:
    #         vc[0].append(interValues_train[i])
    #     elif labels_train[i] == 1:
    #         vc[1].append(interValues_train[i])
    #     elif labels_train[i] == 2:
    #         vc[2].append(interValues_train[i])
    #     elif labels_train[i] == 3:
    #         vc[3].append(interValues_train[i])
    #     elif labels_train[i] == 4:
    #         vc[4].append(interValues_train[i])
    #     elif labels_train[i] == 5:
    #         vc[5].append(interValues_train[i])
    #     elif labels_train[i] == 6:
    #         vc[6].append(interValues_train[i])
    #     elif labels_train[i] == 7:
    #         vc[7].append(interValues_train[i])
    #     elif labels_train[i] == 8:
    #         vc[8].append(interValues_train[i])
    #     else:
    #         vc[9].append(interValues_train[i])
    #
    # for i in range(10):
    #     vc[i] = np.array(vc[i])
    #     print(vc[i].shape)
    #     print(np.max(vc[i]), np.min(vc[i]))
    #
    # print('class0 max:{}, min:{}'.format(np.max(vc[0], axis=0), np.min(vc[0], axis=0)))
    # print('class1 max:{}, min:{}'.format(np.max(vc[1], axis=0), np.min(vc[1], axis=0)))
    # print('class2 max:{}, min:{}'.format(np.max(vc[2], axis=0), np.min(vc[2], axis=0)))
    # print('class3 max:{}, min:{}'.format(np.max(vc[3], axis=0), np.min(vc[3], axis=0)))
    # print('class4 max:{}, min:{}'.format(np.max(vc[4], axis=0), np.min(vc[4], axis=0)))
    # print('class5 max:{}, min:{}'.format(np.max(vc[5], axis=0), np.min(vc[5], axis=0)))
    # print('class6 max:{}, min:{}'.format(np.max(vc[6], axis=0), np.min(vc[6], axis=0)))
    # print('class7 max:{}, min:{}'.format(np.max(vc[7], axis=0), np.min(vc[7], axis=0)))
    # print('class8 max:{}, min:{}'.format(np.max(vc[8], axis=0), np.min(vc[8], axis=0)))
    # print('class9 max:{}, min:{}'.format(np.max(vc[9], axis=0), np.min(vc[9], axis=0)))
    #
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('Sheet1')
    # print(weights_fc3.shape)
    for i in range(84):
        for j in range(interValues_train.shape[0]):
            sheet.write(i, j, str(interValues_train[j][i]))
    workbook.save('errors.xls')
