import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from net import VTSNN_IF, VTSNN_LIF

# ============= 1) UWE (Undistorted Weighted Encoder) ============== #


def _toBinary(num):
    res = str(bin(num))
    res = res[2:]
    long = '0' * (8 - len(res))
    res = long + res
    return res


def UWE(img):

    img = np.array(img)
    img_shape = img.shape
    arr = np.zeros((8, *img_shape))

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            cnt = img[i, j]
            b_cnt = _toBinary(cnt)
            for k in range(8):
                arr[k, i, j] = int(b_cnt[k])

    return arr

# ============= 2) UWD (Undistorted Weighted Decoder) ================ #


def UWD(img, DEVICE, require_grad=True):

    alpha = [[128., 64., 32., 16., 8., 4., 2., 1.]]
    alpha = torch.tensor(alpha, dtype=torch.double).to(DEVICE)
    batch_size, T, H, W = img.shape
    output = torch.zeros((batch_size, 1, H, W), requires_grad=require_grad).to(DEVICE)
    for num in range(batch_size):
        img_num = img[num, :, :, :]
        img_num = img_num.reshape(T, H*W)
        img_num = torch.tensor(img_num, dtype=torch.double)
        img_num_coded = torch.matmul(alpha, img_num)
        img_output = img_num_coded.reshape(1, H, W)
        output[num, :, :, :] = img_output
    return output


def UWD_prediction(img):

    alpha = np.array([[128, 64, 32, 16, 8, 4, 2, 1]], dtype=np.float32)
    T, W, H = img.shape

    img = img.detach().numpy().astype(np.float32)
    img = img.reshape(T, W * H)

    img_output = np.dot(alpha, img)
    img_output = img_output.reshape(W, H)
    return img_output


# ===================== 3) Data Process  ============================== #

class MyDataset(Dataset):
    def __init__(self, data1, data2):
        data1 = np.array(data1, dtype='float32')
        data2 = np.array(data2, dtype='float32')
        txt_data = np.concatenate((data1, data2), axis=1)
        self._x = torch.tensor(data1)
        self._y = torch.tensor(data2)
        self._len = len(txt_data)

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return self._len


def load_data(noise_factor=0.2, dataset='MNIST', BATCH_SIZE_train=50, BATCH_SIZE_test=50):

    base_path = './Datasets/' + dataset
    # input_train1 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(1) + '.npy')
    # input_train2 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(2) + '.npy')
    # input_train3 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(3) + '.npy')
    # input_train4 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(4) + '.npy')
    # input_train5 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(5) + '.npy')
    # input_train6 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(6) + '.npy')
    input_train1 = np.load(base_path + '/train_label' + '_' + str(1) + '.npy')
    input_train2 = np.load(base_path + '/train_label' + '_' + str(2) + '.npy')
    input_train3 = np.load(base_path + '/train_label' + '_' + str(3) + '.npy')
    input_train4 = np.load(base_path + '/train_label' + '_' + str(4) + '.npy')
    input_train5 = np.load(base_path + '/train_label' + '_' + str(5) + '.npy')
    input_train6 = np.load(base_path + '/train_label' + '_' + str(6) + '.npy')

    label_train1 = np.load(base_path + '/train_label' + '_' + str(1) + '.npy')
    label_train2 = np.load(base_path + '/train_label' + '_' + str(2) + '.npy')
    label_train3 = np.load(base_path + '/train_label' + '_' + str(3) + '.npy')
    label_train4 = np.load(base_path + '/train_label' + '_' + str(4) + '.npy')
    label_train5 = np.load(base_path + '/train_label' + '_' + str(5) + '.npy')
    label_train6 = np.load(base_path + '/train_label' + '_' + str(6) + '.npy')

    # label_train1 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(1) + '.npy')
    # label_train2 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(2) + '.npy')
    # label_train3 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(3) + '.npy')
    # label_train4 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(4) + '.npy')
    # label_train5 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(5) + '.npy')
    # label_train6 = np.load(base_path + '/train_' + str(noise_factor) + '_' + str(6) + '.npy')

    input_train = np.concatenate((input_train1, input_train2, input_train3, input_train4,
                                  input_train5, input_train6), axis=0)
    label_train = np.concatenate((label_train1, label_train2, label_train3, label_train4,
                                  label_train5, label_train6), axis=0)

    # input_test = np.load(base_path + '/test_' + str(noise_factor) + '.npy')
    input_test = np.load(base_path + '/test_label.npy')
    label_test = np.load(base_path + '/test_label.npy')
    # label_test = np.load(base_path + '/test_' + str(noise_factor) + '.npy')

    train_set = MyDataset(input_train, label_train)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_train, shuffle=True)

    test_set = MyDataset(input_test, label_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_test, shuffle=True)

    return train_loader, test_loader


# ===================== 4) Prediction ============================== #

def mse(arr1, arr2, img_size=28):
    arr = (arr1/255 - (arr2/255)) ** 2 / (img_size*img_size)
    arr = arr.sum()
    return arr


def prediction(num=0, last_layer_threshold=0.077, net=VTSNN_IF):

    model = net(last_layer_threshold=last_layer_threshold)
    model.eval()
    T = 8
    path = './Weights/weight.pt'
    model_CKPT = torch.load(path)
    model.load_state_dict(model_CKPT, strict=False)

    path_input = './Datasets/MNIST/test_label.npy'
    path_label = './Datasets/MNIST/test_label.npy'

    output = np.zeros((1, 8, 28, 28), dtype=np.float32)
    output = torch.tensor(output)

    input = np.load(path_input).astype('float32')
    label = np.load(path_label).astype('float32')
    input = torch.tensor(input)
    label = torch.tensor(label)
    input = input[num, :, :, :].unsqueeze(0)
    label = label[num, :, :, :].unsqueeze(0)

    for i in range(T):
        output[:, i, :, :] = model(input[:, i, :, :].unsqueeze(1))

    input_end = UWD_prediction(input[0, :, :, :])
    output_end = UWD_prediction(output[0, :, :, :])
    label_end = UWD_prediction(label[0, :, :, :])

    error = mse(input_end, output_end, img_size=28)
    plt.suptitle('loss {:.6f}'.format(error), fontsize=18)
    plt.subplot(1, 3, 1)
    plt.title('input')
    input_end = 255 - input_end
    plt.imshow(input_end, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('output')
    # output = output_end.detach().numpy()
    output_end = 255 - output_end
    plt.imshow(output_end, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('label')
    label_end = 255 - label_end
    plt.imshow(label_end, cmap='gray')
    plt.show()
prediction()
