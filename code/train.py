import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from spikingjelly.clock_driven import functional
from torch.utils.tensorboard import SummaryWriter
from utils import load_data
from net import VTSNN_IF, VTSNN_LIF
import numpy as np

def binarydecoding_faster(img,DEVICE):
    alpha = [[128., 64., 32., 16., 8., 4., 2., 1.]]
    alpha = torch.tensor(alpha,dtype=torch.double).to(DEVICE)
    batch_size, T, H, W = img.shape
    output = torch.zeros((batch_size, 1, H, W),requires_grad=True).to(DEVICE)
    for num in range(batch_size):
        img_num = img[num, :, :, :]
        img_num = img_num.reshape(T, H*W)
        img_num = torch.tensor(img_num, dtype=torch.double)
        img_num_coded = torch.matmul(alpha, img_num)
        img_output = img_num_coded.reshape(1, H, W)
        output[num, :, :, :] = img_output
    return output

def train(epoch):
    model.train()
    for input_train, label_train in train_loader:
        losses = []
        output_end = []
        input_end = []
        for i in range(T):
            input_train, label_train = input_train.to(DEVICE), label_train.to(DEVICE)
            optimizer.zero_grad()
            output = model(input_train[:, i, :, :].unsqueeze(1))
            loss_T = (2 ** (7 - i)) * F.mse_loss(output, label_train[:, i, :, :].unsqueeze(1))
            losses.append(loss_T)
            input_end.append(input_train)
            output_end.append(output)
        output_end_2 = torch.cat([output_end[0], output_end[1], output_end[2], output_end[3],
                                      output_end[4], output_end[5], output_end[6], output_end[7]], dim=1)

        output_end_3 = binarydecoding_faster(output_end_2, DEVICE='cuda')
        input_end_2 = torch.cat([input_end[0], input_end[1], input_end[2], input_end[3],
                                  input_end[4], input_end[5], input_end[6], input_end[7]], dim=1)

        input_end_3 = binarydecoding_faster(input_end_2, DEVICE='cuda')
        loss = sum(losses)
        writer.add_scalars("train_loss", {"Train": loss.item()}, epoch)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)
    loss3=F.mse_loss(output_end_3,input_end_3)
    # print(loss3.item())
    plt.suptitle('train:{}'.format(loss3.item()))
    plt.subplot(1,2,1)
    plt.title('label')
    plt.imshow(input_end_3[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('output')
    plt.imshow(output_end_3[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.show()



def test():
    model.eval()
    with torch.no_grad():
        for input_test, label_test in test_loader:
            losses = []
            output_end=[]
            input_end = []
            for i in range(T):
                input_test, label_test = input_test.to(DEVICE), label_test.to(DEVICE)
                output = model(input_test[:, i, :, :].unsqueeze(1))
                output_end.append(output)
                loss_T = (2 ** (7 - i)) * F.mse_loss(output, label_test[:, i, :, :].unsqueeze(1))
                losses.append(loss_T)
                input_end.append(input_test)

            output_end_2 = torch.cat([output_end[0],output_end[1],output_end[2],output_end[3],
                                           output_end[4],output_end[5],output_end[6],output_end[7]], dim=1)

            output_end_3 = binarydecoding_faster(output_end_2, DEVICE='cuda')

        input_end_2 = torch.cat([input_end[0], input_end[1], input_end[2], input_end[3],
                                 input_end[4], input_end[5], input_end[6], input_end[7]], dim=1)

        input_end_3 = binarydecoding_faster(input_end_2, DEVICE='cuda')
        loss3 = F.mse_loss(output_end_3, input_end_3)
        plt.suptitle('test:{}'.format(loss3.item()))
        plt.subplot(1, 2, 1)
        plt.title('label')
        plt.imshow(input_end_3[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('output')
        plt.imshow(output_end_3[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
        plt.show()

        # writer.add_scalars("test_loss", {"Test": loss.item()}, epoch)


if __name__ == '__main__':

    n_epochs = 60      # epochs for training
    BATCH_SIZE_train = 50   # batch size for training
    BATCH_SIZE_test = 50    # batch size for testing
    T = 8               # Time Step
    noise_factor = 0.2  # range from [0.2, 0.4, 0.6, 0.8]

    train_loader, test_loader = load_data(noise_factor=noise_factor, dataset='MNIST',
                                          BATCH_SIZE_train=BATCH_SIZE_train, BATCH_SIZE_test=BATCH_SIZE_test)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
    model = VTSNN_LIF(last_layer_threshold=0.077).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-05)
    # tensorboard - -logdir = D:\opencv-python\AAAI\snn_codes\runs\Sep28_11-47-30_LAPTOP-85DL9FARtest_your_comment
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
        print(epoch)
    torch.save(model.state_dict(), 'VTSNN_IF_restruction.pt')

