from dataset import FarmDataset
from model import SFNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt

batch_size = 1024
device = torch.device('cpu')
network = SFNet()
pre_trained = True
pre_trained_sample = 135
wind_speed = 9


if __name__ == '__main__':
    network.load_state_dict(torch.load('./checkpoint/' + network.name + str(pre_trained_sample) + '.pth'))
    network.to(device).double()
    resolution = [30, 50]

    wind_data_test = FarmDataset(resolution=resolution, wind_speed=wind_speed)

    test_loader = DataLoader(wind_data_test, batch_size=batch_size)

    network.eval()
    for _, data in enumerate(test_loader):
        low_flow_field = Variable(data).to(device)

        prediction = network(low_flow_field).detach()

        prediction = wind_data_test.ss_high.inverse_transform(prediction.reshape(-1, 1500)) + wind_data_test.mean_high
        low_flow_field = wind_data_test.ss_low.inverse_transform(low_flow_field.reshape(-1, 1500)) + \
                          wind_data_test.mean_low

    vmin = 2
    vmax = 12
    x_dimension = np.zeros((300, 500))
    y_dimension = np.zeros((300, 500))
    y_unit = np.linspace(-189.6, 126.4 * 30.5, 300)
    for i in range(300):
        x_dimension[i] = np.linspace(-126.4, 126.4 * 50, 500)
        y_dimension[i] = [y_unit[i]] * 500

    meshx, meshy = x_dimension.reshape(300, 500), y_dimension.reshape(300, 500)
    low_field_whole = np.zeros((300, 500))
    high_field_whole = np.zeros((300, 500))
    for index in range(100):
        low_field_whole[30 * (index // 10):30 * (index // 10 + 1), 50 * (index % 10):50 * (index % 10 + 1)] = \
            low_flow_field[index].reshape((30, 50))
        high_field_whole[30 * (index // 10):30 * (index // 10 + 1), 50 * (index % 10):50 * (index % 10 + 1)] = \
            prediction[index].reshape((30, 50))

    fig = plt.figure(figsize=(40, 30))
    v = np.round(np.linspace(vmin, vmax, 51), decimals=2)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.contourf(meshx, meshy, np.clip(high_field_whole, vmin, vmax), v, cmap='jet')
    cbar_ax = fig.add_axes(([0.95, 0.15, 0.01, 0.7]))
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 50,
            }
    cbar_ax.set_title('[m/s]', fontdict=font, loc='right')
    cbar_ax.tick_params(labelsize=35)
    plt.colorbar(cax=cbar_ax, ticks=np.arange(2, 12.1, 2))
    plt.show()

    fig = plt.figure(figsize=(40, 30))
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    v = np.round(np.linspace(vmin, vmax, 51), decimals=2)
    plt.contourf(meshx, meshy, np.clip(low_field_whole, vmin, vmax), v, cmap='jet')
    cbar_ax = fig.add_axes(([0.95, 0.15, 0.01, 0.7]))
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 50,
            }
    cbar_ax.set_title('[m/s]', fontdict=font, loc='right')
    cbar_ax.tick_params(labelsize=35)
    plt.colorbar(cax=cbar_ax, ticks=np.arange(2, 12.1, 2))
    plt.show()

