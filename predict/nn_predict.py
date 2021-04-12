import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

torch.multiprocessing.set_sharing_strategy('file_system')


# define a test_dataset that reads the above sample data
class Testdataset(Dataset):
    def __init__(self, data_root='testset.dat'):
        super(Testdataset, self).__init__()
        self.frame = []
        with open(data_root, 'r') as f:
            for line in f:
                values = [float(value) for value in filter(None,line.split(" "))]
                self.frame.append(values)
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        result = self.frame[idx]
        return torch.Tensor(result[:-1]), result[-1]

test_dataset = Testdataset(data_root='testset.dat')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


# define a custom neural network that has 4 fully-connected layers
# can add more
class CustomNN(torch.nn.Module):
    def __init__(self, input_shape=5, output_shape=1):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(input_shape, input_shape*20),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.3),
            torch.nn.Linear(input_shape*20, input_shape*20),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.3),
            torch.nn.Linear(input_shape*20, input_shape*20),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.3),
            torch.nn.Linear(input_shape*20, output_shape),
        )
    def forward(self, input):
        return self.main(input)

net = CustomNN(5, 1)

for i in range(4):

    # begin training
    # nepoch=20
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # torch.optim.lr_scheduler.StepLR(optimizer,100,gamma=0.2,last_epoch=-1)
    loss_function = torch.nn.MSELoss(reduction='mean')

    net.load_state_dict(torch.load("saved_weights%d.pth"%i))
    # net.eval()

    avg_loss = 0
    prediction_y = []
    x_all = []
    y_all = []

    for itx, data in enumerate(test_loader):
        x, y = data
        y = y.float() 

        # forward pass
        pred_y = net(x)
        net.eval()

        # compute the loss
        test_loss = loss_function(pred_y, y)
        avg_loss += test_loss.item()

        x_all.append(x[:,-1])
        y_all.append(y[:])
        y_hat = pred_y.data.detach()
        prediction_y.append(y_hat)
        
    avg_loss /= len(test_loader)
    # avg_loss_train /= len(train_loader)

    x_all = torch.cat(x_all,dim=0).view(-1).cpu().numpy()
    y_all = torch.cat(y_all,dim=0).view(-1).cpu().numpy()
    prediction_y = torch.cat(prediction_y,dim=1).view(-1).cpu().numpy()

    # #save data
    # with open('data_oh_val_train.dat','w')as sd:
    #     for i in range(len(x_all_train)):
    #         new_line = str(x_all_train[i])+' '+str(y_all_train[i])+' '+str(prediction_y_train[i])+'\n'
    #         sd.write(new_line)
            

    with open('data_predict(%d).dat'%i,'w') as sd:
        for i in range(len(x_all)):
            new_line = str(x_all[i])+' '+str(prediction_y[i])+'\n'
            sd.write(new_line)

# scheme
path = os.getcwd()

datapath0 = path + '/data_predict(0).dat'
datapath1 = path + '/data_predict(1).dat'
datapath2 = path + '/data_predict(2).dat'
datapath3 = path + '/data_predict(3).dat'
x_data = []
y_data_0 = []
y_data_1 = []
y_data_2 = []
y_data_3 = []
# train_min
with open(datapath0, 'r') as f:
    for line in f:
        values = [float(value) for value in filter(None,line.split(" "))]
        x_data.append(values[0])
        y_data_0.append(values[1])
with open(datapath1, 'r') as f:
    for line in f:
        values = [float(value) for value in filter(None,line.split(" "))]
        y_data_1.append(values[1])
with open(datapath2, 'r') as f:
    for line in f:
        values = [float(value) for value in filter(None,line.split(" "))]
        y_data_2.append(values[1])
with open(datapath3, 'r') as f:
    for line in f:
        values = [float(value) for value in filter(None,line.split(" "))]
        y_data_3.append(values[1])

# scheme 3
y_data_avg1 = []
for j in range(len(y_data_0)):
    y_temp = (y_data_0[j] + y_data_1[j] + y_data_2[j]) / 3
    y_data_avg1.append(y_temp)

# scheme 4
y_data_avg3 = []
for j in range(len(y_data_0)):
    y_temp_avg3 = (y_data_0[j] + y_data_1[j] + y_data_2[j] + y_data_3[j]) / 4
    y_data_avg3.append(y_temp_avg3)

# scheme 5
y_data_avg2 = []
std_avg2_0 = []
std_avg2_1 = []
std_avg2_2 = []
std_avg2_3 = []
for j in range(len(y_data_1)):
    arr_std_0 = np.std([y_data_0[j], y_data_1[j], y_data_2[j]], ddof = 1)
    std_avg2_0.append(arr_std_0)
    arr_std_1 = np.std([y_data_0[j], y_data_1[j], y_data_3[j]], ddof = 1)
    std_avg2_1.append(arr_std_1)
    arr_std_2 = np.std([y_data_0[j], y_data_2[j], y_data_3[j]], ddof = 1)
    std_avg2_2.append(arr_std_2)
    arr_std_3 = np.std([y_data_1[j], y_data_2[j], y_data_3[j]], ddof = 1)
    std_avg2_3.append(arr_std_3)

avg_std_0 = 0
avg_std_1 = 0
avg_std_2 = 0
avg_std_3 = 0
for j in range(len(std_avg2_0)):
    avg_std_0 += std_avg2_0[j]
    avg_std_1 += std_avg2_1[j]
    avg_std_2 += std_avg2_2[j]
    avg_std_3 += std_avg2_3[j]

avg_std_0 /= len(std_avg2_0)
avg_std_1 /= len(std_avg2_1)
avg_std_2 /= len(std_avg2_2)
avg_std_3 /= len(std_avg2_3)

if(avg_std_0 <= avg_std_1 and avg_std_0 <= avg_std_2 and avg_std_0 <= avg_std_3):
    for j in range(len(y_data_0)):
        y_temp_avg2 = (y_data_0[j] + y_data_1[j] + y_data_2[j]) / 3
        y_data_avg2.append(y_temp_avg2)
elif(avg_std_1 <= avg_std_0 and avg_std_1 <= avg_std_2 and avg_std_1 <= avg_std_3):
    for j in range(len(y_data_0)):
        y_temp_avg2 = (y_data_0[j] + y_data_1[j] + y_data_3[j]) / 3
        y_data_avg2.append(y_temp_avg2)
elif(avg_std_2 <= avg_std_0 and avg_std_2 <= avg_std_1 and avg_std_2 <= avg_std_3):
    for j in range(len(y_data_0)):
        y_temp_avg2 = (y_data_0[j] + y_data_2[j] + y_data_3[j]) / 3
        y_data_avg2.append(y_temp_avg2)
elif(avg_std_3 <= avg_std_0 and avg_std_3 <= avg_std_1 and avg_std_3 <= avg_std_2):
    for j in range(len(y_data_0)):
        y_temp_avg2 = (y_data_1[j] + y_data_2[j] + y_data_3[j]) / 3
        y_data_avg2.append(y_temp_avg2)

# write to file
with open('predict_result.dat','w') as sd:
    new_line = 'temperature   ' + '1(scheme2)   ' + '2    ' + '3    ' + '4    ' + 'scheme3   ' + 'scheme4   ' + 'scheme5   ' + '\n'
    sd.write(new_line)
    for i in range(len(y_data_avg1)):
        new_line = str(x_data[i]) + ' ' + str(y_data_0[i]) + ' ' + str(y_data_1[i]) + ' ' + str(y_data_2[i]) + ' ' + str(y_data_3[i])+ ' ' + str(y_data_avg1[i]) + ' ' + str(y_data_avg3[i]) + ' ' + str(y_data_avg2[i]) + '\n'
        sd.write(new_line)