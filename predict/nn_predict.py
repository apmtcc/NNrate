import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

torch.multiprocessing.set_sharing_strategy('file_system')

# # assign test set and train set randomly
# file_object = open('sample_data_OH.dat','r')
# file_test = open('file_test_test_validation.dat','w')
# file_train = open('file_train_test_validation.dat','w')
# for line in file_object:
#     t = np.random.randint(1,5)
#     if t == 1 :
#         file_test.write(str(line))
#     else:
#         file_train.write(str(line))
        
# define a train_dataset that reads the above sample data
class Traindataset(Dataset):
    def __init__(self, data_root='sample_data_OH.dat'):
        super(Traindataset, self).__init__()
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

train_dataset = Traindataset(data_root='sample_data_OH.dat')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

# define a test_dataset that reads the above sample data
class Testdataset(Dataset):
    def __init__(self, data_root='file_test_test_c6_1.dat'):
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

test_dataset = Testdataset(data_root='file_test_test_c6_1.dat')
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

# begin training
# nepoch=20
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
# torch.optim.lr_scheduler.StepLR(optimizer,100,gamma=0.2,last_epoch=-1)
loss_function = torch.nn.MSELoss(reduction='mean')

net.load_state_dict(torch.load("saved_weights_oh_c6_3(3).pth"))
# net.eval()

# for epoch in range(nepoch):
#     avg_loss_train = 0
#     prediction_y_train = []
#     x_all_train = []
#     y_all_train = []

    # for itx, data in enumerate(train_loader):

    #     x, y = data
    #     y = y.float()

    #     # clear the gradients caculated before
    #     net.zero_grad()
        
    #     # forward pass
    #     pred_y_train = net(x)
    #     # compute the loss
    #     loss = loss_function(pred_y_train, y)
    #     # backward the loss to compute new gradients
    #     loss.backward()
    #     # use optimizer to apply the gradients on the weights in network
    #     optimizer.step()
        
    #     avg_loss_train += loss.item()

    #     x_all_train.append(x[:,-1])
    #     y_all_train.append(y[:])
    #     # prediction_y_train.append(pred_y[:])
    #     y_hat_train = pred_y_train.data.detach()
    #     prediction_y_train.append(y_hat_train)

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

print("testloss: %.5f"%(avg_loss))

    # if(avg_loss_train<0.0001 and avg_loss<0.0001):
    #     break

# x_all_train = torch.cat(x_all_train,dim=0).view(-1).cpu().numpy()
# y_all_train = torch.cat(y_all_train,dim=0).view(-1).cpu().numpy()
# prediction_y_train = torch.cat(prediction_y_train,dim=1).view(-1).cpu().numpy()

x_all = torch.cat(x_all,dim=0).view(-1).cpu().numpy()
y_all = torch.cat(y_all,dim=0).view(-1).cpu().numpy()
prediction_y = torch.cat(prediction_y,dim=1).view(-1).cpu().numpy()

error_percents = []
for i in range(len(y_all)):
    error_percent = (y_all[i]-prediction_y[i])/y_all[i]
    error_percents.append(error_percent)




# #save data
# with open('data_oh_val_train.dat','w')as sd:
#     for i in range(len(x_all_train)):
#         new_line = str(x_all_train[i])+' '+str(y_all_train[i])+' '+str(prediction_y_train[i])+'\n'
#         sd.write(new_line)
        

with open('data_oh_c6_1(4).dat','w') as sd:
    new_line = ' avg_loss: '+str(avg_loss)+'\n'
    sd.write(new_line)
    for i in range(len(x_all)):
        new_line = str(x_all[i])+' '+str(y_all[i])+' '+str(prediction_y[i])+'\n'
        sd.write(new_line)
        

#plot trainset fitting result
#plt.plot(x_all_train,y_all_train,'r.',label='original data')
#plt.plot(x_all_train,prediction_y_train,'k.',label='predicted value')
#plt.title('trainset fitting result')
#plt.xlabel('temperature(K)')
#plt.ylabel('reaction rate(log) (cm3/molecule s)')
#plt.legend()
#plt.show()

#plot testset fitting result
# plt.plot(x_all,y_all,'r.',label='original data')
# plt.plot(x_all,prediction_y,'k.',label='predicted value')
# plt.title('testset fitting result')
# plt.xlabel('temperature(K)')
# plt.ylabel('reaction rate(log) (cm3/molecule s)')
# plt.legend()
# plt.show()

# #plot testset fitting error
# plt.plot(x_all,error_percents,'b.',label='error')
# plt.title('testset fitting error')
# plt.xlabel('temperature(K)')
# plt.ylabel('reaction rate(log) (pred/orig)')
# plt.legend()

# def to_percent(temp,position):
#     return '%.2f%%'%(100*temp)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
# plt.show()

# save the model:
# torch.save(net.state_dict(), "saved_weights_oh_val(1).pth")
