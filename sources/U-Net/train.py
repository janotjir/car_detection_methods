import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from unet_model import SmallerUnet


parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32, help='input batch size')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--lr', type=float, default=0.001, help="dataset path")
parser.add_argument('--optim', default='adam', help="optimizer for backprop")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum for optimizer")
parser.add_argument('--weight_decay', type=float, default=0, help="weight_decay for optimizer")
parser.add_argument('--weight', type=float, default=1, help="weight of loss for noise label")
parser.add_argument('--gpu', type=int, default=-1, help="specify gpu")
opt = parser.parse_args()


def save_model(model, destination):
    torch.save(model.state_dict(), destination, _use_new_zipfile_serialization=False)


def load_model():
    model = SmallerUnet()
    model.load_state_dict(torch.load(opt.model,map_location='cpu'))
    return model


def get_device(gpu=0):  # Manually specify gpu
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = 'cpu'

    return device


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    index = np.argmax(memory_available)
    return int(index)  # Returns index of the gpu with the most memory available


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_file, label_file):
        super().__init__()
        self.data = torch.load(data_file)
        self.labels = torch.load(label_file)
        print(self.data.shape, self.labels.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return {
            'labels': np.asarray(self.labels[i], dtype=int),
            'data': np.asarray(self.data[i], dtype=np.float32),
            'key': i
        }


if __name__ == '__main__':
    if opt.gpu != -1:
        device = get_device(opt.gpu)
    else:
        device = get_device(get_free_gpu())

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    log_file = '%s/log.txt' % opt.outf
    f = open(log_file, 'a+')
    f.write("bs: {}, n_epochs: {}, lr: {}, optim: {}, momentum: {}, weight_dec: {},weight: {}\n".format(opt.bs, opt.nepoch, opt.lr, opt.optim,opt.momentum, opt.weight_decay, opt.weight))
    f.close()

    #----load datasets----
    trn_dataset = Dataset("trn_data_UNet.pt", "trn_labels_UNet.pt")
    val_dataset = Dataset("tst_data_UNet.pt", "tst_labels_UNet.pt")
    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=opt.bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.bs, shuffle=True)

    #----init model----
    if opt.model != '':
        model = load_model().to(device)
    else:
        model = SmallerUnet().to(device)

    #----init optimizer----
    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    class_weight = torch.tensor((opt.weight, 1.), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)

    #------TRAINING LOOP------
    print("Training started")
    for epoch in range(opt.nepoch):
        print("Epoch: {} started".format(epoch))
        model.train()
        loss_list = []
        loss_val = []
        TP_t = FP_t = TN_t = FN_t = 0
        #TRAINING
        for it, batch in enumerate(trn_loader):
            input_data = batch['data'].to(device).requires_grad_()
            label = batch['labels'].to(device)

            predictions = model(input_data)
            loss = criterion(predictions, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())

            pred = torch.argmax(predictions, dim=1).to(device)
            conf_vector = pred/label
            TP_t += torch.sum(conf_vector == 1).item()  # pred=1 and label=1
            FP_t += torch.sum(conf_vector == float('inf')).item()  # pred=1 and label=0
            TN_t += torch.sum(torch.isnan(conf_vector)).item()  # pred=0 and label=0
            FN_t += torch.sum(conf_vector == 0).item()  # pred=0 and label=1

        #print(TP_t, FP_t, TN_t, FN_t, TP_t/(TP_t+FP_t), TP_t/(TP_t+FN_t))

        #VALIDATION
        TP = FP = TN = FN = 0
        with torch.no_grad():
            model.eval()
            #confusion_matrix = np.zeros((2, 2))
            for it_val, batch in enumerate(val_loader):
                data = batch['data'].to(device)
                label = batch['labels'].to(device)  #NxHxW

                output = model(data) #Nx2xHxW
                #loss = torch.nn.functional.cross_entropy(output, label, weight=torch.tensor((opt.weight,1), dtype=torch.float32))
                loss = torch.nn.functional.cross_entropy(output, label, weight=class_weight)
                loss_val.append(loss.item())

                pred = torch.argmax(output, dim=1).to(device)
                conf_vector = pred / label
                TP += torch.sum(conf_vector == 1).item()     #pred=1 and label=1
                FP += torch.sum(conf_vector == float('inf')).item()      #pred=1 and label=0
                TN += torch.sum(torch.isnan(conf_vector)).item()     #pred=0 and label=0
                FN += torch.sum(conf_vector == 0).item()     #pred=0 and label=1


        #print(TP, FP, TN, FN)


        print(f'Epoch: {epoch:03d} \t Trn Loss: {sum(loss_list) / (it + 1):.3f} \t Val Loss: {sum(loss_val) / (it_val + 1):.3f}')
        f = open(log_file, 'a+')
        f.write("Epoch {} started\n".format(epoch))
        f.write(f'Epoch: {epoch:03d} \t Trn Loss: {sum(loss_list) / (it + 1):.3f} \t Val Loss: {sum(loss_val) / (it_val + 1):.3f}\n')
        f.write("TP: {}, FP: {}, TN: {}, FN: {}, P: {}, R: {}\n".format(TP_t, FP_t, TN_t, FN_t, TP_t/(TP_t+FP_t), TP_t/(TP_t+FN_t)))
        f.write("TP: {}, FP: {}, TN: {}, FN: {}, P: {}, R: {}\n".format(TP, FP, TN, FN, TP/(TP+FP), TP/(TP+FN)))
        f.close()
        save_model(model, opt.outf+'/'+'epoch'+str(epoch)+'.pth')
