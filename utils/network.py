#!/usr/bin/python2.7

import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
from sklearn.cluster import SpectralClustering

from tensorboardX import SummaryWriter
import time



# wrapper class to provide videos from the dataset as pytorch tensors
class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        # datastructure for frame indexing
        self.selectors = []
        for video in self.dataset.features:
            self.selectors += [ (video, i) for i in range(self.dataset.features[video].shape[1]) ]#shape[1] number of video frames
            # content of selectors:
            # (video_names,index) index range from 0 to shape[1]-1
            #       video_names covers all videos
            #  there are 3085477 items in selectors, this is to say that there are 3085477 frames

    def __len__(self):
        return len(self.selectors)

    def __getitem__(self, idx):
        assert idx < len(self)
        video = self.selectors[idx][0]
        frame = self.selectors[idx][1]
        features = torch.from_numpy( self.dataset.features[video][:, frame] )
        labels = []
        for c in range(self.dataset.n_classes):
            labels.append( torch.LongTensor([1 if c in self.dataset.action_set[video] else 0]) )
        '''
        print 'labels in __getitem__'
        print video
        print labels
        time.sleep()
        '''
        return features, labels


# the neural network
class Net(nn.Module):

    def __init__(self, input_dim, n_classes):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(input_dim, 256)
        self.fcc = nn.Linear(256, 256)
        self.fcco = nn.Linear(256,48)
        self.out_fc = []
        for c in range(n_classes):
            self.out_fc.append( nn.Linear(48, 2) )
        self.out_fc = nn.Sequential(*self.out_fc)
    '''
    def forward(self, x):
        x = nn.functional.relu(self.fc(x))
        outputs = []
        for c in range(self.n_classes):
            tmp = self.out_fc[c](x)
            tmp = nn.functional.log_softmax(tmp, dim = 1)
            outputs.append(tmp)
        return outputs
    '''
    def forward(self, x):

        s2 = torch.FloatTensor([0,1]).cuda()

        x = nn.functional.relu(self.fc(x))
        x = nn.functional.relu(self.fcc(x))
        x_H = torch.tanh(self.fcco(x))
        outputs = []
        oo = []
        for c in range(self.n_classes):
            tmp = self.out_fc[c](x_H)
            tmp_log = nn.functional.log_softmax(tmp, dim = 1)
            # tmp_nlog = nn.functional.softmax(tmp, dim = 1)
            #print tmp_nlog
            outputs.append(tmp_log)
            # oo_tmp = ( torch.matmul(tmp_nlog,s2) )
            # oo.append(oo_tmp)
        # oo = torch.stack(oo,0) # stack the list as a tensor
        # oo_sm = nn.functional.softmax(oo,0)
        #print 'oo_sm',oo_sm
        return outputs,torch.t(x_H)

# class for network training
class Trainer(object):

    def __init__(self, dataset):
        self.dataset_wrapper = DatasetWrapper(dataset)
        self.net = Net(dataset.input_dimension, dataset.n_classes)
        self.net.cuda()
        self.train_count = 0
        self.writer = SummaryWriter('log1')
        self.add_graph_flag = 0
        # print self.net

    def train(self, batch_size = 512, n_epochs = 2, learning_rate = 0.1):
        dataloader = torch.utils.data.DataLoader(self.dataset_wrapper, batch_size = batch_size, shuffle = True,num_workers=16)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.net.parameters(), lr = learning_rate)
        
        # run for n epochs1
        start=time.time()
        print "train start time",start

        for epoch in range(n_epochs):
            for i, data in enumerate(dataloader, 0):
                optimizer.zero_grad()
                input, target = data         
                input = Variable(input.cuda())
                outputs = self.net(input)


                loss = 0
                for c, output in enumerate(outputs):
                    labels = Variable(target[c].cuda())
                    #labels = Variable(target[c])
                    labels = labels.view(-1)
                    loss += criterion(output, labels)
                loss.backward()
                optimizer.step()    
            print 'loss: ',loss
            
        elapsed=time.time()-start
        print "Time used:",elapsed

    def s_train(self, batch_size = 512, n_epochs = 2, learning_rate = 0.1):
        

        dataloader = torch.utils.data.DataLoader(self.dataset_wrapper, batch_size = batch_size, shuffle = True,num_workers=16)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.net.parameters(), lr = learning_rate)
        clustering_method = SpectralClustering(
                n_clusters=48,
                #affinity='nearest_neighbors', ## change here and L_mat,sparse matrix
                affinity='rbf',
                gamma=0.01,
                eigen_solver='arpack',
                n_jobs=18)
        s2 = torch.FloatTensor([0,1])
        s2 = Variable(s2, requires_grad=True)   # Select the sencond value of outputs 
        # run for n epochs1
        start=time.time()
        #print "train start time",start
        stime = time.asctime(time.localtime(start))
        print "Start time:", stime

        for epoch in range(n_epochs):
            for i_enum, data in enumerate(dataloader, 0):
                optimizer.zero_grad()
                input_, target_ = data  
                num_frame=target_[0].size()[0]
                if(num_frame<batch_size):
                    print 'no_enough_data,skip'
                    continue     
                # input = Variable(input_.cuda())
                input = Variable(input_).cuda()
                outputs,oo_sm = self.net(input)  # Now oo_sm is before NLL's calcluatuion
                
                if(not self.add_graph_flag):
                    self.add_graph_flag = 1
                    self.writer.add_graph(self.net, input)
                
                # calculate Laplacian Matrix
                H1=np.zeros((48,batch_size))
                for i in range(48):
                    H1[i]=target_[i].numpy().flatten()
                H1=H1.T
                
                clustering_method.fit(input_)
                W_mat = clustering_method.affinity_matrix_
                ## Use for sparse matrix, i.e nearest neighbor
                #W_mat = W_mat.toarray()
                #np.savetxt('W_mat-%s'%(i_enum),W_mat)
                D_mat = np.diag(np.sum(W_mat,1))
                #np.savetxt('D_mat-%s'%(i_enum),D_mat)
                L_mat = D_mat - W_mat
                #np.savetxt('L_mat-%s'%(i_enum),L_mat)
                ### L_mat =  torch.FloatTensor(L_mat.toarray())
                L_mat =  torch.FloatTensor(L_mat)
                D_mat =  torch.FloatTensor(D_mat)

                L_mat_sym =torch.matmul( torch.inverse(torch.sqrt(D_mat)) , L_mat)
                L_mat_sym = torch.matmul( L_mat_sym, torch.inverse(torch.sqrt(D_mat))) 
                if(i_enum%100==0):
                    print 'L_mat_sym',L_mat_sym
                L_mat_tan = torch.tanh(L_mat)
                #np.savetxt('L_mat/L_mat_sig-%s'%(i_enum),L_mat_sig.numpy())
                ## print 'save_txt'
                ## print 'L_mat.type',type(L_mat) #<class 'scipy.sparse.csr.csr_matrix'>
                Tr1 =np.trace( (H1.T).dot(L_mat_sym.numpy()).dot(H1) )
                if(i_enum%100==0):
                    print 'Tr1',Tr1
                L_mat = L_mat.cuda()
                L_mat_sym = L_mat_sym.cuda()
                L_mat_tan = L_mat_tan.cuda()
                
                Tr1 = torch.tensor(Tr1).cuda()

                loss = 0
                loss_nll = 0
                loss_sl = 0
                # cal batch loss
                for c, output in enumerate(outputs):
                    labels = Variable(target_[c]).cuda()
                    labels = labels.view(-1)
                    loss_nll += criterion(output, labels)
                
                if(i_enum%100==0):
                    print 'oo_sm',oo_sm
                mm0 = torch.matmul(oo_sm,L_mat_sym)
                mm1 = torch.matmul(mm0,torch.t(oo_sm))
                loss_sl = torch.trace(mm1)
                if(i_enum%100==0):
                    print 'Tr2',loss_sl
                
                #loss=loss_nll + torch.sqrt((Tr1 - loss_sl)*(Tr1 - loss_sl))
                #loss= torch.sqrt( (Tr1 - loss_sl)*(Tr1 - loss_sl) )
                loss = loss_nll + 0.01*loss_sl
                self.writer.add_scalar('loss',loss,self.train_count)
                self.writer.add_scalar('loss_nll',loss_nll,self.train_count)
                self.writer.add_scalar('loss_sl',loss_sl,self.train_count)


                loss.backward()
                optimizer.step()  
                #print 'loss in loop',loss  
                if(i_enum%100==0):
                    print 'loss in loop',loss  
                self.train_count+=1
            print 'loss: ',loss
            
        elapsed=time.time()-start
        print "Start time:", stime
        print "Time used:",elapsed


    def save_model(self, model_file):
        torch.save(self.net.state_dict(), model_file)


# class to forward videos through a trained network
class Forwarder(object):

    def __init__(self, model_file):
        self.model_file = model_file
        self.net = None

    def forward(self, dataset):
        # read the data
        dataset_wrapper = DatasetWrapper(dataset)
        dataloader = torch.utils.data.DataLoader(dataset_wrapper, batch_size = 512, shuffle = False,num_workers=16)
        # load net if not yet done
        if self.net == None:
            self.net = Net(dataset.input_dimension, dataset.n_classes)
            self.net.load_state_dict( torch.load(self.model_file,map_location='cuda:1'))
            #self.net.load_state_dict( torch.load(self.model_file,map_location='cpu') )
            #torch.load(self.model_file, map_location=lambda storage, loc: storage.cuda(1))
            self.net.cuda()
        # output probability container
        log_probs = np.zeros( (dataset.n_frames, dataset.n_classes), dtype=np.float32 )
        offset = 0
        # forward all frames
        for data in dataloader:
            input, _ = data
            #input = Variable(input.cuda())
            input = Variable(input).cuda()
            outputs,oo_sm = self.net(input)
            for c, output in enumerate(outputs):
                log_probs[offset : offset + output.shape[0], c] = output.data.cpu()[:, 1]
            offset += output.shape[0]
        return log_probs

