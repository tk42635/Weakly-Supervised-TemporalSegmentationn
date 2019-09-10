#!/usr/bin/python2.7

import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
from utils.network import Net
import time

class refine(object):
    def __init__(self):
        self.dataset
        self.new_feature
        self.new_label


# wrapper class to provide videos from the dataset as pytorch tensors
class FineDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset,in_fine_labels):
        self.dataset = dataset
        # fine_labels(dict):fine_labels[video]=[class of this frame,xxx ,''',[]]
        self.fine_labels = in_fine_labels
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
            # only one label is 1, all others are 0
            labels.append( torch.LongTensor([1 if c == self.fine_labels[video][frame] else 0]) )
        
        return features, labels

# class for Fine network training
class FineTrainer(object):

    def __init__(self, model_file,dataset,New_GD):
        self.dataset_wrapper = FineDatasetWrapper(dataset,New_GD)
        self.net = Net(dataset.input_dimension, dataset.n_classes)
        # self.net.load_state_dict( torch.load(model_file,map_location='cpu'))
        # print self.net
        self.net.cuda()

    def train(self, batch_size = 512, n_epochs = 2, learning_rate = 0.1):
        dataloader = torch.utils.data.DataLoader(self.dataset_wrapper, batch_size = batch_size, shuffle = True,num_workers=8)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.net.parameters(), lr = learning_rate)
        
        # run for n epochs1
        start=time.time()
        print "train start time",start

        for epoch in range(n_epochs):
            print(time.time()-start)
            #print 'enumerate of dataloader Count',count

            for i, data in enumerate(dataloader, 0):
                # print 'dataloader',i,len(data)
                optimizer.zero_grad()
                input, target = data
                
                # print 'input',input.size()
                # print 'target',len(target)
                # print 'target[0]',len(target[0])
                # print target[0].size()
                             
                input = Variable(input.cuda())
                #input = Variable(input)

                outputs = self.net(input)

                '''
                print 'outputs',len(outputs)
                print 'outputs[0]',outputs[0].size()
                '''

                loss = 0
                for c, output in enumerate(outputs):
                    labels = Variable(target[c].cuda())
                    #labels = Variable(target[c])
                    labels = labels.view(-1)
                    loss += criterion(output, labels)
                #print 'loss',loss
                loss.backward()
                optimizer.step()
                #time.sleep()
            print 'loss',loss
            
        elapsed=time.time()-start
        print "Time used:",elapsed
    def save_model(self, model_file):
        torch.save(self.net.state_dict(), model_file)

