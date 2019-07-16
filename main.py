#import 하는 순간 안에 있는 모든 함수가 순서대로 실행됨.
import net
import numpy as np
import torch
import torch.nn as nn
# import read_data
import datasets
from torch import optim
from criterion import PixelLinkLoss
import loss
import config
import postprocess
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
import os
import cv2
import time
import argparse
import ImgLib.ImgShow as ImgShow
import ImgLib.ImgTransform as ImgTransform
#import moduletest.test_postprocess as test_postprocess
from test_model import test_on_train_dataset

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("first?")
parser = argparse.ArgumentParser(description='')
print("load argparser successfully!")
parser.add_argument('--train', type=bool, default=False, help='True for train, False for test') # default for test
parser.add_argument('--retrain', type=bool, default=False, help='True for retrain, False for train') # default for test
# parser.add_argument('change', metavar='N', type=int, help='an integer for change')
args = parser.parse_args()
print(args)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def retrain():
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    sampler = WeightedRandomSampler([1/len(dataset)]*len(dataset), config.batch_size, replacement=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    my_net = net.Net()
    if config.gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
        if config.multi_gpu:
            my_net = nn.DataParallel(my_net)
    else:
        device = torch.device("cpu")
    my_net.load_state_dict(torch.load(config.saving_model_dir + '%d.mdl' % config.retrain_model_index))
    optimizer = optim.SGD(my_net.parameters(), lr=config.retrain_learning_rate2, \
                            momentum=config.momentum, weight_decay=config.weight_decay)
    optimizer2 = optim.SGD(my_net.parameters(), lr=config.retrain_learning_rate, \
                            momentum=config.momentum, weight_decay=config.weight_decay)
    train(config.retrain_epoch, config.retrain_model_index, dataloader, my_net, optimizer, optimizer2, device)


def train(epoch, iteration, dataloader, my_net, optimizer, optimizer2, device):
    print("train() call")
    for i in range(epoch): #epoch(전첸 training set 1000개를 한번도는 것) 6만번 전체6만번을 돌리겠다.
        print("train() -> for(epoch)")
        for i_batch, sample in enumerate(dataloader): #dataloader를 가지고 dataset을 로드하여 사용한다. enumerate에서
            print("train()-> for(epoch)-> enumerate(dataloader)") #dataloader가 불려서 getitem을 부른다. 
            #i_batch : index 에 해당. sample은 dataloader안에 들은 content에 해당한다.
            print("sample {}".format(sample.keys())) #dataloader 안에 content로 dict가 존재한다.
            #image, pixel_mask, neg_pixel_mask, label, pixel_pos_weight, link_mask 의 key들이 존재하며, 안에는 3차원의 값이 들어있다.
            start = time.time()
            images = sample['image'].to(device) #device에 연결된 GPU로 image연산을 맡긴 image 변수 생성.
            # print(images.shape, end=" ")
            pixel_masks = sample['pixel_mask'].to(device) #각각 마찬가지로 sample에 들어있는 키들 연ㅅ나 역시 GPU에 할당.
            neg_pixel_masks = sample['neg_pixel_mask'].to(device)
            link_masks = sample['link_mask'].to(device)
            pixel_pos_weights = sample['pixel_pos_weight'].to(device)
            print("before forward debug")
            out_1, out_2 = my_net.forward(images) ##문제발생
            loss_instance = PixelLinkLoss()
            # print(out_2)

            pixel_loss_pos, pixel_loss_neg = loss_instance.pixel_loss(out_1, pixel_masks, neg_pixel_masks, pixel_pos_weights)
            pixel_loss = pixel_loss_pos + pixel_loss_neg
            link_loss_pos, link_loss_neg = loss_instance.link_loss(out_2, link_masks)
            link_loss = link_loss_pos + link_loss_neg
            losses = config.pixel_weight * pixel_loss + config.link_weight * link_loss
            print("debug check after loss")
            print("iteration %d" % iteration, end=": ")
            print("pixel_loss: " + str(pixel_loss.tolist()), end=", ")
            # print("pixel_loss_pos: " + str(pixel_loss_pos.tolist()), end=", ")
            # print("pixel_loss_neg: " + str(pixel_loss_neg.tolist()), end=", ")
            print("link_loss: " + str(link_loss.tolist()), end=", ")
            # print("link_loss_pos: " + str(link_loss_pos.tolist()), end=", ")
            # print("link_loss_neg: " + str(link_loss_neg.tolist()), end=", ")
            print("total loss: " + str(losses.tolist()), end=", ")
            if iteration < 100:
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            else:
                optimizer2.zero_grad()
                losses.backward()
                optimizer2.step()
            end = time.time()
            print("time: " + str(end - start))
            if (iteration + 1) % 200 == 0:
                # if args.change:
                #     saving_model_dir = config.saving_model_dir3
                # else:
                saving_model_dir = config.saving_model_dir
                torch.save(my_net.state_dict(), saving_model_dir + str(iteration + 1) + ".mdl")
            iteration += 1

def main():
    print("argpaser() ==> main()")
    ##print("config_image_dir : {}, config_labels_dir : {}".format(config.train_image_dir, config_train_image_dir))
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir) #train_image와 ground_truch 경로가 전달.
    print("dataset type : {}\ndataset info : {}\n dataset len : {}".format(type(dataset), dataset, len(dataset)))
    sampler = WeightedRandomSampler([1/len(dataset)]*len(dataset), config.batch_size, replacement=True)
    print("sampler: {}".format(len(sampler)))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler) #dataloader는 np의 array와 유사한  tensor로 저장.    ## dataloader = DataLoader(dataset, config.batch_size, shuffle=True)
    print("my_net call")
    my_net = net.Net()
    
    if config.gpu:
        device = torch.device("cuda:0") #CUDA는 NVIDIA에서 개발한 GPU사용을 위한 툴이라고 할 수 있다. #계산 device로 GPU를 쓰겠다.
        my_net = my_net.cuda() #net같은 network 역시 GPU로 하겠다.
        if config.multi_gpu: #GPU가 여러개 있다면,
            my_net = nn.DataParallel(my_net) #parallel한 방식으로 나눠서 진행하겠다.
    else:
        device = torch.device("cpu") #그렇지 않다면 계산 device를 CPU로 연결하겠다.
    print("config.gpu: check success!")
    ##nn.init.xavier_uniform_(list(my_net.parameters()))
    my_net.apply(weight_init) #network weight parameter를 initialize 한다.
    optimizer = optim.SGD(my_net.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    #weight_decay : L2 regulization
    ##if args.change:
    #Stochastic Gradient Descent(SGD)로 미니 배치사이즈를 이용해 gradient descent를 진행하며 optimization한다. 
    print("parameter : {}".format(my_net.parameters()))
    optimizer2 = optim.SGD(my_net.parameters(), lr=config.learning_rate2, momentum=config.momentum, weight_decay=config.weight_decay)
    ##else:
    ##     optimizer2 = optim.SGD(my_net.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    print("Go to train")
    iteration = 0
    train(config.epoch, iteration, dataloader, my_net, optimizer, optimizer2, device) #여기서 training 시작

if __name__ == "__main__":
    if args.retrain:
        print("argparser call retrain()")
        retrain()
    elif args.train:
        print("argparser call main()")
        main()
    else:
        print("argparser call test_on_train_dataset()")
        test_on_train_dataset()
        test_model(i)
