import torch.nn as nn
import config

class Net(nn.Module): #nn에 있는 모듈을 상속받아서 구현한다. network architecture는 기본 VGG net을 따른다.
    def __init__(self):
        super(Net, self).__init__()
        # TODO: modify padding
        print("[class:Net][def:init]")
        print("fix")
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1) #3,64,3 = input image channel, output image channel, 3*3 conv
        self.relu1_1 = nn.ReLU() #activation function : vanishing gradient 문제를 해결있다. x>0 x else 0
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1) #in : 64 / out: 64 / filter: 3 
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) 
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1) #in: 64 / out: 128 / filter: 3
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1) #in: 128 / out: 128 / filter: 3
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1) #in: 128 / out: 256 / filter: 3
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1) #in: 256 / out: 256 / filter: 3
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1) #in: 256 / out: 256 / filter: 3
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1) #in: 256 / out: 512 / filter: 3
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1) #in: 512 / out: 512 / filter: 3
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1) #in: 512 / out: 512 / filter: 3
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        print("check pool: {}".format(self.pool4))
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1) #in: 512 / out: 512 / filter: 3
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1) #in: 512 / out: 512 / filter: 3
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1) #in: 512 / out: 512 / filter: 3
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=1, padding=1)
        #dilation: 
        if config.dilation: #FCnet 6,7 번이pixellink에선 conv net으로 전환된다.
            self.conv6 = nn.Conv2d(512, 1024, 3, stride=1, padding=6, dilation=6)
        else:
            self.conv6 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.relu7 = nn.ReLU()

        self.out1_1 = nn.Conv2d(128, 2, 1, stride=1, padding=0) #conv 1x1, [out_channel]: 2
        self.out1_2 = nn.Conv2d(128, 16, 1, stride=1, padding=0)#conv 1x1, [out_channel]: 16
        self.out2_1 = nn.Conv2d(256, 2, 1, stride=1, padding=0) #conv 1x1, [out_channel]: 2
        self.out2_2 = nn.Conv2d(256, 16, 1, stride=1, padding=0)#conv 1x1, [out_channel]: 16
        self.out3_1 = nn.Conv2d(512, 2, 1, stride=1, padding=0) #conv 1x1, [out_channel]: 2
        self.out3_2 = nn.Conv2d(512, 16, 1, stride=1, padding=0)#conv 1x1, [out_channel]: 16
        self.out4_1 = nn.Conv2d(512, 2, 1, stride=1, padding=0) #conv 1x1, [out_channel]: 2
        self.out4_2 = nn.Conv2d(512, 16, 1, stride=1, padding=0)#conv 1x1, [out_channel]: 16
        self.out5_1 = nn.Conv2d(1024, 2, 1, stride=1, padding=0)#conv 1x1, [out_channel]: 2
        self.out5_2 = nn.Conv2d(1024, 16, 1, stride=1, padding=0)#conv 1x1, [out_channel]: 16

        self.final_1 = nn.Conv2d(2, 2, 1, stride=1, padding=0) #text/non-text prediction [channel]: 2
        self.final_2 = nn.Conv2d(16, 16, 1, stride=1, padding=0) #link prediction [channel]: 16

    def forward(self, x):

        print("[Class:Net][def:forward]")
        print("[def:foward] conv1_1->relu->conv1_2->relu->pool1")
        x = self.pool1(self.relu1_2(self.conv1_2(self.relu1_1(self.conv1_1(x))))) 
        #[Convolution Stage1](2번) + [Pool1,/2]
        print("[def:forwad] conv2_1->relu->conv2_2->relu")
        x = self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(x)))) 
        #[Convolution Stage2](2번)

        l1_1x = self.out1_1(x) #[conv 1x1, 2] Text/non-text Prediction /After [conv stage2]
        l1_2x = self.out1_2(x) #[conv 1x1, 16] Link Prediction /After [conv stage2]
    
        x = self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(self.pool2(x))))))) 
        #[Pool2, /2] + [Convolution Stage3](3번) 
        
        l2_1x = self.out2_1(x) #[conv 1x1, 2]  Text/non-text Prediction /After [conv stage3]
        l2_2x = self.out2_2(x) #[conv 1x1, 16] Link Prediction /After [conv stage3]

        x = self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(self.relu4_1(self.conv4_1(self.pool3(x)))))))
        #[Convolution Stage4](3번)
        
        l3_1x = self.out3_1(x) #[conv 1x1, 2] Text/non-text Prediction /After [conv stage4]
        l3_2x = self.out3_2(x) #[conv 1x1, 16] Link Prediction /After [conv stage4]
        
        x = self.relu5_3(self.conv5_3(self.relu5_2(self.conv5_2(self.relu5_1(self.conv5_1(self.pool4(x)))))))
        #[pool4, /2] + [Convolution Stage5](3번)

        l4_1x = self.out4_1(x) #[conv 1x1, 2] Text/non-text Prediction /After [conv stage5]
        l4_2x = self.out4_2(x) #[conv 1x1, 16] Link Prediction /After [conv stage5]
        
        x = self.relu7(self.conv7(self.relu6(self.conv6(self.pool5(x)))))
        #[pool5, /1] + [FCnet -> Convolution 6,7]

        l5_1x = self.out5_1(x) #[conv 1x1, 2] Text/non-text Prediction /After [conv 6,7]
        l5_2x = self.out5_2(x) #[conv 1x1, 16] Link Prediction /After [conv 6,7]

        upsample1_1 = nn.functional.upsample(l5_1x + l4_1x, scale_factor=2, mode="bilinear", align_corners=True)
        #Bi-linear Interporation 방식을 이용하여 Image 의 해상도를 깨지않고 Upsampleing 한다.
        #[conv stage 5] + [conv 6,7]의 Text/non-text Prediction 의 결과에 대한 Upsampling.

        upsample2_1 = nn.functional.upsample(upsample1_1 + l3_1x, scale_factor=2, mode="bilinear", align_corners=True)
        #마찬가지의 방식으로 Upsampling
        #[Upsample1_1] + [conv stage 4]의 Text/non-text Prediction의 결과에 대한 Upsampling
        
        #마찬가지의 방식으로 Upsampling을 진행한다. version 2s라면 한번 더 진행한다.
        if config.version == "2s":
            upsample3_1 = nn.functional.upsample(upsample2_1 + l2_1x, scale_factor=2, mode="bilinear", align_corners=True)
            out_1 = upsample3_1 + l1_1x
        else:
            out_1 = upsample2_1 + l2_1x
        
        #Link Prediction에 대한 Upsampling. 방식은 위와 동일하다.
        upsample1_2 = nn.functional.upsample(l5_2x + l4_2x, scale_factor=2, mode="bilinear", align_corners=True)
        upsample2_2 = nn.functional.upsample(upsample1_2 + l3_2x, scale_factor=2, mode="bilinear", align_corners=True)
        if config.version == "2s":
            upsample3_2 = nn.functional.upsample(upsample2_2 + l2_2x, scale_factor=2, mode="bilinear", align_corners=True)
            out_2 = upsample3_2 + l1_2x
        else:
            out_2 = upsample2_2 + l2_2x

        return [out_1, out_2] #Upsampling 까지한 최종 결과를 반환한다.

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
