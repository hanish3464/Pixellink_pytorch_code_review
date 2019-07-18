import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import config
import torch
import codecs
import random
import math
import copy
import time
import cv2
import os
import numpy as np
from torchvision import transforms
import ImgLib.ImgTransform as ImgTransform
import ImgLib.util
class ICDAR15Dataset(Dataset):
    def __init__(self, images_dir, labels_dir):
      #  print("ICDAR15Dataset() class init")
        # self.all_images = self.read_datasets(images_dir, config.all_trains)
        self.images_dir = images_dir #config의 적혀있던 directory의 training 사진의 경로가 여기에 저장된다.
        self.labels_dir = labels_dir #config의 적혀있던 directory의 Ground Truth 텍스트의 경로가 여기에 저장된다.
        #print("self.image_dir : {}, self.labels_dir {}".format(self.images_dir, self.labels_dir))
        self.all_labels = self.read_labels(labels_dir, config.all_trains) #ground truth의 txt파일과 전체갯수가 argument로 넘어간다.
        #all_labels에는 전처리가 끝나서 refined 된 ground_truth가 들어있다.

    def __len__(self):
      #  print("len() call!")
        return len(self.all_labels)

    def __getitem__(self, index):
     #   print("ICDAR15Dataset __getitem__ call")
        if isinstance(index, int):
            return {'image': self.read_image(self.images_dir, index), 'label': all_labels[index]}

    def read_image(self, dir, index):
        index += 1
    #    print("read_image call! /index : {}".format(index)) 
        filename = os.path.join(dir, "img_" + str(index) + ".jpg") #random으로 넘어온 index번호에 맞는 이미지를 디렉터리에서 불러온다.
        image = ImgTransform.ReadImage(filename) #filename에 맞는디렉터리에가서 해당이미지를 불러온다.
        return image

    def read_datasets(self, dir, num):
        res = []
        for i in range(1, num+1):
            image = Image.open(dir+ "img_" + str(i) + ".jpg")
            res.append(image)
            if i % 100 == 0:
                print(i)
        # print(res[0].shape)
        return res

    def read_labels(self, dir, num):
   #     print("read_labels() call!")
        res = [[] for i in range(num)] #res 라는 이차원 배열안에 1차원의 list 1000개(ground_truth 개수)를 생성한다.
        for i in range(1, num+1):
            # utf-8_sig for bom_utf-8
            # print("read %d" % i)

            with codecs.open(dir + "gt_img_" + str(i) + ".txt", encoding="utf-8_sig") as file:
                #dir = train_images\ground_truth 경로str이 있고 그안에 txt파일을utf-8 방식으로 인코딩을 시작한다.
                data = file.readlines() #한줄씩읽는다. 이미지속 4쌍의 coordinate가 0:8 element에 저장, 9번째에 text가 입력되어있다.
            
                tmp = {} #tmp라는 dictionary 선언해서 좌표, text 내용, 무시된 것, area를 담아서 관리한다.
                tmp["coor"] = []
                tmp["content"] = []
                tmp["ignore"] = []
                tmp["area"] = []
                for line in data: #ex> "377,117,463,117,465,130,378,130,Genaxis Theatre"
                    content = line.split(",")#ex> ["377", "117" "463", "117", "465", "130", "378", "130", "Genaxis Theatre"]
                    coor = [int(n) for n in content[:8]] #ex> coor = [377, 117, 463, 117, 465, 130, 378, 130]
                    tmp["coor"].append(coor) #tmp["coor"] 이라는 dictinary의 coor이라는 key에 위의 list가 저장된다.
                    content[8] = content[8].strip("\r\n") #\r\n과 같은 문자를 strip으로 지운다음에 문자열만을 content[8]에 저장한다.
                    tmp["content"].append(content[8]) #마찬가지로 tmp["content"]라는 dictionary 의 content라는 key에 저장된다.
                    if content[8] == "###": #이해못할정도로 문자가 부서졌으면 무시해도 되는 것으로 간주하고 True를 저장한다.
                        tmp["ignore"].append(True)
                    else: #그렇지 않으면 False를 저장한다.
                        tmp["ignore"].append(False)
                    coor = np.array(coor).reshape([4,2]) #[[377,117, [463,117], [465,130], [378,130]]의 4개쌍의 좌표가 된다.
                    tmp["area"].append(cv2.contourArea(coor)) #이것을 면적에 해당하는 contourArea를 구하고 저장한다. 대충 사각형넓이
                res[i-1] = tmp #이렇게 구해진 tmp dictionary를 res안의 list에 하나씩 넣는다.
        return res #이 함수를 거치면 1000개의 tmp dictionary의 정보가 담긴 1000개의 element를 가진 ground_truth res가 완성된다.

class PixelLinkIC15Dataset(ICDAR15Dataset):
    #print("PixelLinkIC15Dataset() call")
    #print("[class:PixelLinkIC15Dataset]")
    def __init__(self, images_dir, labels_dir, train=True):
        super(PixelLinkIC15Dataset, self).__init__(images_dir, labels_dir)
        self.train = train
        # self.all_images = torch.Tensor(self.all_images)

    def __getitem__(self, index):
        # print(index, end=" ")
        if self.train:
#            #print("train : __getitem__ -> train_data_transform(index)")
            image, label = self.train_data_transform(index)
        else:
            image, label = self.test_data_transform(index)
        image = torch.Tensor(image) #CNN에서 사용하기 위해 torch.tensor르 이용해 image를 알맞게 바꿔준다.

        pixel_mask, neg_pixel_mask, pixel_pos_weight, link_mask = \
            PixelLinkIC15Dataset.label_to_mask_and_pixel_pos_weight(label, list(image.shape[1:]), version=config.version)
            #계산이 완료된 후에 dictionary 형태로 반환된다. 
        return {'image': image, 'pixel_mask': pixel_mask, 'neg_pixel_mask': neg_pixel_mask, 'label': label,
                'pixel_pos_weight': pixel_pos_weight, 'link_mask': link_mask}

    def test_data_transform(self, index):
 #       #print("test_transform() call")
        img = self.read_image(self.images_dir, index)
        labels = self.all_labels[index]
        labels, img, size = ImgTransform.ResizeImageWithLabel(labels, (512, 512), data=img)
        img = ImgTransform.ZeroMeanImage(img, config.r_mean, config.g_mean, config.b_mean)
        img = img.transpose(2, 0, 1)
        return img, labels

    def train_data_transform(self, index): #Data augmentation 부분. 같은 image Data를 활용하여 Data를 변형시켜서 수를 늘린다.
  #      #print("train_data_transform() call")
        img = self.read_image(self.images_dir, index) #image 를 불러온다.
        labels = self.all_labels[index] #거기에 맞는 ground_truth 를 불러온다.

        rotate_rand = random.random() if config.use_rotate else 0 #Image를 회전한다.
        crop_rand = random.random() if config.use_crop else 0 #Image의 특정영역으로 자른다. 
        # rotate
        if rotate_rand > 0.5: #모든 이미지를 다 회전시키는 것이 아니라 50퍼센트의 확률로 선택받은 이미지만 회전한다.
            labels, img, angle = ImgTransform.RotateImageWithLabel(labels, data=img) #여기서 회전된 이미지와 ground_truch, 각도를 얻음
        # crop
        if crop_rand > 0.5: #마찬가지로 선택받은 이미지에 대해서 확률적으로 잘라낸다.
            scale = 0.1 + random.random() * 0.9 #scale을 이런식으로 조절한다.
            labels, img, img_range = ImgTransform.CropImageWithLabel(labels, data=img, scale=scale) #추가적으로 이미지 range도 얻음.
            labels = PixelLinkIC15Dataset.filter_labels(labels, method="rai") #staticmethod 사용. 뭘하는지는 모르겠다.
        # resize
        labels, img, size = ImgTransform.ResizeImageWithLabel(labels, (512, 512), data=img) #변형된 이미지를 512x512로 조정한다.
        # filter unsatifactory labels
        # labels = PixelLinkIC15Dataset.filter_labels(labels, method="msi")
        # zero mean
        img = ImgTransform.ZeroMeanImage(img, config.r_mean, config.g_mean, config.b_mean) #zeroMean을 하는게 뭔지 잘 모르겠다.
        # HWC to CHW Height Width Channels 를 의미한다. caffe에선 CHW를 사용. tensorflow는 HWC 사용. 이미지 전처리를 위해 사용.
        img = img.transpose(2, 0, 1) #3차원의 tensor를 conv에서 사용할때 Channel Height Width 순서로 사용하기 때문에 순서를 바꾼다.
        return img, labels #0번(weight)가 중간으로 가고 2번(Channel)이 맨 앞으로 온다. 최종적으로 변형된 이미지와 g_t가 반환된다.

    @staticmethod
    def filter_labels(labels, method):
        print("@staticmethod filter_labels() call")
        """
        method: "msi" for min area ignore, "rai" for remain area ignore
        """
        def distance(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        def min_side_ignore(label):
            label = np.array(label).reshape(4, 2)
            dists = []
            for i in range(4):
                dists.append(distance(label[i], label[(i+1)%4]))
            if min(dists) < 10:
                return True # ignore it
            else:
                return False

        def remain_area_ignore(label, origin_area):
            label = np.array(label).reshape(4, 2)
            area = cv2.contourArea(label)
            if area / origin_area < 0.2:
                return True
            else:
                return False
        if method == "msi":
            ignore = list(map(min_side_ignore, labels["coor"]))
        elif method == "rai":
            ignore = list(map(remain_area_ignore, labels["coor"], labels["area"]))
        else:
            ignore = [False] * 8
        labels["ignore"] = list(map(lambda a, b: a or b, labels["ignore"], ignore))
        return labels

    @staticmethod
    def label_to_mask_and_pixel_pos_weight2(label, img_size, version="2s", neighbors=8):
        """
        8 neighbors:
            0 1 2
            7 - 3
            6 5 4
        """
        def is_valid_coor(h_index, w_index, h, w):
            if h_index < 0 or w_index < 0:
                return False
            elif h_index >= h or w_index >= w:
                return False
            else:
                return True

        def get_neighbors(h_index, w_index):
            res = []
            res.append([h_index - 1, w_index - 1])
            res.append([h_index - 1, w_index])
            res.append([h_index - 1, w_index + 1])
            res.append([h_index, w_index + 1])
            res.append([h_index + 1, w_index + 1])
            res.append([h_index + 1, w_index])
            res.append([h_index + 1, w_index - 1])
            res.append([h_index, w_index - 1])
            return res

        factor = 2 if version == "2s" else 4
        label_coor = np.array(label["coor"]).reshape([-1, 1, 4, 2])
        pixel_mask_size = [int(i / factor) for i in img_size]
        link_mask_size = [neighbors, ] + pixel_mask_size
        pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8)
        pixel_weight = np.zeros(pixel_mask_size, dtype=np.float)
        link_mask = np.zeros(link_mask_size, dtype=np.uint8)
        label_coor = (label_coor / factor).astype(int)

        bbox_masks = []
        num_positive_bboxes = 0
        for i, coor in enumerate(label_coor):
            pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
            cv2.drawContours(pixel_mask_tmp, coor, -1, 1, thickness=-1)
            bbox_masks.append(pixel_mask_tmp)
            if not label["ignore"][i]:
                pixel_mask += pixel_mask_tmp
                num_positive_bboxes += 1
        pos_pixel_mask = (pixel_mask == 1).astype(np.int)
        num_pos_pixels = np.sum(pos_pixel_mask)
        sum_mask = np.sum(bbox_masks, axis=0)
        neg_pixel_mask = (sum_mask != 1).astype(np.int)
        not_overlapped_mask = sum_mask == 1
        for bbox_index, bbox_mask in enumerate(bbox_masks):
            bbox_positive_pixel_mask = bbox_mask * pos_pixel_mask
            num_pos_pixel = np.sum(bbox_positive_pixel_mask)
            if num_pos_pixel > 0:
                per_bbox_weight = num_pos_pixels * 1.0 / num_positive_bboxes
                per_pixel_weight = per_bbox_weight / num_pos_pixel
                pixel_weight += bbox_positive_pixel_mask * per_pixel_weight
            for link_index in range(neighbors):
                link_mask[link_index][np.where(bbox_positive_pixel_mask)] = 1
            bbox_contours = ImgLib.util.find_contours(bbox_positive_pixel_mask)
            bbox_border_mask = np.zeros(pixel_mask_size, dtype=np.int)
            bbox_border_mask *= bbox_positive_pixel_mask
            bbox_border_cords = np.where(bbox_border_mask)
            border_points = list(zip(*bbox_border_cords))
            def in_bbox(nx, ny):
                return bbox_positive_pixel_mask[ny, nx]
            for h_index, w_index in border_points:
                neighbors = get_neighbors(h_index, w_index)
                for nei_index, [nei_h_index, nei_w_index] in enumerate(neighbors):
                    if not is_valid_coor(h_index, w_index, *img_size) or not in_bbox(nei_h_index, nei_w_index):
                        link_mask[nei_index, h_index, w_index] = 0
        return torch.LongTensor(pixel_mask), torch.LongTensor(neg_pixel_mask), \
            torch.Tensor(pixel_weight), torch.LongTensor(link_mask)

    @staticmethod
    def label_to_mask_and_pixel_pos_weight(label, img_size, version="2s", neighbors=8):
        """
        8 neighbors:
            0 1 2
            7 - 3
            6 5 4
        """
        factor = 2 if version == "2s" else 4
        #print("label[ignore]: {}, label[coor]: {}".format(label['ignore'], label['coor']))
        ignore = label["ignore"] #각ground_truth에서 잡혀지는 txt박스의 개수와 무시해도되는지 여부의 ignore가 있다.
        label = label["coor"] #여기에는 4쌍의 박스 좌표쌍이 총 detected된 txt박스 개수만큼 존재한다.
        assert len(ignore) == len(label) #둘의 개수가 같은지를 확인한다.
        label = np.array(label)
        label = label.reshape([-1, 1, 4, 2])
        pixel_mask_size = [int(i / factor) for i in img_size] #img_size 512*512 / pixel_mask_size = 256*256
        link_mask_size = [neighbors, ] + pixel_mask_size #link_mask_size 8*256*256

        pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8) #pixel_mask 256*256 , 0으로 초기화
        pixel_weight = np.zeros(pixel_mask_size, dtype=np.float) #pixel_weight 256*256, 0으로 초기화
        link_mask = np.zeros(link_mask_size, dtype=np.uint8) #link_mask 256*256*8, 8개짜리가 256개에 다시 256개
        # if label.shape[0] == 0:
            # return torch.LongTensor(pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)
        label = (label / factor).astype(int) # label's coordinate value should be divided

        # cv2.drawContours(pixel_mask, label, -1, 1, thickness=-1)
        real_box_num = 0
        # area_per_box = []
        #print("label.shape[0]: {} [1] :{} [2] : {} [3] : {}".format(label.shape[0], label.shape[1], label.shape[2], label.shape[3]))
        for i in range(label.shape[0]): #label.shape[0]는 감지된 최종 box의 총 개수를 담고 있다.
            if not ignore[i]: #실제로 ignore하면 안되는 것들 = 마땅히 detected되어야 하는 text area에 대해서...
                pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8) #똑같은 mask_size로 하나 더 만든다.
                #print("pixel_mask_tmp : {}".format(sum(sum(pixel_mask_tmp))))
                cv2.drawContours(pixel_mask_tmp, label[i], -1, 1, thickness=-1) #우리가 찾은 면적을 직접 그린다.
                #print("[drwaContours] pixel_mask_tmp : {}".format(sum(sum(pixel_mask_tmp))))
                pixel_mask += pixel_mask_tmp #drawContour를 수행한 이후에 pixel_mask_tmp의 값이 변한다. 따라서 각 text 가 속하는 구역
                                             #이 모두 1(positive pixel)이 되어 pixel_mask에 전부 저장되는 것이라 할 수 있음.
                                             #결과적으로 각 text 영역에 속하는 pixel 값들만 1이되어 mask에 덮어 씌어 진다고 할 수있음.
                                             #pixel-wise semantic segmetation과 유사하게 됨.
                ############IMAGE DEBUG#############
                #plt.subplot(1,2,1)
                #plt.imshow(pixel_mask, cmap='bone')
                #plt.title("image {}".format(i))
                ##print("image text index: {}".format(i))
                #plt.axis('off')
                #plt.tight_layout()
                #plt.show()
                #time.sleep(1)
        neg_pixel_mask = (pixel_mask == 0).astype(np.uint8) #0으로 된부분이 모두 1로 바뀌어 negative_pixel만이 표시된다.
                                                            #결과적으로 text영역이 아닌부분만을 check하게 된다.
        pixel_mask[pixel_mask != 1] = 0 #1을 넘어간 부분은 0으로 바꾼다. 즉, 중복되어 겹치는 부분은 text가 없다는 것을 판단하겠다.
        # assert not (pixel_mask>1).any()
        pixel_mask_area = np.count_nonzero(pixel_mask) # total area //positive pixel로 찍힌 pixel을 모두 count하겠다.
        for i in range(label.shape[0]):
            if not ignore[i]:
                pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
                cv2.drawContours(pixel_mask_tmp, label[i], -1, 1, thickness=-1) #label[i]는 텍스트박스 i의 4개의 좌표쌍을 들고있다.
                pixel_mask_tmp *= pixel_mask  #그전에 구했던 pixel_mask와 새롭게 구한것을 곱해서 진짜로 1인 부분만 다시 check
                                              #그리고 그것을 drawcontours를 통해 그리고 pixel_mask
                #print("pixel_mask_tmp : {} , sum: {}".format(pixel_mask_tmp,sum(pixel_mask_tmp)))
                if np.count_nonzero(pixel_mask_tmp) > 0:                        #map에 1로 labeling하면서 그리게 된다.
                    real_box_num += 1 #그렇게 최종적으로 check된 area를 세어서 text_box의 개수를 세겠다.

        if real_box_num == 0: #이미지 내에서 text를 발견하지 못한 case에 해당한다.
            # print("box num = 0")
            return torch.LongTensor(pixel_mask), torch.LongTensor(neg_pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)
        #pixel_mask에는 text영역의 pixel들이 0,1 맵으로 형성되어있다. box가 없을 경우는 int64 형태로 넘기고 weight는 float형태로 한다.
        avg_weight_per_box = pixel_mask_area / real_box_num #box의 평균 weight를 mask_area라는 pixel 점들의 합을 박스갯수로 나눈다.
        #즉, 텍스트 박스크기에 따라서도 weight 값이 조금씩 달라질 수 있다는 것을 암시한다.
        #plt.subplot(1,2,1)
        #plt.imshow(pixel_mask, cmap='bone')
        #plt.title("IMAGE TEXT NUM:: {}".format(label.shape[0]))
        #plt.axis('off')
        #plt.tight_layout()
        #plt.show()
        #time.sleep(3)

        for i in range(label.shape[0]): # num of box
            if not ignore[i]:
                pixel_weight_tmp = np.zeros(pixel_mask_size, dtype=np.float) 
                cv2.drawContours(pixel_weight_tmp, [label[i]], -1, avg_weight_per_box, thickness=-1)
                #평균적인 box크기를 가지고 pixel_weight_tmp에 contour를 그린다.
                pixel_weight_tmp *= pixel_mask #기존에 구했던 mask 를 곱해서 겹치는 부분만을 골라낸다.
                area = np.count_nonzero(pixel_weight_tmp) # area per box
                if area <= 0: #area가 없다면 다음 text_box를 탐색한다.
                      # print("area label: " + str(label[i]))
                      # print("area:" + str(area))
                      continue
                pixel_weight_tmp /= area #있다면 area로 나눠서 weight의 평균을 구한다.
                # print(pixel_weight_tmp[pixel_weight_tmp>0])
                pixel_weight += pixel_weight_tmp #하나의 box의 weight를 weight sum에 하나씩 더해간다.
                
                # link mask
                weight_tmp_nonzero = pixel_weight_tmp.nonzero() #0이 아닌 다른 값이 담긴 index를 weight_tmp_nonzero에 저장한다.
                # pixel_weight_nonzero = pixel_weight.nonzero() #pixel_weight에서 0이 아닌 값들이 담긴 index의 위치를 저장한다.
                link_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
                # for j in range(neighbors): # neighbors directions
                ##1로 셋팅된 weight pixel에 대해서 이하 아래의 각 이웃의 8방향을 모두 1로 셋팅하겠다.
                link_mask_tmp[weight_tmp_nonzero] = 1
                link_mask_shift = np.zeros(link_mask_size, dtype=np.uint8)
                w_index = weight_tmp_nonzero[1] #width 즉, 가로 index를 w_index에 저장
                h_index = weight_tmp_nonzero[0] #height 즉, 세로 index를 h_index에 저장
                w_index1 = np.clip(w_index + 1, a_min=None, a_max=link_mask_size[1] - 1)
                w_index_1 = np.clip(w_index - 1, a_min=0, a_max=None)
                h_index1 = np.clip(h_index + 1, a_min=None, a_max=link_mask_size[2] - 1)
                h_index_1 = np.clip(h_index - 1, a_min=0, a_max=None)
                link_mask_shift[0][h_index1, w_index1] = 1
                link_mask_shift[1][h_index1, w_index] = 1
                link_mask_shift[2][h_index1, w_index_1] = 1
                link_mask_shift[3][h_index, w_index_1] = 1
                link_mask_shift[4][h_index_1, w_index_1] = 1
                link_mask_shift[5][h_index_1, w_index] = 1
                link_mask_shift[6][h_index_1, w_index1] = 1
                link_mask_shift[7][h_index, w_index1] = 1
                
                for j in range(neighbors): #이런식으로 하면 대략link_mask에는 떨어져있는 pixel표시는 사라진다. 붙어있는 instance단위
                    #에 대해서만 1이 되어 실질적인 instance 사이의 분리가 이루어진다.
                    # +0 to convert bool array to int array
                    link_mask[j] += np.logical_and(link_mask_tmp, link_mask_shift[j]).astype(np.uint8)
        ######IMAGE DEBUG#######
        #plt.subplot(1,2,1)
        #plt.imshow(pixel_weight)
        #plt.title("IMAGE WEIGHT: {}".format(label.shape[0]))
        #plt.axis('off')
        #plt.tight_layout()
        #plt.show()
        #for i in link_mask:
         #   plt.subplot(1,2,1)
          #  plt.imshow(i)
           # plt.title("IMAGE LINK: {}".format(label.shape[0]))
            #plt.axis('off')
            #plt.tight_layout()
            #plt.show()
        return [torch.LongTensor(pixel_mask), torch.LongTensor(neg_pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)]

if __name__ == '__main__':
    start = time.time()
    dataset = PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    end = time.time()
    print("time to read datasets: " + str(end - start)) # about 0.12s

    start = time.time()
    sample = dataset.__getitem__(588)
    end = time.time()
    print("time to get 1000 items: " + str(end - start)) # about 34s

    # pixel_mask = sample['pixel_pos_weight']
    # link_mask = sample['link_mask']
    image = sample['image'].data.numpy() * 255
    image = np.transpose(image, (1, 2, 0))
    image = np.ascontiguousarray(image)
    # shape = image.shape
    # image = image.reshape([int(shape[0]/2), 2, int(shape[1]/2), 2, shape[2]])
    # image = image.max(axis=(1, 3))
    # cv2.imwrite("trans0.jpg", image)
    # pixel_mask = pixel_mask.unsqueeze(2).expand(-1, -1, 3)
    # pixel_mask = pixel_mask.numpy()
    # import IPython 
    # IPython.embed()
    # link_mask = link_mask.unsqueeze(3).expand(-1, -1, -1, 3)
    # link_mask = link_mask.numpy()
    # image = image * pixel_mask
    label = sample['label'].reshape([-1, 4, 2])
    cv2.drawContours(image, label, -1, (255, 255, 0))
    cv2.imwrite("trans1.jpg", image)
