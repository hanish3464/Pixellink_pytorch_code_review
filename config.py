
version = "2s"
epoch = 60000
learning_rate = 1e-3
learning_rate2 = 1e-2
all_trains = 1000 #read_labels() 를 호출할때 num parameter의 argument에 해당함. 사진이 1000장 있으니 1000장 넘어간다.
batch_size = 5 #read_image call을 해서 24번 불러옴
momentum = 0.9 #momentum은 gradient descent를 하는 과정에서 일종에 관성을 주는 방식으로 과거의 이동방식을 기억하는 것을 말한다.
               #보통 0.9정도의 momentum term의 값을 사용한다.
weight_decay = 5e-4 #regularization 의 parameter lamda라고 할 수 있다. 
dilation = True
use_crop = False
use_rotate = True #rotate 를 사용
# iterations = 10
gpu = True #gpu 사용관련 parameter
multi_gpu = False # only useful when gpu=True
pixel_weight = 2
link_weight = 1

r_mean = 123. #data transform 을 하는 단계에서 사용하는 parameter
g_mean = 117.
b_mean = 104.

image_height = 512
image_width = 512
image_channel = 3

link_weight = 1
pixel_weight = 2
neg_pos_ratio = 3 # parameter r in paper

train_images_dir = "train_images/images/" #여기서 train data를 불러온다.
train_labels_dir = "train_images/ground_truth/" #여기서 train data ground_truth(학습 이미지에 대한 정답인 담긴 label)를 불러온다. 
saving_model_dir = "models/"
retrain_model_index = 26200 # retrain from which model, e.g. ${saving_model_dir}/156600.mdl
test_model_index = 43800 # test for which model, e.g. ${saving_model_dir}/156600.mdl
test_batch = 1
# saving_model_dir1 = "standard_models/"
# saving_model_dir2 = "change_models/"
# saving_model_dir3 = "change_models_without_replacement/"

retrain_epoch = 60000
retrain_learning_rate = 1e-2
retrain_learning_rate2 = 3e-3

