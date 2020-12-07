import os
import cv2
import json
import argparse
from PIL import Image
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.exceptions import UndefinedMetricWarning
from datafolder.folder import Test_Dataset
from net import get_model
from torchvision import transforms as T

######################################################################
# Settings
# ---------
use_gpu = True
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--video_path', default='', type=str, help='path to the dataset')
parser.add_argument('--wts_path', default='', type=str, help='path to the dataset')
parser.add_argument('--onnx_path', default='./checkpoints/out.onnx', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
parser.add_argument('--batch-size', default=50, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=2, type=int, help='num_workers')
parser.add_argument('--which-epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--print-table',action='store_true', help='print results with table format')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

print("---> args: ",args)

dataset_name = dataset_dict[args.dataset]
data_dir = args.video_path
model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
model_dir = os.path.join('./checkpoints', args.dataset, model_name)
result_dir = os.path.join('./result', args.dataset, model_name)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

######################################################################
# Function
# ---------
def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network


def check_metric_vaild(y_pred, y_true):
    if y_true.min() == y_true.max() == 0:   # precision
        return False
    if y_pred.min() == y_pred.max() == 0:   # recall
        return False
    return True


######################################################################
# Model
# ---------
num_label = 30 if args.dataset=="market" else 23
num_id = 451 if args.dataset=="market" else 702
model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
model = load_network(model)
if use_gpu:
    model = model.cuda()
model.train(False)  # Set model to evaluate mode

# write to wts (tensorrt)
if len(args.wts_path) >=4:
    print("---> convet to wts (tensorrt)")
    import struct
    f = open(args.wts_path, 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        #print("  -->k, v :\n",k, v )
        #print("---k:",k)
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

if len(args.onnx_path) >= 4:
    output_onnx = args.onnx_path
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3, 288, 144).cuda()

    #torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,input_names=input_names, output_names=output_names)
    torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=True)


    print("==> save pt")
    model_pt = torch.jit.trace(model,torch.rand(1,3,288,144).cuda())
    model_pt.save("./checkpoints/out.pt")

    print("--->done ")


######################################################################
# Testing
# ---------
preds_tensor = np.empty(shape=[0, num_label], dtype=np.byte)   # shape = (num_sample, num_label)
labels_tensor = np.empty(shape=[0, num_label], dtype=np.byte)   # shape = (num_sample, num_label)


if len(args.video_path) >= 4:
    if os.path.isfile(args.video_path):
        print("---> video ,path:",args.video_path)
        cap = cv2.VideoCapture(args.video_path)
    else:
        print("---->images, path:",args.video_path)
        cap = os.listdir(args.video_path)
else:
    print("---> camera ,indx: 0 ")
    cap = cv2.VideoCapture(0)



######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        print("--------------------------------------")
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src

def  det_image1(img_path):
    images = cv2.resize(img_path,(144,288))
    img = images / 255.0  # 0 - 255 to 0.0 - 1.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img).float()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    if use_gpu:
        img = img.cuda()
    #img = load_image(img_path).cuda()
    if not args.use_id:
        out = model.forward(img)
    else:
        out, _ = model.forward(img)

    pred = torch.gt(out, torch.ones_like(out) / 2)  # threshold=0.5

    Dec = predict_decoder(args.dataset)
    Dec.decode(pred)


def  det_image(images):
    images = cv2.resize(images,(144,288))
    img = images / 255.0  # 0 - 255 to 0.0 - 1.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img).float()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # print("---> convet time:",(time.time()-t1)*1000)

    if use_gpu:
        img = img.cuda()
    # img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # print("---> convet time:",(time.time()-t1)*1000)

    # forward
    if not args.use_id:
        pred_label = model(img)
    else:
        pred_label, _ = model(img)
    # print("-->pred_label size:",pred_label.size())
    print("  -->pre_label:",pred_label)
    # torch.ones_like函数和torch.zeros_like函数的基本功能是根据给定张量，生成与其形状相同的全1张量或全0张量
    preds_t = torch.gt(pred_label, torch.ones_like(
        pred_label) / 2)  # torch.gt(a,b)函数比较a中元素大于（这里是严格大于）b中对应元素，大于则为1，不大于则为0，这里a为Tensor，b可以为与a的size相同的Tensor或常数。
    # transform to numpy format
    preds_t = preds_t.cpu().numpy()

    ''' # 年纪 0~3
      0  "young",         
         "teenager",
         "adult",
         "old",
         # 背包 4
         "backpack",
         # 包 5
         "bag",
         #手提包 6
         "handbag",
         #下半身衣服类型7，false裙子，true裤子
         "clothes",
         #下半身衣服 8，false长， true短
         "down",
         #上半身衣服sleeve  9，false长， true短
         "up",
         #头发10，false短发，ture长发
      10 "hair",
         #帽子11   false不带，true带
         "hat",
         #性别12， false男，true女
         "gender",
         #上半身衣服颜色13~20
         "upblack",
         "upwhite",
         "upred",
         "uppurple",
         "upyellow",
         "upgray",
         "upblue",
      20 "upgreen",
         #下半身衣服颜色21~30
         "downblack",
         "downwhite",
         "downpink",
         "downpurple",
         "downyellow",
         "downgray",
         "downblue",
         "downgreen",
         "downbrown"
     '''
    preds = pred_label.cpu().numpy()[0]
    age = preds[0:3]        # young(0), teenager(1), adult(2), old(3)
    backpack = preds[4]
    bag = preds[5]
    handbag = preds[6]
    clothes = preds[7]      #dress(0), pants(1)
    down = preds[8]         #long lower body clothing(0), short(1)
    up = preds[9]           #long sleeve(0), short sleeve(1)
    hair = preds[10]        #short hair(0), long hair(1)
    hat = preds[11]
    gender = preds[12]      # male(0), female(1)
    up_color = preds[13:20] # upblack, upwhite, upred, uppurple, upyellow, upgray, upblue, upgreen
    up_color = preds[21:30] # downblack, downwhite, downpink, downpurple, downyellow, downgray, downblue, downgreen, downbrown
    # print("  -->preds:",preds)

    print("-->age      :", age)  # young(0), teenager(1), adult(2), old(3)
    print("-->backpack :", backpack)
    print("-->bag      :", bag)
    print("-->handbag  :", handbag)
    print("-->clothes  :", clothes)  # dress(0), pants(1)
    print("-->down     :", down)  # long lower body clothing(0), short(1)
    print("-->up       :", up)  # long sleeve(0), short sleeve(1)
    print("-->hair     :", hair)  # short hair(0), long hair(1)
    print("-->hat      :", hat)
    print("-->gender   :", gender)  # male(0), female(1)
    print("-->up_color :", up_color)  # upblack, upwhite, upred, uppurple, upyellow, upgray, upblue, upgreen
    print("-->up_color :", up_color)  # downblack, downwhite, downpink, downpurple, downyellow, downgray, downblue, downgreen, downbrown
    print("--------------------------------------")
    #for i, pred in enumerate(preds_t[0]):
    #    if pred:
    #        print("   ", labels[i])


frames = 0
labels = ["young","teenager","adult","old","backpack","bag","handbag","clothes","down","up","hair","hat","gender","upblack","upwhite","upred","uppurple","upyellow","upgray","upblue","upgreen","downblack","downwhite","downpink","downpurple","downyellow","downgray","downblue","downgreen","downbrown"]
with torch.no_grad():
    if isinstance(cap,list):
        for img_name in cap:
            img_path = os.path.join(args.video_path,img_name)
            images = cv2.imread(img_path)
            cv2.imshow("images",images)

            #det_image(images)
            det_image1(images)

            key = cv2.waitKey(100000)
            if key & 0xff == ord("q"):
                break
            if key & 0xff == ord("n"):
                continue
    else:
        while cap.isOpened():
            ret, images = cap.read()
            cv2.imshow("images",images)

            det_image(images)

            key = cv2.waitKey(100000)
            if key & 0xff == ord("q"):
                break
            if key & 0xff == ord("n"):
                continue

