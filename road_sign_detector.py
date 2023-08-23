import torch
from torch.utils.data import Dataset
import scipy.io
import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
from torchvision.models import resnet50
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
import matplotlib.pyplot as plt




import torchvision
from torchvision import transforms
import xml.etree.ElementTree as ET

index_from_label={}
label_from_index={}
availableind=0
def read_content(xml_file: str):
    global availableind

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    filename = root.find('filename').text

    for boxes in root.iter('object'):

        label=boxes.find("name").text
        if label not in index_from_label:
            index_from_label[label]=availableind
            label_from_index[availableind]=label
            availableind+=1
        print(label)
        label=index_from_label[boxes.find("name").text]


        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_single_boxes,label


def set_up_data():
    data=[]
    annotations=os.listdir('./data/road_sign/annotations/')
    index=1
    t=0
    for inx,f in enumerate(annotations):

        name, boxes,label = read_content('./data/road_sign/annotations/'+f)
        print(name,boxes)


        image = Image.open('./data/road_sign/images/'+name)
        w, h = image.size
        # print(w,h,ymax,im)

        xmin,ymin,xmax,ymax =boxes
        xmin /= w
        ymin /= h
        xmax /= w
        ymax /= h
        data.append({'image':'./data/road_sign/images/'+name,'bbox':torch.tensor(np.array([xmin,ymin,xmax,ymax]).astype(np.float64)),'label':np.long(label)})
        print(data[-1])
        #print(data[-1])





    return data

data=set_up_data()

class custom_dataset(Dataset):

    def __init__(self,data,transform=None):
        self.data=data
        self.transforms=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im=self.data[idx]['image']
        label=self.data[idx]['label']
        xmin,ymin,xmax,ymax=self.data[idx]['bbox']
        image=Image.open(im).convert('RGB')
        w,h=image.size
        print(image.size)
        #print(w,h,ymax,im)

        bbox=torch.tensor([xmin,ymin,xmax,ymax])
        #print(image.mode)
        if image.mode=='L':
            image=Image.merge('RGB',(image,image,image))
        #image.show()

        if self.transforms:

            return self.transforms(image),label,bbox,im

        return (image, label, bbox,im)


class object_detector(Module):
    def __init__(self):
        super().__init__()
        self.backbone=resnet50(pretrained=True)
        #for param in self.backbone.parameters():
            #param.requires_grad = False
        self.regressor = Sequential(
            Linear(self.backbone.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )

        self.classifier = Sequential(
            Linear(self.backbone.fc.in_features, 256),
            ReLU(),
            Dropout(),
            Linear(256, 128),
            ReLU(),
            Dropout(),
            Linear(128, 4)
        )
        self.backbone.fc=Identity()

    def forward(self,x):
        x=self.backbone(x)
        r=self.regressor(x)
        c=self.classifier(x)

        return (r,c)







transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


split=train_test_split(data,test_size=0.2)
train_data=split[0]
test_data=split[1]
train_dataset=custom_dataset(train_data,transforms)
test_dataset=custom_dataset(test_data,transforms)

train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model=object_detector()
model.to(device)
print(device)

regressor_loss=MSELoss(reduction='sum')
classifier_loss=CrossEntropyLoss()
opt=Adam(model.parameters(),lr=5e-5)
epochs=20
losses=[]
r_losses=[]
c_losses=[]
for i in range(epochs):
    model.train()
    for (images,labels,bboxes,im) in train_data_loader:
        images=images.to(device)
        labels=labels.to(device)
        bboxes=bboxes.to(device)
        print(im)
        r,c=model(images)
        print(r)
        print(bboxes)
        r_loss=regressor_loss(r.float(),bboxes.float()).float()
        c_loss=classifier_loss(c,labels).float()
        loss=r_loss+c_loss
        r_losses.append(r_loss.item())
        c_losses.append(c_loss.item())
        losses.append(loss.item())
        print("rloss  ",r_loss.item())
        print((c.argmax(1) == labels).type(torch.float).sum().item())
        print(c.argmax(1))

        print(labels)


        opt.zero_grad()
        loss.backward()
        opt.step()

plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title("total loss")
plt.subplot(1, 3, 2)
plt.plot(r_losses)
plt.title("regression loss")
plt.subplot(1, 3, 3)
plt.plot(c_losses)
plt.title("classification loss")
plt.show()
torch.save(model.state_dict(),'model')


model=object_detector()
model.to(device)
model.load_state_dict(torch.load('model'))


total=0
correct=0
num=0
with torch.no_grad():
    model.eval()
    for (images,labels,bboxes,address) in test_data_loader:
        num+=1
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        r, c = model(images)
        print(r)
        im=cv2.imread(address[0])
        h,w,j=im.shape
        print(bboxes[0])
        xmin,ymin,xmax,ymax=bboxes[0]
        xmin*=w
        ymin*=h
        xmax*=w
        ymax*=h
        print(xmin.item(),ymin.item(),xmax.item(),ymax.item())
        print(int(r[0][2]),int(r[0][3]))

        im=cv2.rectangle(im,(int(xmin.item()),int(ymin.item())),(int(xmax.item()),int(ymax.item())),(255,0,0),2)
        im=cv2.rectangle(im,(int(r[0][0]*w),int(r[0][1]*h)),(int(r[0][2]*w),int(r[0][3]*h)),(0,255,0),2)

        cv2.putText(im,label_from_index[int(c.argmax(1)[0].item())],(int(r[0][0]*w),int(r[0][1]*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)

        #cv2.imshow('im',im)
       # cv2.waitKey(0)
        cv2.imwrite('data/road_sign/evaluation/'+str(num)+'.jpg',im)


        #plt.imshow(images.cpu().numpy()[0].swapaxes(0,1).swapaxes(1,2))
        #plt.show()

        correct+=(c.argmax(1) == labels).type(torch.float).sum().item()

        print((c.argmax(1) == labels).type(torch.float).sum().item())
        print(c.argmax(1))

        print(labels)
        total+=1
    print(correct/total)