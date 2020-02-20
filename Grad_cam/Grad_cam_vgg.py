import torch
import cv2
import numpy as np



from torch.nn import functional as F


class FeatureExtractor_vgg():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers): # target_layers = 35 ==> VGG19에서 가장 마지막 MaxPool2D전 ReLU함수
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)


    def __call__(self, x):
        self.gradients = []
        for name, module in self.model._modules.items(): # 모든 layer에 대해서 직접 접근
            x = module(x)
            if name in self.target_layers: # target_layer라면 해당 layer에서의 gradient를 저장
                x.register_hook(self.save_gradient) #
                target_feature_maps = x # x's shape = 512X14X14(C,W,H) feature map
        return target_feature_maps, x


class ModelOutputs_vgg():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor_vgg(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output) # feature extract를 통해서 나온 값을 활용하여 classification 진행
        #print("ModelOutputs().output.shape : ",output[0])
        #print("ModelOutputs().target_activations.shape :",target_activations[0])
        return target_activations, output

class GradCam_vgg:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda: # GPU일 경우 model을 cuda로 설정
            self.model = model.cuda()

        self.extractor = ModelOutputs_vgg(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda: # GPU일 경우 input을 cuda로 변환하여 전달
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        #print("features : ",features.cpu().data.numpy().shape) # 해당 위치에서 추출된 feature map ( 512,14,14 ) (ChannelX14X14)
        #print("output : ",output.cpu().data.numpy().shape) # class를 의미함
        probs, idx = 0,0
        #print("index : ", index)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())  # index = 정답이라고 추측한 class index
            h_x = F.softmax(output,dim=1).data.squeeze()
            probs, idx = h_x.sort(0,True)
        #print("index : ", index)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1 # 정답이라고 생각하는 class의 index 리스트 위치의 값만 1로
        one_hot = torch.from_numpy(one_hot).requires_grad_(True) # numpy배열을 tensor로 변환
        # requires_grad == True 텐서의 모든 연산에 대하여 추적
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        #print("grads_val : ",grads_val.shape) # 512 X 14 X 14
        target = features  # A^k
        target = target.cpu().data.numpy()[0, :]

        cam = None

        weights = np.mean(grads_val, axis=(2, 3))[0, :]  # 논문에서의 global average pooling 식에 해당하는 부분
        grad_cam = np.zeros(target.shape[1:], dtype=np.float32)  # 14X14

        for i, w in enumerate(weights):  # calcul grad_cam
            grad_cam += w * target[i, :, :]  # linear combination L^c_{Grad-CAM}에 해당하는 식에서 ReLU를 제외한 식

        grad_cam = np.maximum(grad_cam, 0)  # 0보다 작은 값을 제거
        grad_cam = cv2.resize(grad_cam, (224, 224))  # 224X224크기로 변환
        grad_cam = grad_cam - np.min(grad_cam)  #
        grad_cam = grad_cam / np.max(grad_cam)  # 위의 것과 해당 줄의 것은 0~1사이의 값으로 정규화하기 위한 정리
        return grad_cam, cam, index, probs, idx,

