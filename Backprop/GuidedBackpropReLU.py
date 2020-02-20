import torch
import numpy as np
from torch.autograd import Function
from torchsummary import summary

class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        """
        순전파 단계에서는 입력을 갖는 Tensor를 받아 출력을 갖는 Tensor를 반환합니다.
        self는 컨텍스트 객체(context object)로 역전파 연산을 위한 정보 저장에
        사용합니다. self.save_for_backward method를 사용하여 역전파 단계에서 사용할 어떠한
        객체도 저장(cache)해 둘 수 있습니다.
        """
        positive_mask = (input > 0).type_as(input) # 0보다 큰 값들 = 1 그외의 값은 0을 나타내는 위치 mask

        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask) # ReLU연산 0보다 큰 값들의 값은 그대로 그외의 값은 0으로
        #addcmul = element-wise multiplication tensor1 = input / tensor2 = positive_mask / self = torch.zeros
        self.save_for_backward(input, output) # backward에 사용될 값들 저장

        return output


    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors # forward에서 저장된 값들 가져오기


        positive_mask_1 = (input > 0).type_as(grad_output) # 0보다 큰 값들을 1로 (input 기준)
        positive_mask_2 = (grad_output > 0).type_as(grad_output) # 0보다 큰 값들을 1로 (grad_output기준)

        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2) # Guided Backpropagation
        # R_i^l = (f > 0)*(R^{l+1}_i>0)*R^{l+1}_i
        """
        입력위치의 기울기를 구하기에 torch size = input크기에 맞게 
        grad_ouput과 positive_mask_1의 element-wise multiplication - 1 (f>0)*R^{l+1}_i = (1)
        1로 계산된 값과 positive_mask_2의 element-wise multiplication - 2 (R^{l+1}_i>0)*(1)
        2의 결과 값을 grad_input에 저장
        """

        return grad_input


class GuidedBackpropReLUModel_resnet:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model._modules.items():
            if module.__class__.__name__ == 'ReLU': # 해당 layer가 ReLU인 경우
                self.model._modules[idx] = GuidedBackpropReLU.apply
            if module.__class__.__name__ == 'Sequential':
                for block_idx, block_module in module._modules.items(): # Basicblock or Bottlenect 접근
                    for block_sub_idx , block_sub_module in block_module._modules.items(): # Block안에 있는 modules 접근
                        if block_sub_module.__class__.__name__ == 'ReLU':
                            self.model._modules[idx][int(block_idx)]._modules[block_sub_idx] = GuidedBackpropReLU.apply


    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)
        output = input.grad.cpu().data.numpy()
        print(output.shape) # 1 X 3 X 224 X 224 입력 크기
        output = output[0, :, :, :]

        return output



class GuidedBackpropReLUModel_vgg:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU': # 해당 layer가 ReLU인 경우
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        #self.model.features.zero_grad()
        #self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)
        output = input.grad.cpu().data.numpy()
        print(output.shape) # 1 X 3 X 224 X 224 입력 크기
        output = output[0, :, :, :]

        return output