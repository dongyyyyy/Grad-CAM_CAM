from torchvision import models
import argparse
from ImageNet_classes import *
from Grad_cam.Grad_cam_resnet import *
from Grad_cam.Grad_cam_vgg import *
from utils.Image_processing import *
from utils.Make_heatmap import *
from Backprop.BackpropReLU import *
from Backprop.GuidedBackpropReLU import *
import os


model_name = {1:'VGG19',2:'ResNet18',3:'ResNet34'}


def get_args(): # 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available() # True = GPU / False = CPU
    #print(args.use_cuda)
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == '__main__':
    for i in range(1,4):
        os.makedirs("Result_{}".format(model_name[i]),exist_ok=True)
    select_num = int(input('1 : VGG19 / 2 : ResNet18 / 3 : ResNet34 ==>'))
    args = get_args()
    final_layer = 'layer4'
    #final_layer = 'features'
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    #model = models.vgg19(pretrained=True)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    target_index = None

    gradient = []
    features = []
    print("Model name is {}".format(model_name[select_num]))
    if select_num == 1:
        model1 = models.vgg19(pretrained=True)
        model2 = models.vgg19(pretrained=True)
        model3 = models.vgg19(pretrained=True)
        grad_cam = GradCam_vgg(model=model1,target_layer_names=["35"], use_cuda=args.use_cuda)
        bp_model = BackpropReLUModel(model=model2, use_cuda=args.use_cuda)
        gb_model = GuidedBackpropReLUModel_vgg(model=model3, use_cuda=args.use_cuda)
    elif select_num == 2:
        model1 = models.resnet18(pretrained=True)
        model2 = models.resnet18(pretrained=True)
        model3 = models.resnet18(pretrained=True)
        grad_cam = GradCam_resnet(model=model1,target_layer_names=[final_layer],target_sub_layer_names=["conv2"],use_cuda=args.use_cuda)
        bp_model = BackpropReLUModel(model=model2, use_cuda=args.use_cuda)
        gb_model = GuidedBackpropReLUModel_resnet(model=model3, use_cuda=args.use_cuda)
    elif select_num ==3:
        model1 = models.resnet34(pretrained=True)
        model2 = models.resnet34(pretrained=True)
        model3 = models.resnet34(pretrained=True)
        grad_cam = GradCam_resnet(model=model1,target_layer_names=[final_layer],target_sub_layer_names=["conv2"],use_cuda=args.use_cuda)
        bp_model = BackpropReLUModel(model=model2, use_cuda=args.use_cuda)
        gb_model = GuidedBackpropReLUModel_resnet(model=model3, use_cuda=args.use_cuda)
    else :
        print("잘못된 번호 입력")
    mask_gradcam, mask_cam,index, probs, idx = grad_cam(input, target_index)

    if target_index==None: # 지정 target이 없을 경우 상위 2개의 class 정보 출력
        for i in range(0, 2):
            line = 'index : {} class\'s name : {:.3f} -> {}'.format(idx[i].item(),probs[i], ImageNet_classes[idx[i].item()])
            print(line)
    print("Class : ", ImageNet_classes[index])

    save_heatmap_gradcam = show_cam_on_image(img, mask_gradcam)
    if select_num >= 2 and select_num <=3:
        save_heatmap_cam= show_cam_on_image(img,mask_cam)
    cam_mask = cv2.merge([mask_gradcam, mask_gradcam, mask_gradcam])


    bp = bp_model(input, index=target_index)  # 입력에 대한 gradient값을 gb에 저장
    bp = bp.transpose((1, 2, 0))  # RBG -> BGR
    cam_bp = deprocess_image(cam_mask * bp)
    bp = deprocess_image(bp)

    gb = gb_model(input, index=target_index)  # 입력에 대한 gradient값을 gb에 저장
    gb = gb.transpose((1, 2, 0))  # RBG -> BGR
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)



    cv2.imwrite("Result_{}/gradcam.jpg".format(model_name[select_num]), save_heatmap_gradcam)
    if select_num >= 2 and select_num <=3:
        cv2.imwrite("Result_{}/cam.jpg".format(model_name[select_num]), save_heatmap_cam)
    cv2.imwrite('Result_{}/gb.jpg'.format(model_name[select_num]), gb)
    cv2.imwrite('Result_{}/bp.jpg'.format(model_name[select_num]), bp)
    cv2.imwrite('Result_{}/cam_gb.jpg'.format(model_name[select_num]), cam_gb)
    cv2.imwrite('Result_{}/cam_bp.jpg'.format(model_name[select_num]), cam_bp)

