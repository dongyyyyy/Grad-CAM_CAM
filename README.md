# Grad-CAM_CAM - pytorch

---

## Grad-CAM & CAM

---

### 작업 환경

#### - Pytorch : 1.3

#### - CUDNN : 7.6.5

#### - CUDA Toolkit : 10.1

### 필요라이브러리

#### - opencv(cv2) : pip install opencv-python
#### - numpy : conda install numpy
#### - torchsummary(필수X) : pip install torchsummary

---

### Main 실행시 하나의 정수 입력을 받음. 
1. VGG19
2. ResNet18
3. ResNet34

### 선택한 모델로 동작하여 아래와 같은 결과를 얻을 수 있다.
 1. CAM
 2. Grad-CAM
 3. Backprop_img
 4. Backprop_img + Grad-CAM
 5. GuidedBackprop_img
 6. GuidedBackprop_img + Grad-CAM
### 각 결과는 해당 모델의 이름과 일치하는 폴더에 모두 저장된다.

### 자세한 사항은 [다음](https://dydeeplearning.tistory.com/10) 에서 얻을 수 있다.
