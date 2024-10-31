# Machine Learning Project Title

이 프로젝트는 양질/다량의 데이터셋 확보가 어려운 지면 투과 후방산란 엑스선 영상의 특성에 적응하기 위해,
Single Image Super Resolution (SISR) 변환을 위하여, 지능형부품센서연구실에서 개발한 딥러닝 SW 입니다. 
주요 기능 및 특징은 다음과 같습니다. 

- 기능 : Single image 입력, Single image 출력 기반의 빠르고 간편한 노이즈 제거 및 초해상 변환
- 특징 : 메모리가 부족한 일반 보급형 GPU 에서도 동작 가능한 경량화 모델

<br>

## Table of Contents
- [Prerequisites](#prerequisites)
- [Data](#data)
- [Inference](#inference)
- [Authors](#authors)

<br>

## Prerequisites

- ML 프레임워크 : CUDA 11.6, Pytorch 1.12.1, torchvision 0.13.1 torchaudio 0.12.1, BasicSR 1.3.4.9
- DRCT 설치 방법 : 
```
git clone https://github.com/ming053l/DRCT.git
conda create --name drct python=3.8 -y
conda activate drct
cd DRCT
pip install einops basicsr==1.3.4.9 cython schedule watchdog
python setup.py develop
substitute 폴더 아래 python 파일을 원본경로와 1:1 대체
- (원본경로) C:\Users\admin\anaconda3\envs\gpx_sisr\Lib\site-packages\basicsr\test.py
- (대체파일) substitute\test.py
- (원본경로) C:\Users\USER\.conda\envs\drct\Lib\site-packages\basicsr\data\__init__.py
- (대체파일) substitute\__init__.py
- (원본경로) C:\Users\USER\.conda\envs\drct\Lib\site-packages\basicsr\data\single_image_dataset.py
- (대체파일) substitute\single_image_dataset.py
- (원본경로) C:\Users\USER\.conda\envs\drct\Lib\site-packages\basicsr\data\img_util.py
- (대체파일) substitute\img_util.py
- (원본경로) ./DRCT/drct/models/drct_model.py
- (대체파일) substitute\drct_model.py
DRCT_pretrained_model 폴더 아래 model.pth 파일 이동
```
- 학습 데이터셋 : 연구사업에서 자체 구축한 데이터셋 (후방산란 엑스선 영상)
- 컴퓨팅 환경 : NVIDIA RTX3090 GPU (또는 RAM 16GB 이상의 NVIDIA GPU)

<br>

## Data

#### 데이터셋 설명

**데이터셋 내용 및 구조**

SISR 에서는 raw, tiff, bin 확장자를 가진 후방산란 엑스선 영상을 데이터셋으로 사용합니다.

본 데이터셋은 연구사업에서 자체적으로 구축한 데이터셋입니다. 

샘플 raw 영상을 함께 제공합니다.

<br>

**데이터셋 소스 정보**

추가 데이터셋 보유 부서 및 연락처 : 지능형부품센서연구실 최성훈 (schoi@etri.re.kr) 

<br>

#### 데이터 처리 방법

**데이터 준비, 전처리 절차**

이미지 데이터셋의 준비, 전처리 절차는 다음과 같습니다.
본 프로젝트에서는 DRCT 학습을 추가로 진행하지 않았습니다.

1. 데이터셋을 준비한다.
2. parameters 폴더 안의 DRCT_SRx4_GPX.yml 을 참고하여 자신의 설정에 맞게 파라미터를 수정한다.
3. 프로그램을 실행하여 데이터셋을 추론한다.
4. 결과 영상을 저장한다.


<br>


## Inference

####  추론 방법

이미지 인식을 위한 학습 모델을 평가하기 위한 방법 및 추론 구조는 다음과 같습니다.

<br>

**코드 실행 방법**

새로운 데이터로 모델 테스트(추론)를 위한 Python 코드의 파일명은 run.py 입니다.  코드 실행 절차는 다음과 같습니다.
* 주피터 노트북에서 실행 방법 또는
* Python 스크립트로 실행 방법 또는
* IDE(PyCharm)에서 실행 방법 또는
* Docker 이미지의 실행 방법

<br>

## Authors
* 최성훈 &nbsp;&nbsp;&nbsp;  schoi@etri.re.kr   

<br>
<br>
<br>


## Version
* 1.0
<br>
<br>


## Thanks
* DRCT (https://github.com/ming053l/DRCT)
* BasicSR (https://github.com/XPixelGroup/BasicSR)
<br>
