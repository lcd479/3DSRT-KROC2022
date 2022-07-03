# 3D Scene Reconstruction Using Transformer in Monocular

<br/>

> <h6>Transformer를 활용한 단안 영상 3D 장면 재구성(3D Scene Reconstruction Using Transformer in Monocular)   
> <h6>ChangDae Lee, Kwang-Hyun Park   
> <h6>[KRoC 2022]한국로봇종합학술대회-일반논문   

<br/>

## Network Architecture
<img width="100%" src="https://user-images.githubusercontent.com/63233441/177045936-6a208ca8-f184-4a5d-91cd-787a52794b7d.png"/>



## Results
<h5> Qualitative Results <h5/>
<img width="80%" src="https://user-images.githubusercontent.com/63233441/177045919-2408f2b3-cef6-4f3e-b87f-c3c7659d43eb.png"/>
  
<h5> Quantitative Results <h5/>
  
| Method | Complete | Recall | Precision | F-Score |
| ---| --- | --- |--- |  ---|
|NeuralRecon|	0.892878|	0.361921|<span style ="color.red">**0.74732**</span>|	0.487668|
|ST-S|	**0.954388**|	0.177450|	0.34948|	0.23538|
|ST-B(Ours)|	0.897876|	**0.376966**|	0.73736|	**0.498885**|

  
  
  
<br/>  

## Training on ScanNet
  
```bash
  python -m torch.distributed.launch --nproc_per_node=1 Train.py --cfg ./config/train.yaml
```
  
  
<br/>
 
## Acknowledgment
  
>Some of the code in this repo is borrowed from [NeuralRecon](https://github.com/zju3dv/NeuralRecon), thanks Jiaming Sun!   
>Some of the code in this repo is borrowed from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), thanks Ze Liu!
