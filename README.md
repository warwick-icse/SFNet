## [Multi-fidelity modeling of wind farm wakes based on a novel super-fidelity network](https://www.sciencedirect.com/science/article/pii/S0196890422009633) 

*Energy Conversion and Management*, 2022, [Rui Li](https://lironui.github.io/),  [Jincheng Zhang](https://www.linkedin.com/in/this-is-jincheng-zhang/) and [Xiaowei Zhao](https://warwick.ac.uk/fac/sci/eng/people/xiaowei_zhao/).

## Introduction

This project (SFNet) is the pretrained model and test code for [Multi-fidelity modeling of wind farm wakes based on a novel super-fidelity network](https://www.sciencedirect.com/science/article/pii/S0196890422009633).

## Preparation

python==3.6.13  <br />
torch==1.10.2  <br />
torchvision==0.11.3  <br />
floris==2.4  <br />
pandas==1.1.5  <br />
numpy==1.19.5

## Visualization

After installing required libraries mentioned above, then you can run the ```test.py``` based on provided low-fidelity flow fields generated by FLORIS (8 m/s, 9 m/s and 10 m/s). We provide three pretrained models which are trained based on 45, 90 and 135 samples (you can choose different models by changing the value of ```pre_trained_sample``` in test.py). To test different wind speeds, you need to change of the value of ```wind_speed``` in test.py.

Or you can generate your own data using ```gene_floris_farm.py```. 

## Results

![FLORIS](https://github.com/warwick-icse/SFNet/blob/main/result/floris.png) 
![SFNet](https://github.com/warwick-icse/SFNet/blob/main/result/sfnet.png) 

## Citation

If you find this project useful in your research, please consider citing our paper：

@article{ <br />
&ensp; &ensp; &ensp; &ensp; li2022multi,  <br />
&ensp; &ensp; &ensp; &ensp; title={Multi-fidelity modeling of wind farm wakes based on a novel super-fidelity network}, <br />
&ensp; &ensp; &ensp; &ensp; author={Li, Rui and Zhang, Jincheng and Zhao, Xiaowei}, <br />
&ensp; &ensp; &ensp; &ensp; journal={Energy Conversion and Management}, <br />
&ensp; &ensp; &ensp; &ensp; volume={270}, <br />
&ensp; &ensp; &ensp; &ensp; pages={116185}, <br />
&ensp; &ensp; &ensp; &ensp; year={2022}, <br />
&ensp; &ensp; &ensp; &ensp; publisher={Elsevier} <br />
}

## Acknowledgement

- [SOWFA](https://www.nrel.gov/wind/nwtc/sowfa.html)
- [FLORIS](https://github.com/NREL/floris)
