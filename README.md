# Common-Neural-Network-Models
## Run method
As the network is very simple all you need is just run the related python file on your own computer without any parameters  
for DCGAN data, you could download it from : https://drive.google.com/file/d/0BxgrWCfoOvPWZkpmOHpwY1ZQM0k/view?usp=sharing  
extract it under the data folder then you could   
Train:  
python main.py --input_height 96 --input_width 96 --output_height 48 --output_width 48 --dataset anime --crop --train --epoch 300 --input_fname_pattern "*.jpg"  
Test:  
python main.py --input_height 96 --input_width 96 --output_height 48 --output_width 48 --dataset anime --crop --epoch 300 --input_fname_pattern "*.jpg"  

