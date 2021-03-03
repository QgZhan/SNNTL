# Effective Transfer Learning Algorithm in Spiking Neural Networks.
Python implementation of transfer learning on SNNs with the centered kernel alignment (CKA). 

## Requirement
- Python 3.7
- Pytorch 1.7.1+cu101
- prefetch_generator 1.0.1
- tqdm 4.54.1

## Training and testing
To run the codes on transfer learning dataset in /transfer_data, you should first download the corrosponding dataset file. 
Then, you should put the samples into different folders as each folder requires (e.g., /transfer_data/PACS/giraffe/image_names.txt). 

To do so, you can run the following command to train and test the model:

`$ python snn_tl.py`

