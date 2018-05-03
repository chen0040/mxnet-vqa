# mxnet-vqa

Yet another visual question answering in MXNet 

# Overview

For those of you who studied the [http://gluon.mxnet.io](http://gluon.mxnet.io). You might think that the
author of [http://gluon.mxnet.io](http://gluon.mxnet.io) already has a 
[VQA Chapter](http://gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html) on how to implement
a visual question answering (VQA) in MXNet, why bother create another project on this?  
However, for someone who tries to recreate a working code from
the [VQA Chapter](http://gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html) of the ebook,
one might find that the sample codes illustrated in the ebook contains some issues that need to be addressed (
for example, F was used for a gluon.nn.Block forward() method and was not declared in the import either, 
the MCB implementation also has some problem in terms of the S and H second dimension, there is no detail
handling on that as well). Furthermore, the question matrix, feature matrix was pre-computed using pre-trained
model and directly downloaded from mxnet.io in the codes without showing how they can be implemented.

As a result, I decided to publish this project which contains the working codes of VQA in MXNet. This project
is both a reproduction of the [VQA Chapter](http://gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html)
and more. Instead of just implementing the algorithms in the [VQA Chapter](http://gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html), 
I decided to try several ideas of my own, which are describe below.

Due to the limitation in terms of both hardware and time to train the model, I decided to use only the validation
set of the MS COCO 2014 dataset (and only the first 100,000 records) as both the training set and validation set
for the training codes in the [demo](demo) folder, But you are welcomed to use the much large MS COCO 2014 training set (
The MS COCO data loader and feature extractor are contained in the [mxnet_vqa/data](mxnet_vqa/data) folder).  

# Implementation

### Generate image features

In this implementation, VGG16 feature extractor codes are included which converts val2014 images into
(?, 1000) matrix. The VGG16 feature extractor codes can be found in 

* [mxnet_vqa/data/coco_images.py](mxnet_vqa/data/coco_images.py)
* [mxnet_vqa/utils/image_utils.py](mxnet_vqa/utils/image_utils.py)

Note that before you can run the codes you need to download val2014 from [COCO](http://cocodataset.org/#download)
and extract the images in the downloaded compressed file into [demo/data/coco/val2014] folder. 

Also before you run any training or testing codes, it is recommended that you build the pre-computed image features first by running the 
[build_image_feats.py](demo/build_image_feats.py):

```bash
python demo/build_image_feats.py
```  

### Generate question matrix

In ths implementation, the glove downloader and loader codes are included which converts the questions into
glove embedding encoded matrix. The codes can be found in 

* [mxnet_vqa/data/coco_questions.py](mxnet_vqa/data/coco_questions.py)
* [mxnet_vqa/data/glove.py](mxnet_vqa/data/glove.py)
* [mxnet_vqa/utils/glove_loader.py](mxnet_vqa/utils/glove_loader.py)

Before you run any training or testing codes, it is recommended that you download the glove embedding first before by running
the [build_glove_embedding.py](demo/build_glove_embedding.py):

```bash
python demo/build_glove_embedding.py
``` 



### Handle varying-length of the questions

In the original implementation of the [VQA Chapter](http://gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html), bucket with
a customized data iterator was used to produce fixed length question for batching. The implementation looks
complex, therefore I borrow two concepts I learned from working on Keras for NLP and NLU as alternative approaches.

* The first approach is to re-implement the pad_sequences() method available in Keras. This requires to set the
question_mode to 'concat' when generating the question matrix;
* The second approach uses a summation of individual glove embedding of the corresponding word, collapsing a shape=(seq_length, 300) question tensor
where 300 is the glove embedding dimension and seq_length is the length of the question, which varies from
questions to questions)  to a shape=(300, ) tensor. This requires to set the question_mode to 'add' when generating
the question matrix

### VQA1: MLP Network as VQA Network

The first VQA net implementation is in [mxnet_vqa/library/vqa1.py](mxnet_vqa/library/vqa1.py) and is the 
implementation of Net1 as outlined in [VQA Chapter](http://gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html).
The main difference is the this network accepts two version of question matrix batch:

* version = '1': this uses question_mode = 'add', which reduces a varying-length question tensor to a (300, ) tensor (the second 
approach mentioned in the previous section). The training code is within [vqa1_v1_train.py](demo/vqa1_v1_train.py)
* version = '2': this uses question_mode = 'concat' and set max_question_seq_length = 10, which uses the Keras-inspired
pad_sequences approach to reduce a varying-length question to a fixed dimension tensor. The training code is
within [vqa1_v2_train.py](demo/vqa1_v2_train.py)

To run the training code, for example, one can run the following command:

```bash
python demo/vqa1_v1_train.py
```

After training, the trained models will be saved into the [demo/models](demo/models) folder.

To test the trained mode, one can run the script [vqal1_v1_test.py](demo/vqa3_v1_test.py):

```bash
python demo/vqa1_v1_test.py
```


### VQA2: MCB in VQA Network

The second VQA net implementation corresponds to the Net2 outlined in [VQA Chapter](http://gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html).
The implemented version has some difference from Net2 implementation outlined:

* The number of 1s added to S and H for the sketch count is fixed and concatenated with the image-feats and question matrix
* The dimension of S and H are different for x1 and x2 (The Net2 uses the same dimension of 3072, which I don't
seem to see the reason why it is needed for the sketch count to work)

The training code is within [vqal2_v1_train.py](demo/vqa2_v1_train.py)

To run the training code, for example, one can run the following command:

```bash
python demo/vqa2_v1_train.py
```

After training, the trained models will be saved into the [demo/models](demo/models) folder.

To test the trained mode, one can run the script [vqal2_v1_test.py](demo/vqa3_v1_test.py):

```bash
python demo/vqa2_v1_test.py
```


### VQA3: Incorporate Recurrent Network into the VQA Network

The question structure is a sequence of characters in nature. Therefore, one idea that I try is to feed the
 question batch into a LSTM layer followed by a dense layer before concatenating with the image features. 
 The implementation of this can be found in 
 
* [mxnet_vqa/library/vqa3.py](mxnet_vqa/library/vqa3.py)

The training code is within [vqal3_v1_train.py](demo/vqa3_v1_train.py)

To run the training code, for example, one can run the following command:

```bash
python demo/vqa3_v1_train.py
```

After training, the trained models will be saved into the [demo/models](demo/models) folder.

To test the trained mode, one can run the script [vqal3_v1_test.py](demo/vqa3_v1_test.py):

```bash
python demo/vqa3_v1_test.py
```



Below shows a sample of the output for running the test:

```text
image:  C:/Users/chen0/git/mxnet-vqa/demo\data/coco\val2014\COCO_val2014_000000197683.jpg
question is:  How many people are on the motorcycle?
predicted:  1 actual:  2
image:  C:/Users/chen0/git/mxnet-vqa/demo\data/coco\val2014\COCO_val2014_000000460312.jpg
question is:  What is on the screen?
predicted:  ski poles actual:  icons
image:  C:/Users/chen0/git/mxnet-vqa/demo\data/coco\val2014\COCO_val2014_000000245153.jpg
question is:  What kind of birds are these?
predicted:  ski poles actual:  puffin
image:  C:/Users/chen0/git/mxnet-vqa/demo\data/coco\val2014\COCO_val2014_000000099053.jpg
question is:  What color is the tablecloth?
predicted:  blue actual:  blue
image:  C:/Users/chen0/git/mxnet-vqa/demo\data/coco\val2014\COCO_val2014_000000289941.jpg
question is:  In which hand is the remote?
predicted:  right actual:  right
image:  C:/Users/chen0/git/mxnet-vqa/demo\data/coco\val2014\COCO_val2014_000000433984.jpg
question is:  Are his socks and shoes the same brand?
predicted:  yes actual:  yes
image:  C:/Users/chen0/git/mxnet-vqa/demo\data/coco\val2014\COCO_val2014_000000429261.jpg
question is:  Is this considered fine dining?
predicted:  no actual:  no
image:  C:/Users/chen0/git/mxnet-vqa/demo\data/coco\val2014\COCO_val2014_000000397045.jpg
question is:  Is this photo sideways?
predicted:  no actual:  yes
image:  C:/Users/chen0/git/mxnet-vqa/demo\data/coco\val2014\COCO_val2014_000000456239.jpg
question is:  What is she holding on her hand?
predicted:  umbrella actual:  umbrella
```



# Note

### Training with GPU

Note that the default training scripts in the [demo](demo) folder use GPU for training, therefore, you must configure your
graphic card for this (or remove the "model_ctx=mxnet.gpu(0)" in the training scripts). 


* Step 1: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (you should download CUDA® Toolkit 9.0)
* Step 2: Download and unzip the [cuDNN 7.0.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 
