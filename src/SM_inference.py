import unet
import mxnet as mx
import json
import mxnet.ndarray as nd
import logging
import math
import numpy as np
import os 
import gzip
import pickle
import base64
from os import path

class BatchLoader:
    def __init__(self, images, batch_size, ctx, multisp):
        """
        Class to load all images from the given folders
        :param images: list of images
        :param batch_size: int number of images per batch
        :param ctx: mx.gpu(n) or mx.cpu()
        :param multisp: boolean (multispectral) If true, images will be read from .npy files,
                        and only the first 3 bands will be used for cloud classification.
        """
        self.ctx = ctx
        self.batch_size = batch_size
        self.images = []
        self.multisp = multisp

        self.images=images

        if self.images:
            self.channels, self.imgsize, _ = self._read_img(self.images[0]['data']).shape

        logging.info("Found a total of {} images".format(len(self.images)))

    def __len__(self):
        return len(self.filenames)

    def _preprocess(self, data):
        data = nd.array(data).astype('float32').as_in_context(self.ctx)
        if not self.multisp:
            print ("NOT SELF MULTISP")
            data = data / 255
            data = nd.transpose(data, (2, 0, 1))
        return data

    def _read_img(self, img):
        return self._preprocess(img)

    def _load_batch(self, images):
        batch = mx.nd.empty((len(images), self.channels, self.imgsize, self.imgsize), self.ctx)
        for idx, fn in enumerate(images):
            batch[idx] = self._read_img(fn['data'])
        return batch

    def get_batches(self):
        for n in range(int(math.ceil(len(self.images)/self.batch_size))):
            if (n + 1) * self.batch_size <= len(self.images):
                files_batch = self.images[n * self.batch_size:(n + 1) * self.batch_size]
            else:
                files_batch = self.images[n * self.batch_size:]

            yield self._load_batch(files_batch)



#Load Model
def model_fn(model_dir):
    """function used to load pretrained model"""
    ctx =  mx.cpu()
    net = unet.Unet()
    print ("Loading", model_dir)
    if path.exists(model_dir+"/unet_RGB.params"):
        print ("Loading RGB Model")
        net.load_params(model_dir+"/unet_RGB.params", ctx)
        print ("RGB Model Loaded")
        
    elif path.exists(model_dir+"/unet_ALL_BANDS.params"):
        print ("Loading ALL_BANDS Model")
        net.load_params(model_dir+"/unet_ALL_BANDS.params", ctx)
        print ("ALL_BANDS Model Loaded")
        
    else:
        print ("Model Missing")
        net=None
    return (net)
    
def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    ctx =  mx.cpu() 
    batch_size=64
    output=[]
    print ("Start Parsing input")
    parsed = json.loads(data)
    print ("End Parsing input")
    images=[]
    multisp=False
    
    #Check for Multispectral input type
    if 'type' in parsed.keys():
        if parsed['type'].lower() == 'rgb':
            multisp=False
            job_data=parsed['instances']
        else:
            multisp=True
            encodedBytes=parsed['instances'].encode("utf-8")
            zip_value=base64.b64decode(encodedBytes)
            dump_value=gzip.decompress(zip_value)
            job_data=pickle.loads(dump_value)
            
    print ("Multispacial",multisp)
            
    for item in job_data:
        image_data=np.array(item['data'])
        print ("Input Image Shape:", image_data.shape)
        images.append ({'data':image_data})
    
    print ("MultiSP:", multisp)
    loader=BatchLoader(images,64,ctx,multisp)
    
    for idxb, batch in enumerate(loader.get_batches()):
        preds = nd.argmax(net(batch), axis=1)
        for pred in preds:
            output.append({'data':pred.asnumpy().astype('uint8').tolist() })

    response_body = json.dumps({'predictions':output})
    return response_body, output_content_type
