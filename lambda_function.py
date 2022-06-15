import warnings
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
import time
import numpy as np

import os
import boto3
BUCKET_NAME = os.environ.get('BUCKET_NAME')

ctx = mx.cpu()


def timer(thunk, repeat=1, number=10, dryrun=3, min_repeat_ms=1000):
    """Helper function to time a function"""
    for i in range(dryrun):
        thunk()
    ret = []
    for _ in range(repeat):
        while True:
            beg = time.time()
            for _ in range(number):
                thunk()
            end = time.time()
            lat = (end - beg) * 1e3
            if lat >= min_repeat_ms:
                break
            number = int(max(min_repeat_ms / (lat / number) + 1, number * 1.618))
        ret.append(lat / number)
    return ret

def load_model(model_name):
    s3_client = boto3.client('s3') 

    os.makedirs(os.path.dirname(f'/tmp/{model_name}/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'mxnet/base/{model_name}/model-symbol.json', f'/tmp/{model_name}/model-symbol.json')
    s3_client.download_file(BUCKET_NAME, f'mxnet/base/{model_name}/model-0000.params', f'/tmp/{model_name}/model-0000.params')

    model_json = f"/tmp/{model_name}/model-symbol.json"
    model_params = f"/tmp/{model_name}/model-0000.params"

    if model_name == "bert_base":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gluon.nn.SymbolBlock.imports(model_json, ['data0','data1','data2'], model_params, ctx=ctx)

    elif model_name == "distilbert":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gluon.nn.SymbolBlock.imports(model_json, ['data0','data1'], model_params, ctx=ctx)
    return model 

def benchmark(model_name,batch_size,seq_length,dtype='float32'):
    model = load_model(model_name)
    inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
    token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
    valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
        
    inputs_nd = mx.nd.array(inputs, ctx=ctx)
    token_types_nd = mx.nd.array(token_types, ctx=ctx)
    valid_length_nd = mx.nd.array(valid_length, ctx=ctx)

    if model_name == "bert_base":
        # Prepare input data
        model.hybridize(static_alloc=True)
        mx_out = model(inputs_nd, token_types_nd, valid_length_nd)
        mx_out.wait_to_read()
        res = timer(lambda: model(inputs_nd,token_types_nd,valid_length_nd).wait_to_read(),
                    repeat=3,
                    dryrun=5,
                    min_repeat_ms=1000)

    elif model_name == "distilbert":
        model.hybridize(static_alloc=True)  
        mx_out = model(inputs_nd, valid_length_nd,)
        mx_out.wait_to_read()

        # Benchmark the MXNet latency
        res = timer(lambda: model(inputs_nd, valid_length_nd).wait_to_read(),
                    repeat=3,
                    dryrun=5,
                    min_repeat_ms=1000)
    
    print(f"MXNet {model_name} latency for batch {batch_size} : {np.mean(res):.2f} ms")

    return res 

def lambda_handler(event, context):
    
    model_name = event['model_name']
    batchsize = event['batchsize']
    seq_length = event['seq_length']

    start_time = time.time()
    inference_time = benchmark(model_name,batchsize,seq_length)
    running_time = time.time() - start_time
    return {'handler_time': running_time , 'inference_time':inference_time}


BUCKET_NAME = 'dl-converted-models'
model_name = 'bert_base'
batchsize = 1
seq_length = 128

start_time = time.time()
inference_time = benchmark(model_name,batchsize,seq_length)
running_time = time.time() - start_time
print('handler_time', running_time , 'inference_time',inference_time)
