# [filter size, stride, padding]
#Assume the two dimensions are the same
#Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
#Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

import numpy as np
from pdb import set_trace


layerInfos = [] # global


def _outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]
    if isinstance(p, str):
        p = {'same':1, 'valid':0, 'causal':1}.get(p.lower())
    
    if s > 1-1E-9: # pooling
        n_out = np.floor((n_in - k + 2*p)/s)+1
    else: # upsampling
        n_out = np.ceil((n_in-k + 2*p+1)/s)
    actualP = (n_out-1)*s - n_in + k
    pL = np.ceil(actualP/2)
    pR = np.floor(actualP/2)

    j_out = j_in * s
    r_out = r_in + (k - 1)*j_in
    start_out = start_in + ((k-1)/2 - pL)*j_in
    return n_out, j_out, r_out, start_out

def _printLayer(layer, layer_name, conv):
    if len(conv) > 1:
        if "conv" in layer_name.lower(): #pooling
            print(layer_name + ": kernel={}  stride={}  padding={}".format(*conv))   
        elif 'upsample' in layer_name.lower():
            print(layer_name + ": kernel={}  size={:.0f}  padding={}".format(conv[0], 1/conv[1], conv[2]))   
        else: #pool
            print(layer_name + ": kernel={}  pool_size={}  padding={}".format(*conv))
    else: # input layer
        print(layer_name + ": ")
    print("\t n features: {:.0f} \n \t jump: {:.0f} \n \t receptive size: {:.0f} \t start: {:.1f}".format(*layer))

def ReceptiveField_Conv2D(convnet, layer_names, imsize, calc_all=False, query_centers=False):
    """
    Assume the two dimensions are the same
    Each kernel requires the following parameters:
     - k_i: kernel size
     - s_i: stride
     - p_i: padding (if padding is uneven, right padding will be higher than left padding;
                    "SAME" option in tensorflow)

    Each layer i requires the following parameters to be fully represented:
     - n_i: number of feature (data layer has n_1 = imagesize )
     - j_i: distance (projected to image pixel distance) between center of two adjacent features
     - r_i: receptive field of a feature in layer i
     - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

    * Inputs:
        - convnet: [[kernel_size, stride, padding], [], ...]
        - layer_names: list of names of the layers
        - imsize: image size, single integer
        - query_centers: whether or not get details on where the centers of the receptive fields are [False]
        - calc_all: When query_centers=True, whether or not calculate every single receptive field and 
            its center for all neurons. Default False, which uses user prompts.
        
    * Outputs:
        Only when set calc_all = True. For each layer, returns a dictionary 
        with the following fields (dictionary of dictionary)
        - centers: a n_i x n_i x 2 matrix of receptive field centers (pair of coordinates 
                    in the last dimension)
        - receptive: receptive field size
        - n: n_i, explained above
        - j: j_i, explained above
        - start: start_i, explained above

    For example:
    ```
    convnet =   [[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0], [1, 1, 0]]
    layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv', 'fc7-conv']
    imsize = 227
    layer_dict = ReceptiveField_Conv2D(convnet, layer_names, imsize, calc_all=True)
    ```
    """
    #first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    print ("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]

    _printLayer(currentLayer, "input image", [imsize])
    # Going through each layer and calcualte the infos
    for i in range(len(convnet)):
        currentLayer = _outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        _printLayer(currentLayer, layer_names[i], convnet[i])
    print ("------------------------")
    
    # Calculate the receptive fields
    if not query_centers:
        return
    
    if not calc_all:
        layer_name = input("Layer name where the feature in: ") # prompt user
        layer_idx = layer_names.index(layer_name)
        idx_x = int(input("index of the feature in x dimension (from 0)"))
        idx_y = int(input ("index of the feature in y dimension (from 0)"))
    
        n = layerInfos[layer_idx][0]
        j = layerInfos[layer_idx][1]
        r = layerInfos[layer_idx][2]
        start = layerInfos[layer_idx][3]
        assert(idx_x < n)
        assert(idx_y < n)
        
        print("\n (x, y) = ({:.0f} , {:.0f})".format(idx_x, idx_y))
        print ("receptive field: ({}, {})".format(r, r))
        print ("center: ({:f}, {:f})".format(start+idx_x*j, start+idx_y*j))
    else: # calcualte all of them
        layer_dict = {}
        for layer_name in layer_names:
            layer_dict[layer_name] = {}
            layer_idx = layer_names.index(layer_name)
            n = int(layerInfos[layer_idx][0])
            j = layerInfos[layer_idx][1]
            r = layerInfos[layer_idx][2]
            start = layerInfos[layer_idx][3]
            
            xx, yy = np.meshgrid(range(n), range(n), indexing='ij')
            centers = np.stack([xx, yy], axis=2)
            centers = start + centers * j
            
            #centers = np.empty((n, n, 2))
            #for idx_x in range(n):
            #    for idx_y in range(n):
            #        centers[idx_x, idx_y, :] = start+idx_x*j, start+idx_y*j
            
            layer_dict[layer_name]['centers'] = centers
            layer_dict[layer_name]['receptive'] = [r, r]
            layer_dict[layer_name]['n'] = n
            layer_dict[layer_name]['j'] = j
            layer_dict[layer_name]['start'] = start
            
        return layer_dict
        
if __name__ == '__main__': 
    
    #Conv:[kernel_size, stride=1, padding=1('same')]
    #Pool: [kernel_size (from Conv), pool_size, padding=1('same')]
    #UpSampling: [kernel_size, 1/size, padding=1('same')]
    layer_names = ['conv-1', 'pool-1', 'conv0','pool0', 
                   'conv1','pool1','conv2','pool2',
                   'conv3', 'upsample3', 'conv4', 'upsample4', 
                   'conv5', 'upsample5', 'conv6', 'upsample6', 'conv7']
    k = 3
    convnet =  [[k, 1, 'same'], [k, 2, 'same'], [k, 1, 'same'], [k, 2, 'same'],
                [k, 1, 'same'], [k, 2, 'same'], [k, 1, 'same'], [k, 2, 'same'], 
                [k, 1, 'same'], [k, 1/2, 1], [k, 1, 'same'], [k, 1/2, 'same'], 
                [k, 1, 'same'], [k, 1/2, 1], [k, 1, 'same'], [k, 1/2, 1], [k, 1, 'same']]
    
    
    #layer_names = ['conv1','pool1','conv2','pool2','conv3', 'upsample3', 'conv4', 'upsample4', 'conv5']
    # = 3
    #convnet =  [[k, 1, 'same'], [k, 2, 'same'], [k, 1, 'same'], [k, 2, 'same'], 
    #            [k, 1, 'same'], [k, 1/2, 1], [k, 1, 'same'], [k, 1/2, 'same'], 
    #            [k, 1, 'same']]
    
    imsize = 16384

    #convnet =   [[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0], [1, 1, 0]]
    #layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv', 'fc7-conv']
    #imsize = 227
    layer_dict = ReceptiveField_Conv2D(convnet, layer_names, imsize, calc_all=False)
    
    
