import torch as t
from utils.diffeo_container import diffeo_container

def retrieve_layer_activation(model, input, layer_index):
    '''
    Returns activation of specified layers and model output given input.
    The function only works with some vision model and ViT, since it unwraps a specific number of children module. 

    Args:
        model: a torch model to evaluate on
        input: tensor, the input to be fed into the model
        layer_index: a list of integers specifying which layer to retrieve
        
    Returns:
        Tuple of 
            dictionary: key: the layer index, value: the tensor that corresponds to the layer activation
            tensor: output of model, i.e. model(input) 
    '''
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    handles = []

    if len(input) == 3: input = input[None, :, :, :]

    # layers = list(model.children())
    # layers_flat = flatten(layers)
    if hasattr(model, 'encoder'):
        layers_flat = get_flatten_children(model.encoder)
    else:
        layers_flat = get_flatten_children(model)

    for index in layer_index:
        handles.append(layers_flat[index - 1].register_forward_hook(getActivation(index)))

    with t.no_grad(): result = model(input)
    for handle in handles: handle.remove()

    return activation, result

def get_flatten_children(model):
    return flatten(list(model.children()))

def flatten(array):
    result = []
    for element in array:
        if hasattr(element, "__iter__"):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


def inv_diff_hook(inverse_diffeo, batch_size = None):
    if not isinstance(inverse_diffeo, diffeo_container):        
        raise Exception('diffeo is not a diffeo_container')
    def hook(module, input, output):
        if batch_size == None:
        # normal situation
            return inverse_diffeo(output, in_inference=True)
        if batch_size != None:
            # stack in the batch dimension of the steered result
            output = t.unflatten(output, 0, (-1, batch_size))
            ref = output[0]
            output = t.flatten(output, start_dim = 0, end_dim = 1)
            output = t.cat([output, inverse_diffeo(ref, in_inference = True)], dim = 0)
            return output
    return hook