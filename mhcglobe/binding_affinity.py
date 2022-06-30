import numpy 

def to_ic50(x, max_ic50=50000.0):
    """ From MHCFlurry Code """
    return max_ic50 ** (1.0 - x)

def from_ic50(ic50, max_ic50=50000.0):
    """ From MHCFlurry Code """
    x = 1.0 - (numpy.log(numpy.maximum(ic50, 1e-12))/numpy.log(max_ic50))
    return numpy.minimum(1.0, numpy.maximum(0.0, x))