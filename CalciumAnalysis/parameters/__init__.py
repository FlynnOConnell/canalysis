import yaml

__all__ = []


def get_params():
    return Params(parameters)


class Params:
    def __init__(self, parameter_dict):
        for key in parameter_dict:
            setattr(self, key, parameter_dict[key])


# load parameters
with open(
        r'C:\Users\dilorenzo\Documents\repos\CalciumAnalysis\CalciumAnalysis\parameters'
        r'\params.yaml',
        'rb') as f:
    parameters = yaml.safe_load(f.read())

get_params()
