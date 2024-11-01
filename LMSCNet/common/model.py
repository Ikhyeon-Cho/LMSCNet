from LMSCNet.models.LMSCNet import LMSCNet
from LMSCNet.models.LMSCNet_SS import LMSCNet_SS
from LMSCNet.models.SSCNet_full import SSCNet_full
from LMSCNet.models.SSCNet import SSCNet


def get_model(model_type, dataset):
    """
    Since the model requires 
    1. which model to be used,
    2. number of classes,
    3. grid dimensions,
    4. class frequencies,
    we use the dataset to get these parameters.
    """

    nbr_classes = dataset.nbr_classes
    grid_dimensions = dataset.grid_dimensions
    class_frequencies = dataset.class_frequencies

    # LMSCNet ----------------------------------------------------------------------------------------------------------
    if model_type == 'LMSCNet':
        model = LMSCNet(class_num=nbr_classes, input_dimensions=grid_dimensions,
                        class_frequencies=class_frequencies)
    # ------------------------------------------------------------------------------------------------------------------

    # LMSCNet_SS -------------------------------------------------------------------------------------------------------
    elif model_type == 'LMSCNet_SS':
        model = LMSCNet_SS(class_num=nbr_classes, input_dimensions=grid_dimensions,
                           class_frequencies=class_frequencies)
    # ------------------------------------------------------------------------------------------------------------------

    # SSCNet_full ------------------------------------------------------------------------------------------------------
    elif model_type == 'SSCNet_full':
        model = SSCNet_full(class_num=nbr_classes)
    # ------------------------------------------------------------------------------------------------------------------

    # SSCNet -----------------------------------------------------------------------------------------------------------
    elif model_type == 'SSCNet':
        model = SSCNet(class_num=nbr_classes)
    # ------------------------------------------------------------------------------------------------------------------

    else:
        assert False, 'Wrong model selected'

    return model
