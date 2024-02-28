from omegaconf import OmegaConf


def get_in_c_neck(target):
    key = target.split(".")[-1]
    if key == "EfficientNet":
        return [56, 160]
    elif key == "Encoder_res50":
        return [512, 1024]
    elif key == "Encoder_res101":
        return [512, 1024]
    elif key == "WrapVisionTransformer":
        return [768]
    else:
        raise NotImplementedError


def get_neck_interm_c(target):
    key = target.split(".")[-1]
    if key == "EfficientNet":
        return 128
    elif key == "Encoder_res50":
        return 512
    elif key == "Encoder_res101":
        return 512
    elif key == "WrapVisionTransformer":
        return 768
    else:
        raise NotImplementedError


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))
OmegaConf.register_new_resolver("sumlist", lambda x: sum(x))
OmegaConf.register_new_resolver("get_in_c_neck", lambda x: get_in_c_neck(x))
OmegaConf.register_new_resolver("get_neck_interm_c", lambda x: get_neck_interm_c(x))
