from .phtrans import PHTrans


def build_coarse_model(config, is_VAL = False):
    if config.MODEL.COARSE.TYPE == 'phtrans':
        model = PHTrans(
            img_size = config.DATASET.COARSE.SIZE, 
            base_num_features = config.MODEL.COARSE.BASE_NUM_FEATURES, 
            num_classes = config.DATASET.COARSE.LABEL_CLASSES, 
            num_only_conv_stage = config.MODEL.COARSE.NUM_ONLY_CONV_STAGE, 
            num_conv_per_stage = config.MODEL.COARSE.NUM_CONV_PER_STAGE,
            feat_map_mul_on_downscale = config.MODEL.COARSE.FEAT_MAP_MUL_ON_DOWNSCALE,  
            pool_op_kernel_sizes = config.MODEL.COARSE.POOL_OP_KERNEL_SIZES,
            conv_kernel_sizes = config.MODEL.COARSE.CONV_KERNEL_SIZES, 
            dropout_p = config.MODEL.COARSE.DROPOUT_P,
            deep_supervision = config.MODEL.DEEP_SUPERVISION if not is_VAL else False,
            max_num_features = config.MODEL.COARSE.MAX_NUM_FEATURES, 
            depths = config.MODEL.COARSE.DEPTHS, 
            num_heads = config.MODEL.COARSE.NUM_HEADS,
            window_size = config.MODEL.COARSE.WINDOW_SIZE, 
            mlp_ratio = config.MODEL.COARSE.MLP_RATIO, 
            qkv_bias = config.MODEL.COARSE.DROP_RATE, 
            qk_scale = config.MODEL.COARSE.QK_SCALE,
            drop_rate = config.MODEL.COARSE.DROP_RATE, 
            drop_path_rate = config.MODEL.COARSE.DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f"Unkown model: {config.MODEL.COARSE.TYPE}")

    return model



def build_fine_model(config, is_VAL = False):
    if config.MODEL.FINE.TYPE == 'phtrans':
        model = PHTrans(
            img_size = config.DATASET.FINE.SIZE, 
            base_num_features = config.MODEL.FINE.BASE_NUM_FEATURES, 
            num_classes = config.DATASET.FINE.LABEL_CLASSES, 
            num_only_conv_stage = config.MODEL.FINE.NUM_ONLY_CONV_STAGE, 
            num_conv_per_stage = config.MODEL.FINE.NUM_CONV_PER_STAGE,
            feat_map_mul_on_downscale = config.MODEL.FINE.FEAT_MAP_MUL_ON_DOWNSCALE,  
            pool_op_kernel_sizes = config.MODEL.FINE.POOL_OP_KERNEL_SIZES,
            conv_kernel_sizes = config.MODEL.FINE.CONV_KERNEL_SIZES, 
            dropout_p = config.MODEL.FINE.DROPOUT_P,
            deep_supervision = config.MODEL.DEEP_SUPERVISION if not is_VAL else False,
            max_num_features = config.MODEL.FINE.MAX_NUM_FEATURES, 
            depths = config.MODEL.FINE.DEPTHS, 
            num_heads = config.MODEL.FINE.NUM_HEADS,
            window_size = config.MODEL.FINE.WINDOW_SIZE, 
            mlp_ratio = config.MODEL.FINE.MLP_RATIO, 
            qkv_bias = config.MODEL.FINE.DROP_RATE, 
            qk_scale = config.MODEL.FINE.QK_SCALE,
            drop_rate = config.MODEL.FINE.DROP_RATE, 
            drop_path_rate = config.MODEL.FINE.DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f"Unkown model: {config.MODEL.FINE.TYPE}")

    return model



