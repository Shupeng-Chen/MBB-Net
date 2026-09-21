from .ShapeNetCars_dataset import ShapeNetCarsDataset


def build_dataset_from_cfg(cfg, default_args):
    """
    Minimal dataset factory required by KITTI FD/MMD evaluation.
    """
    cfg.update(default_args)

    if cfg.NAME == "ShapeNetCars":
        return ShapeNetCarsDataset(cfg)

    raise NotImplementedError(
        f"Dataset {cfg.NAME!r} is not supported by "
        "the portable KITTI evaluator."
    )
