from .shapenet_table2_model import ShapeNet55Table2Model


class MBB_Model_ShapeNet(ShapeNet55Table2Model):
    """
    Compatibility constructor used only for old-vs-release audit.

    It exposes the historical constructor:

        MBB_Model_ShapeNet(
            cfg,
            use_mbb=True,
            mbb_mode="ours"
        )

    while executing the new static release implementation.
    """

    def __init__(
        self,
        cfg,
        use_mbb=True,
        mbb_mode="ours",
    ):
        if not use_mbb:
            mode = "baseline"
        else:
            mode = mbb_mode

        super().__init__(
            cfg,
            mode=mode,
        )
