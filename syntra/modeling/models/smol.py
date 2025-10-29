
from . import SemSegModelBase


class SMOLModel(SemSegModelBase):
    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
    ):
        super().__init__(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
        )