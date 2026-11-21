"""Start coding here; connect this function to the official starter-kit loader."""

def predict_masks(images, methods, messages):
    """Return binary torch.uint8 masks [N,128,128] in input order.

    images: RGB images; normalize according to the selected official decoder.
    methods: watermark method per image (PixelSeal, TrustMark, StegaStamp).
    messages: known embedded bit strings per image.
    """
    raise NotImplementedError("Implement watermark localization here")
