# Official localization research

Checked 2026-11-21 against first-party repositories. These localizers identify their own invisible watermark families; neither is documented as a general visible-logo, text-watermark, or arbitrary-watermark detector (inference from their documented purposes).

## TrustMark (Adobe)

- Bounding-box localization arrived in **0.9.0**. Hosting changed in **0.9.1**; older installations may retain obsolete Netlify URLs. [Official changelog](https://github.com/adobe/trustmark/blob/main/python/CHANGELOG.md).
- Use `TrustMark(model_type='Q', loadBBoxDetector=True, loadRemover=False)` and `localize(rgb_pil, return_all=True)`. Default construction otherwise loads encoder/decoder as well. Q and P have published bbox entries; B/C do not. [Official implementation](https://github.com/adobe/trustmark/blob/main/python/trustmark/trustmark.py).
- Exact Q detector downloads: [trustmark_bbox_Q.yaml](https://cai-watermark.adobe.net/watermarking/trustmark-models/trustmark_bbox_Q.yaml) and [trustmark_bbox_Q.ckpt](https://cai-watermark.adobe.net/watermarking/trustmark-models/trustmark_bbox_Q.ckpt). Official MD5: YAML `749b0d62106f8f6648e6f781c3143105`; checkpoint `9d15428a33e15140ea16aa378416d304`. These URLs are constructed by the official loader; this research did not download or verify the large checkpoint. [URL/checksum declarations](https://github.com/adobe/trustmark/blob/main/python/trustmark/trustmark.py).
- Localizer preprocessing: RGB PIL image; preserve aspect ratio, bilinear downscale only when longest edge exceeds 1600. Pass a list containing a float CHW tensor normalized to **[-1,1]**. Output is normalized **[x1,y1,x2,y2]**, or `None`; default returns highest-score box. No segmentation foreground class index applies. [localize implementation](https://github.com/adobe/trustmark/blob/main/python/trustmark/trustmark.py#L319).
- Decode crops with Q's matching decoder: resize to **245×245**, batch RGB BCHW, **[-1,1]**. `decode(..., DETECTFIRST=True)` crops detector boxes before decoding. [Decoder implementation](https://github.com/adobe/trustmark/blob/main/python/trustmark/trustmark.py#L366).
- Q is the original balanced robustness/quality variant; watermark variants are **not mutually decodable**. Use Q detector/decoder for Q-encoded images; do not silently substitute P. TrustMark works on RGB raster content, including synthetic images, and its published models were trained on licensed Adobe Stock. [Official FAQ](https://github.com/adobe/trustmark/blob/main/FAQ.md).

## StegaStamp (Tancik et al.)

- Historical official downloads: [detector_models.tar.xz](http://people.eecs.berkeley.edu/~tancik/stegastamp/detector_models.tar.xz), expected extracted SavedModel path `detector_models/stegastamp_detector`; [saved_models.tar.xz](http://people.eecs.berkeley.edu/~tancik/stegastamp/saved_models.tar.xz), decoder path `saved_models/stegastamp_pretrained`. The current README removed the download instructions; these exact links remain in the [official historical README](https://github.com/tancik/StegaStamp/blob/8e4eeda30a4210c7e0bcf5b47c35a9bc734e9bfa/README.md). Availability is not established by that historical reference.
- Load TensorFlow SavedModel serving signature input **image**, output **detections**. Convert OpenCV BGR to **RGB**, resize directly to **1024×1024**, float32 NHWC **[1,1024,1024,3]**, divide by 255: **[0,1]**. Argmax output over final axis. **Class 0 is foreground**, class 1 background: the official code maps class 0 to white, class 1 to black, then finds nonzero-mask contours. [Official detector.py](https://github.com/tancik/StegaStamp/blob/master/detector.py#L49).
- The official postprocessor fits quadrilaterals, rectifies each candidate to **400×400** RGB **[0,1]**, and tries four rotations with the matching decoder. [Official detector.py](https://github.com/tancik/StegaStamp/blob/master/detector.py#L94).
- Intended domain: one or more StegaStamp photos inside larger frames, including print-and-camera capture. The BiSeNet segmenter was trained using warped StegaStamps composited onto larger images. Authors recommend 1920×1080 input video and warn that segmentation is noisy and sensitive to stamp size. Training code for this detector is not supplied. [Official README](https://github.com/tancik/StegaStamp#detector), [author project page](https://www.matthewtancik.com/stegastamp).

### Download availability and recovery

The original detector archive currently returns 404 in the parent experiment. Author Matthew Tancik confirms that the university stopped hosting the models after his departure and that he no longer has the files; this explains removal of the README downloads. [Author response, official issue 59](https://github.com/tancik/StegaStamp/issues/59). No working first-party replacement was found.

A **community mirror**, not an authenticated official release, contains a TensorFlow detector SavedModel: [Gu-f/StegaStamp](https://github.com/Gu-f/StegaStamp/tree/d95ea307c73410c96557200656d2e0a59574eedb/detector_models/stegastamp_detector). Pinned mirror commit: `d95ea307c73410c96557200656d2e0a59574eedb`. Its files are Git LFS pointers in raw.githubusercontent.com; use media.githubusercontent.com for actual bytes. HEAD requests for the graph and data shard returned HTTP 200 on the research date. The following hashes/sizes come from that mirror's LFS pointers and establish mirror integrity only, not equivalence to the lost official archive:

| File | Size | SHA-256 |
|---|---:|---|
| [saved_model.pb](https://media.githubusercontent.com/media/Gu-f/StegaStamp/d95ea307c73410c96557200656d2e0a59574eedb/detector_models/stegastamp_detector/saved_model.pb) | 1,380,738 | `f5526a12f0caf3bedb637ced8bf270f7d833a10de8c0858d64e1248c02c962b0` |
| [variables.data-00000-of-00001](https://media.githubusercontent.com/media/Gu-f/StegaStamp/d95ea307c73410c96557200656d2e0a59574eedb/detector_models/stegastamp_detector/variables/variables.data-00000-of-00001) | 113,817,448 | `a608276fce5487fc59e4a79bb38ccf21936e6993107170c4e74dd2399f435069` |
| [variables.index](https://media.githubusercontent.com/media/Gu-f/StegaStamp/d95ea307c73410c96557200656d2e0a59574eedb/detector_models/stegastamp_detector/variables/variables.index) | 12,563 | `7b60f3b7d14537a81b7b58b00053b1a37284962cbb74dcb1c83bf7159fc00640` |

The community Google Drive linked in [issue 61](https://github.com/tancik/StegaStamp/issues/61) has reports of missing variable shards; it is a weaker recovery option. This research did not download or execute any recovered model.
