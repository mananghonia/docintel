"""DocIntel ML package: invoice field extraction with active learning.

Data flow:
    document image/PDF
      -> ml.ocr          : preprocess + OCR -> list[Token] (text, bbox, page)
      -> ml.labeling     : ground-truth bboxes -> BIO tag per token
      -> ml.features     : Token sequence -> feature matrix
      -> ml.baseline / models_classical / models_bilstm : tags + probabilities
      -> ml.calibration  : honest confidences
      -> ml.postprocess  : field values + business-rule consistency
      -> ml.active_learning : which docs to route to human review
"""

__version__ = "0.1.0"
