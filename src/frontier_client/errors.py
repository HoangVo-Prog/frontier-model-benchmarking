"""Safe public exceptions: messages never include raw HTTP credentials."""


class FrontierError(RuntimeError):
    """Base class for inference client failures."""


class AuthenticationError(FrontierError):
    """Missing or unusable Kaggle account / Model Proxy credentials."""


class ImageInputError(FrontierError, ValueError):
    """Invalid image reference, format, or size."""


class UnsupportedParameterError(FrontierError, ValueError):
    """Option not supported by the SDK/model."""


class ModelRequestError(FrontierError):
    """Model Proxy or Kaggle request failed."""


class QuotaExhaustedError(ModelRequestError):
    """Kaggle confirmed this account cannot spend more on Model Proxy."""
