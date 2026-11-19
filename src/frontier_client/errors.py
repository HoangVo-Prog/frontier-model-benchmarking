"""Errors safe to display without leaking Model Proxy credentials."""

class FrontierError(RuntimeError):
    """Base class for inference client failures."""

class AuthenticationError(FrontierError):
    """Missing or expired Kaggle Model Proxy credentials."""

class ImageInputError(FrontierError, ValueError):
    """Invalid image reference, format, or size."""

class UnsupportedParameterError(FrontierError, ValueError):
    """Option not supported by the SDK/model."""

class ModelRequestError(FrontierError):
    """Model Proxy request failed; inspect safe error category."""
