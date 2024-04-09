import warnings

__all__ = ["duplicates_queries_warning"]


def duplicates_queries_warning() -> None:
    message = """Duplicate queries found. Provide distinct queries."""
    warnings.warn(message=message)
