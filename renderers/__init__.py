from .base import DomainRenderer, Rendered  # noqa: F401


def get_renderer(domain: str) -> DomainRenderer:
    return DomainRenderer(domain)
