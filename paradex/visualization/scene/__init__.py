"""Backend-agnostic scene + animation API.

One call surface, two backends:

* :class:`ViserScene` — interactive browser viewer with a built-in player.
* :class:`Open3DScene` — headless offscreen renderer for figures / mp4.

Backends are imported lazily so importing this package does not require both
``viser`` and ``open3d`` to be installed.

    from paradex.visualization.scene import ViserScene   # or Open3DScene

See ``agent_docs/visualization/usage.md``.
"""

from paradex.visualization.scene.base import Scene, Timeline, normalize_color

__all__ = ["Scene", "Timeline", "normalize_color", "ViserScene", "Open3DScene"]


def __getattr__(name):
    if name == "ViserScene":
        from paradex.visualization.scene.viser_backend import ViserScene
        return ViserScene
    if name == "Open3DScene":
        from paradex.visualization.scene.open3d_backend import Open3DScene
        return Open3DScene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
