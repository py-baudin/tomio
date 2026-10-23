"""the Python examples of the README section "Registration" run (on small synthetic volumes)"""
import re
from pathlib import Path

import numpy as np
import pytest

import tomio as io

README = Path(__file__).parents[1] / "README.md"


def section_blocks(title, end):
    text = README.read_text(encoding="utf-8")
    section = text[text.index(f"## {title}\n"): text.index(f"## {end}\n")]
    return re.findall(r"```python\n(.*?)```", section, flags=re.S)


@pytest.fixture
def images():
    """fixed and moving volumes (same smooth blobs, moving shifted and of another contrast), ROI on some slices"""
    shape, spacing = (32, 32, 12), (1.5, 1.5, 3.0)
    rng = np.random.default_rng(0)
    centers = rng.uniform([8, 8, 6], [40, 40, 30], size=(10, 3))
    points = np.indices(shape).reshape(3, -1).T * spacing

    def blobs(p):
        return np.exp(-((p[:, None] - centers) ** 2).sum(axis=-1) / 40).sum(axis=1).reshape(shape)

    fixed = io.Volume(blobs(points).astype(np.float32), spacing=spacing)
    moving = io.Volume(np.exp(-blobs(points - [1.5, -1.0, 0.0])).astype(np.float32), spacing=spacing)
    labelmap = io.Volume((np.asarray(moving) < 0.5).astype(np.uint8), spacing=spacing)
    roi = labelmap.copy()
    roi[..., 1::3] = 0  # labelled on some slices
    roi[..., 2::3] = 0
    return dict(fixed=fixed, moving=moving, roi=roi, labelmap=labelmap)


def test_readme_registration(images, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # files written by the examples
    blocks = section_blocks("Registration", "Config files")
    assert len(blocks) >= 2
    namespace = dict(images)
    for block in blocks:
        exec(compile(block, "README.md", "exec"), namespace)
    assert namespace["check"].ok
    assert namespace["roi_full"].shape == images["roi"].shape
