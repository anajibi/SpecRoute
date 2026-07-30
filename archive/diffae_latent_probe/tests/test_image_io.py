from pathlib import Path

from PIL import Image

from diffae_tools.image_io import load_image_tensor


def test_load_image_tensor_returns_diffae_shape(tmp_path: Path):
    img = Image.new("RGB", (320, 240), color=(128, 64, 32))
    path = tmp_path / "sample.png"
    img.save(path)

    tensor = load_image_tensor(path, image_size=256, normalize=True)
    assert tensor.shape == (3, 256, 256)
    assert tensor.min() >= -1.0 - 1e-5
    assert tensor.max() <= 1.0 + 1e-5

