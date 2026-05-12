from PIL import Image
from pathlib import Path

ico_path = Path('data/fav_icon.ico')
img = Image.open(ico_path)
print(img.format, img.size, img.info)

# list all embedded sizes if ICO
if hasattr(img, 'n_frames'):
    sizes = []
    for i in range(img.n_frames):
        img.seek(i)
        sizes.append(img.size)
    print('frames', sizes)
