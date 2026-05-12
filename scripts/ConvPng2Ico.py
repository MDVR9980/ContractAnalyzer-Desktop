from PIL import Image

# Open the image
img = Image.open('fav_icon.png')

# Save as .ico
# We save with multiple sizes to ensure it looks sharp in both Taskbar and Window
img.save('favicon.ico', format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)])
