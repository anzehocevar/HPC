from PIL import Image

# Open the PGM file
img = Image.open("output.pgm")

# Save it as PNG
img.save("gray_scott_V.png")
