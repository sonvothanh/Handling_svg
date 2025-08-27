# Convert svg to png 
# Already have 6 files svg:
#       /content/Dice_1.svg
#       /content/Dice_2.svg
#       /content/Dice_3.svg
#       /content/Dice_4.svg
#       /content/Dice_5.svg
#       /content/Dice_6.svg

# Need to do: 
# !pip install cairosvg

import cairosvg
import os

svg_files = [f"/content/Dice_{i}.svg" for i in range(1, 7)]
output_dir = "/content/"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for svg_file in svg_files:
    if os.path.exists(svg_file):
        png_file = os.path.join(output_dir, os.path.splitext(os.path.basename(svg_file))[0] + ".png")
        print(f"Converting {svg_file} to {png_file}...")
        cairosvg.svg2png(url=svg_file, write_to=png_file)
        print(f"Saved {png_file}")
    else:
        print(f"SVG file not found: {svg_file}")
