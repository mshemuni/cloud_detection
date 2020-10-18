from dag_cld import ast


file = "data/asc.fits"

img = ast.Image(file)
print(img.header())
