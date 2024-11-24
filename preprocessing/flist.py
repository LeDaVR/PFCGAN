import os
import glob
from tqdm import tqdm

import argparse
import os

parser = argparse.ArgumentParser(description="Selecciona el modo de operación.")
parser.add_argument('--folder', type=str, help="Ruta a la carpeta para continuar.")
parser.add_argument('--out', type=str, help="Nombre del archivo de salida")
args = parser.parse_args()

f_out = open(args.out, "w")

for f in tqdm(glob.glob(os.path.join(args.folder, "*.jpg")), desc="Procesando imágenes", unit="imagen"):
    path = os.path.abspath(f)
    f_out.write(path + '\n')

f_out.close()
