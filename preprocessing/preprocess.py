import cv2
import numpy as np
import glob
from tqdm import tqdm

import argparse
import os

import dlib

def crop(img):
    height, _ = img.shape[:2]
    cropped_img = img[20:height-20, :]
    return cropped_img

# Crear el parser
parser = argparse.ArgumentParser(description="Selecciona el modo de operación.")

# Añadir el argumento `--mode`
parser.add_argument('--mode', choices=['clean', 'continue'], required=True, 
                    help="Define el modo de operación: 'clean' para limpiar o 'continue' para continuar.")

# Añadir los argumentos `--folder` y `--model`, pero solo serán necesarios si el modo es 'continue'
parser.add_argument('--folder', type=str, help="Ruta a la carpeta para continuar.")
parser.add_argument('--model', type=str, help="Ruta al modelo para continuar.")

# Argumento para el archivo de progreso
parser.add_argument('--progress_file', type=str, help="Archivo que contiene el nombre de la última imagen procesada (solo para --mode continue).")

# Parsear los argumentos
args = parser.parse_args()

print("starting")
# Lógica para el modo 'clean'
if args.mode == 'clean':
    print("Modo limpio seleccionado. Limpiando...")
    # Aquí va el código para el modo 'clean'


# Lógica para el modo 'continue', donde los argumentos `--folder` y `--model` son requeridos
elif args.mode == 'continue':
    if not args.folder or not args.model:
        print("Error: Si el modo es 'continue', los argumentos --folder y --model son obligatorios.")
    elif not args.progress_file:
        print("Error: Si el modo es 'continue', debes proporcionar el archivo de progreso (--progress_file).")
    else:
        # Verificar si el archivo de progreso existe
        if os.path.exists(args.progress_file):
            # Leer el archivo de progreso para obtener la última imagen procesada
            with open(args.progress_file, 'r') as save:
                last_image = save.read().strip()
            print(f"Continuando desde la imagen: {last_image}")
        else:
            # Si no existe el archivo, se empieza desde 0
            print("No se encontró el archivo de progreso. Empezando desde el principio...")
            last_image = None  # Aquí puedes definir el comportamiento para iniciar desde la primera imagen


        not_detected_file = 'not_detected.txt'

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(args.model)

        f_out = open(not_detected_file, "a")

        # Look for last image before processing

        last_image_found = last_image == None

        for f in tqdm(glob.glob(os.path.join(args.folder, "*.jpg")), desc="Procesando imágenes", unit="imagen"):
            # print("Processing file: {}".format(f))
            if not last_image_found:
                last_image_found = f == last_image
                continue


            img = dlib.load_rgb_image(f)
            img = crop(img)
            img = cv2.resize(img, (128, 128))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            dets = detector(img, 2)
            # print("Number of faces detected: {}".format(len(dets)))

            if len(dets) == 0:
                f_out.write(f"{f}\n")

            for k, d in enumerate(dets):
                shape = predictor(img, d)
                landmarks = np.array([(p.x, p.y) for p in shape.parts()])

                landmark_img = np.zeros(img.shape, dtype = np.uint8)
                for (x, y) in landmarks:
                    cv2.circle(landmark_img, (x, y), 1, (255, 255, 255), -1)

                hull = cv2.convexHull(landmarks)
                mask = np.zeros_like(img_gray)
                cv2.fillConvexPoly(mask, hull, 255)

                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

                face_part_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)


                name = f.split('\\')[-1].split(".")[0]
                work = cv2.imwrite(f"out/original/{name}.jpg", img_rgb)
                work = cv2.imwrite(f"out/processed/{name}_landmarks.jpg", landmark_img)
                cv2.imwrite(f"out/processed/{name}_mask.jpg", mask)
                cv2.imwrite(f"out/processed/{name}_face_part.jpg", face_part_img)

            with open(args.progress_file, 'w') as save:
                save.write(f)
                # print(f"Generated images for file: {f}")

        print("Processing complete.")

        f_out.close()
