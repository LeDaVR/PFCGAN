import sys
import concurrent.futures
import os
import dlib
import cv2
import numpy as np
import glob
from tqdm import tqdm

import argparse
import os
dlib.DLIB_USE_CUDA = True 

def crop(img):
    height, _ = img.shape[:2]
    cropped_img = img[20:height-20, :]
    return cropped_img


# Verificar y crear los directorios de salida si no existen
def create_output_dirs(output_folder):
    original_dir = os.path.join(output_folder, "original")
    processed_dir = os.path.join(output_folder, "processed")

    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

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

# Argumento para especificar la carpeta de salida
parser.add_argument('--output_folder', type=str, default="out", help="Carpeta donde se guardarán los resultados procesados (predeterminado: 'out').")


# Parsear los argumentos
args = parser.parse_args()

# Lógica para el modo 'clean'
# if args.mode == 'clean':
#     print("Modo limpio seleccionado. Limpiando...")
#     # Aquí va el código para el modo 'clean'

# # Lógica para el modo 'continue', donde los argumentos `--folder` y `--model` son requeridos
# elif args.mode == 'continue':
#     if not args.folder or not args.model:
#         print("Error: Si el modo es 'continue', los argumentos --folder y --model son obligatorios.")
#     elif not args.progress_file:
#         print("Error: Si el modo es 'continue', debes proporcionar el archivo de progreso (--progress_file).")
#     else:
#         # Verificar si el archivo de progreso existe
#         if os.path.exists(args.progress_file):
#             # Leer el archivo de progreso para obtener la última imagen procesada
#             with open(args.progress_file, 'r') as save:
#                 last_image = save.read().strip()
#             print(f"Continuando desde la imagen: {last_image}")
#         else:
#             # Si no existe el archivo, se empieza desde 0
#             print("No se encontró el archivo de progreso. Empezando desde el principio...")
#             last_image = None  # Aquí puedes definir el comportamiento para iniciar desde la primera imagen


#         not_detected_file = 'not_detected.txt'

#         detector = dlib.get_frontal_face_detector()
#         predictor = dlib.shape_predictor(args.model)

#         f_out = open(not_detected_file, "a")
#         create_output_dirs(args.output_folder)

#         # Look for last image before processing

#         last_image_found = last_image == None

#         for f in tqdm(glob.glob(os.path.join(args.folder, "*.jpg")), desc="Procesando imágenes", unit="imagen"):
#             # print("Processing file: {}".format(f))
#             if not last_image_found:
#                 last_image_found = f == last_image
#                 continue


#             img = dlib.load_rgb_image(f)
#             img = crop(img)
#             img = cv2.resize(img, (128, 128))
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#             # Detect faces in the image
#             dets = detector(img, 2)
#             # print("Number of faces detected: {}".format(len(dets)))

#             if len(dets) == 0:
#                 f_out.write(f"{f}\n")

#             for k, d in enumerate(dets):
#                 shape = predictor(img, d)
#                 landmarks = np.array([(p.x, p.y) for p in shape.parts()])

#                 landmark_img = np.zeros(img.shape, dtype = np.uint8)
#                 # for (x, y) in landmarks:
#                 #     cv2.circle(landmark_img, (x, y), 2, (255, 255, 255), -1)
#                 for i in range(0, 16):
#                     pt1 = (shape.part(i).x, shape.part(i).y)
#                     pt2 = (shape.part(i+1).x, shape.part(i+1).y)
#                     cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
                
#                 # Right eyebrow
#                 for i in range(17, 21):
#                     pt1 = (shape.part(i).x, shape.part(i).y)
#                     pt2 = (shape.part(i+1).x, shape.part(i+1).y)
#                     cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
                
#                 # Left eyebrow
#                 for i in range(22, 26):
#                     pt1 = (shape.part(i).x, shape.part(i).y)
#                     pt2 = (shape.part(i+1).x, shape.part(i+1).y)
#                     cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
                
#                 # Nose bridge
#                 for i in range(27, 30):
#                     pt1 = (shape.part(i).x, shape.part(i).y)
#                     pt2 = (shape.part(i+1).x, shape.part(i+1).y)
#                     cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
                
#                 # Nose bottom
#                 for i in range(30, 35):
#                     pt1 = (shape.part(i).x, shape.part(i).y)
#                     pt2 = (shape.part(i+1).x, shape.part(i+1).y)
#                     cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
                
#                 # Right eye
#                 for i in range(36, 41):
#                     pt1 = (shape.part(i).x, shape.part(i).y)
#                     pt2 = (shape.part((i+1) % 42).x, shape.part((i+1) % 42).y)
#                     cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
                
#                 # Left eye
#                 for i in range(42, 47):
#                     pt1 = (shape.part(i).x, shape.part(i).y)
#                     pt2 = (shape.part((i+1) % 48).x, shape.part((i+1) % 48).y)
#                     cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
                
#                 # Outer lip
#                 for i in range(48, 59):
#                     pt1 = (shape.part(i).x, shape.part(i).y)
#                     pt2 = (shape.part((i+1) % 60).x, shape.part((i+1) % 60).y)
#                     cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
                
#                 # Inner lip
#                 for i in range(60, 67):
#                     pt1 = (shape.part(i).x, shape.part(i).y)
#                     pt2 = (shape.part((i+1) % 68).x, shape.part((i+1) % 68).y)
#                     cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)

#                 hull = cv2.convexHull(landmarks)
#                 mask = np.zeros_like(img_gray)
#                 cv2.fillConvexPoly(mask, hull, 255)

#                 kernel = np.ones((5, 5), np.uint8)
#                 mask = cv2.dilate(mask, kernel, iterations=1)

#                 face_part_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)


#                 # Guardar las imágenes procesadas en las carpetas correspondientes
#                 name = f.split('\\')[-1].split(".")[0]
#                 work = cv2.imwrite(os.path.join(args.output_folder, "original", f"{name}.jpg"), img_rgb)
#                 work = cv2.imwrite(os.path.join(args.output_folder, "processed", f"{name}_landmarks.jpg"), landmark_img)
#                 cv2.imwrite(os.path.join(args.output_folder, "processed", f"{name}_mask.jpg"), mask)
#                 cv2.imwrite(os.path.join(args.output_folder, "processed", f"{name}_face_part.jpg"), face_part_img)


#             with open(args.progress_file, 'w') as save:
#                 save.write(f)
#                 # print(f"Generated images for file: {f}")

#         print("Processing complete.")

#         f_out.close()


def process_image(f, predictor, detector, output_folder):
    img = dlib.load_rgb_image(f)
    img = crop(img)
    img = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar caras en la imagen
    dets = detector(img, 2)

    if len(dets) == 0:
        return f, None

    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmarks = np.array([(p.x, p.y) for p in shape.parts()])

        landmark_img = np.zeros(img.shape, dtype=np.uint8)
        # Dibuja los landmarks
        for i in range(0, 16):
            pt1 = (shape.part(i).x, shape.part(i).y)
            pt2 = (shape.part(i+1).x, shape.part(i+1).y)
            cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
        
        # Right eyebrow
        for i in range(17, 21):
            pt1 = (shape.part(i).x, shape.part(i).y)
            pt2 = (shape.part(i+1).x, shape.part(i+1).y)
            cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
        
        # Left eyebrow
        for i in range(22, 26):
            pt1 = (shape.part(i).x, shape.part(i).y)
            pt2 = (shape.part(i+1).x, shape.part(i+1).y)
            cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
        
        # Nose bridge
        for i in range(27, 30):
            pt1 = (shape.part(i).x, shape.part(i).y)
            pt2 = (shape.part(i+1).x, shape.part(i+1).y)
            cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
        
        # Nose bottom
        for i in range(30, 35):
            pt1 = (shape.part(i).x, shape.part(i).y)
            pt2 = (shape.part(i+1).x, shape.part(i+1).y)
            cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
        
        # Right eye
        for i in range(36, 41):
            pt1 = (shape.part(i).x, shape.part(i).y)
            pt2 = (shape.part((i+1) % 42).x, shape.part((i+1) % 42).y)
            cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
        
        # Left eye
        for i in range(42, 47):
            pt1 = (shape.part(i).x, shape.part(i).y)
            pt2 = (shape.part((i+1) % 48).x, shape.part((i+1) % 48).y)
            cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
        
        # Outer lip
        for i in range(48, 59):
            pt1 = (shape.part(i).x, shape.part(i).y)
            pt2 = (shape.part((i+1) % 60).x, shape.part((i+1) % 60).y)
            cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)
        
        # Inner lip
        for i in range(60, 67):
            pt1 = (shape.part(i).x, shape.part(i).y)
            pt2 = (shape.part((i+1) % 68).x, shape.part((i+1) % 68).y)
            cv2.line(landmark_img, pt1, pt2, (255, 255, 255), 1)

        # Máscara y extracción de la parte de la cara
        hull = cv2.convexHull(landmarks)
        mask = np.zeros_like(img_gray)
        cv2.fillConvexPoly(mask, hull, 255)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        face_part_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        # Guardar las imágenes procesadas
        name = f.split('\\')[-1].split(".")[0]
        cv2.imwrite(os.path.join(output_folder, "original", f"{name}.jpg"), img_rgb)
        cv2.imwrite(os.path.join(output_folder, "processed", f"{name}_landmarks.jpg"), landmark_img)
        cv2.imwrite(os.path.join(output_folder, "processed", f"{name}_mask.jpg"), mask)
        cv2.imwrite(os.path.join(output_folder, "processed", f"{name}_face_part.jpg"), face_part_img)

    return f, True


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.model)

# Función principal para manejar el procesamiento en paralelo
def main():
    # Inicializar detector y predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model)
    create_output_dirs(args.output_folder)
    
    # Crear lista de imágenes a procesar
    image_files = glob.glob(os.path.join(args.folder, "*.jpg"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Crear barra de progreso
        with tqdm(total=len(image_files), desc="Procesando imágenes", unit="imagen") as pbar:
            # Enviar tareas al executor
            futures = {executor.submit(process_image, f, predictor, detector, args.output_folder): f for f in image_files}
            
            # Manejar resultados conforme se completan las tareas
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                try:
                    _, result = future.result()
                    if not result:
                        with open('not_detected.txt', 'a') as f_out:
                            f_out.write(f"{file}\n")
                except Exception as e:
                    print(f"Error procesando {file}: {e}")
                
                # Actualizar tqdm
                pbar.update(1)

if __name__ == "__main__":
    main()