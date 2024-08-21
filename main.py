# Librerias
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math

#Funcion Limpiar Imagen
def clean_lbl():
    # Limpia El Espacio De Imagen
    lblimg.config(image='')

#Funcion Definir Imagen
def images(img):
    img = img

    # Deteccion De Imagen
    img = np.array(img, dtype="uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)

    img_ = ImageTk.PhotoImage(image=img)
    lblimg.configure(image=img_)
    lblimg.image = img_

# Funcion Escaneo
def Scanning():
    global img_metal, img_glass, img_plastic, img_carton, img_medical, img_organic, img_wood, img_electronic
    global lblimg, pantalla

    # Interfaz
    lblimg = Label(pantalla)
    lblimg.place(x=30, y=300)

    # Interpretar Captura De Video
    if cap is not None:
        ret, frame = cap.read()
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # True
        if ret == True:
            # Yolo | AntiSpoof
            results1 = model(frame, stream=True, verbose=False)
            results2 = model2(frame, stream=True, verbose=False)
            detect = False  # Iniciar Deteccion
            for res in results1:
                # Contorno
                boxes = res.boxes
                for box in boxes:
                    detect = True
                    # Delimitando Contorno
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Error < 0
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 < 0: x2 = 0
                    if y2 < 0: y2 = 0

                    # Clase
                    cls = int(box.cls[0])

                    # Aprendizaje
                    conf = math.ceil(box.conf[0])
                    print(f"Clase: {cls} Confidence: {conf}")

                    # Metal
                    if cls == 0:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0),cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        # Clasificacion
                        images(img_metal)

                    # Vidrio
                    if cls == 1:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # Clasificacion
                        images(img_glass)

                    #Plastico
                    if cls == 2:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        # Clasificacion
                        images(img_plastic)

                    # Carton
                    if cls == 3:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (181, 178, 178), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # Clasificacion
                        images(img_carton)

                    # Medico
                    if cls == 4:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        # Clasificacion
                        images(img_medical)

            '''for res in results2:
                # Contorno
                boxes = res.boxes
                for box in boxes:
                    detect = True
                    # Delimitando Contorno
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Error < 0
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 < 0: x2 = 0
                    if y2 < 0: y2 = 0

                    # Clase
                    cls = int(box.cls[0])

                    # Aprendizaje
                    conf = math.ceil(box.conf[0])
                    print(f"Clase: {cls} Confidence: {conf}")

                    # Organicos
                    if cls == 0:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (111, 78, 55), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName2[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0),cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (111, 78, 55), 2)
                        # Clasificacion
                        images(img_organic)

                    # Metal
                    if cls == 1:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName2[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        # Clasificacion
                        images(img_metal)

                    # Madera
                    if cls == 2:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 165, 0), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName2[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                        # Clasificacion
                        images(img_wood)

                    # Electronicos
                    if cls == 3:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (128, 128, 128), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName2[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                        # Clasificacion
                        images(img_electronic)

                    # Carton
                    if cls == 4:
                        # Generar Delimitador
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # Colocar Informacion Al Delimitador
                        text = f'{clsName2[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        # Fondo Del Texto
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (181, 178, 178), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # Clasificacion
                        images(img_carton)'''<

            if detect == False:
                # Clean
                clean_lbl()

            # Redimiensionar Frame
            frame_show = imutils.resize(frame_show, width=640)

            # Convertimos el video
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)

            # Mostramos en el GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Scanning)

        else:
            cap.release()

# main
def ventana_principal():
    global cap, lblVideo, model, clsName, model2, clsName2, pantalla
    global img_metal, img_glass, img_plastic, img_carton, img_medical, img_organic, img_wood, img_electronic
    # Ventana principal
    pantalla = Tk()
    pantalla.title("RECICLAJE INTELIGENTE")
    pantalla.geometry("1280x720")

    # Fondo
    imagenF = PhotoImage(file="setUp/Canva.png")
    background = Label(image=imagenF, text="Inicio")
    background.place(x=0, y=0, relwidth=1, relheight=1)

    # Clases Modelo 1: 0 -> Metal | 1 -> Glass | 2 -> Plastic | 3 -> Carton | 4 -> Medical
    # Clases Modelo 2: 0 -> Organicos | 1 -> Metal | 2 -> Madera | 3 -> Electronicos | 4 -> Carton
    # Model
    model = YOLO('Modelos/best.pt')
    model2 = YOLO('Modelos/best2.pt')  # Segundo modelo

    # Clases
    clsName = ['Metal', 'Vidrio', 'Plastico', 'Papel/Carton', 'Sanitario']
    clsName2 = ['Organicos', 'Metal', 'Madera', 'Electronicos', 'Papel/Carton']

    # Images
    img_metal = cv2.imread("setUp/metal.png")
    img_glass = cv2.imread("setUp/vidrio.png")
    img_plastic = cv2.imread("setUp/plastico.png")
    img_carton = cv2.imread("setUp/carton.png")
    img_medical = cv2.imread("setUp/medicos.png")
    img_organic = cv2.imread("setUp/organicos.png")
    img_wood = cv2.imread("setUp/madera.png")
    img_electronic = cv2.imread("setUp/electronicos.png")

    # Video
    lblVideo = Label(pantalla)
    lblVideo.place(x=320, y=180)

    # Elegimos la camara

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(3, 1280)
    cap.set(4, 720)
    Scanning()

    # Eject
    pantalla.mainloop()

if __name__ == "__main__":
    ventana_principal()
