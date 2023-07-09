import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Initial page config
st.set_page_config(
     page_title='SightAi',
     page_icon=":eyes:",
     layout="wide",
)

# Sidebar Menu
def side_menu():
    with st.sidebar:
        selected = option_menu(
            menu_title="Détection d'objet",
            options=["Image/Vidéo", "Webcam"],
            menu_icon="eye",
            default_index=0,
        )
    return selected

def detect_objects(image):
    # Chargez le modèle YOLOv5
    model = YOLO('./models/best.pt')

    # Effectuez la détection d'objets sur l'image
    results = model(image, save=True, project="predicted")

        # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Class probabilities for classification outputs

    st.write(results)


    return results

def view_form():
    st.title(f"VIEW FORM")

    # Affichez un formulaire permettant aux utilisateurs de télécharger une image ou une vidéo
    file_type = st.selectbox("Sélectionnez le type de fichier", ["Image", "Vidéo"])
    if file_type == "Image":
        uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Lisez l'image téléchargée
            image = Image.open(uploaded_file)
            # Affichez l'image
            st.image(image, caption="Image téléchargée", use_column_width=True)

            # Vérifiez si le bouton "Détecter" est cliqué
            if st.button("Détecter"):
                # # Convertir l'image en tableau numpy
                # image_array = np.array(image)

                # Effectuez la détection d'objets sur l'image
                predictions = detect_objects(image)

    #             # Dessinez les bounding boxes et les annotations sur l'image
    #             for _, row in predictions.iterrows():
    #                 x1, y1, x2, y2, class_name, confidence = row
    #                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    #                 # Dessinez la bounding box
    #                 cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #                 # Ajoutez l'étiquette de classe et le score à la bounding box
    #                 label = f"{class_name}: {confidence:.2f}"
    #                 cv2.putText(image_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # # Affichez l'image avec les bounding boxes et les annotations
                # st.image(image_array, caption="Image avec détection d'objets", use_column_width=True)
    else:
        st.warning("La détection d'objets sur les vidéos n'est pas prise en charge pour le moment.")

def view_cam():
    st.title(f"VIEWCAM")
    st.warning("La détection d'objets en temps réel à partir de la webcam n'est pas prise en charge pour le moment.")

if __name__ == "__main__":
    selected = side_menu()
    if selected == "Image/Vidéo":
        view_form()
    if selected == "Webcam":
        view_cam()
