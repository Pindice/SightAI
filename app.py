import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np

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

# Fonction de détection d'objets
def detect_objects(image):
    # Utilisez le modèle YOLO-v8 pour détecter les objets dans l'image
    # Effectuez les prédictions et obtenez les bounding boxes et les classes des objets détectés
    
    # Retournez les résultats de détection (bounding boxes, classes, etc.)
    return None


def view_form():
    st.title(f"VIEW FORM")

    # Affichez un formulaire permettant aux utilisateurs de télécharger une image ou une vidéo
    file_type = st.selectbox("Sélectionnez le type de fichier", ["Image", "Vidéo"])
    if file_type == "Image":
        uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Lisez l'image téléchargée
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            # Affichez l'image
            st.image(image, channels="BGR", caption="Image téléchargée", use_column_width=True)
    else:
        uploaded_file = st.file_uploader("Télécharger une vidéo", type=["mp4"])
        if uploaded_file is not None:
            # Lisez la vidéo téléchargée
            video = cv2.VideoCapture(uploaded_file)
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                # Affichez chaque frame de la vidéo
                st.image(frame, channels="BGR", caption="Frame vidéo", use_column_width=True)

def view_cam():
    st.title(f"VIEW CAM")
    # Capturer la vidéo en direct à partir de la webcam

if __name__ == "__main__":
    selected = side_menu()
    if selected == "Image/Vidéo":
        view_form()
    if selected == "Webcam":
        view_cam()
