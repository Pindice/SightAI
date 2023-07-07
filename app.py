import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import torch
import onnxruntime as ort


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


def preprocess_image(image):
    # Effectuez les prétraitements spécifiques requis par le modèle ONNX
    resized_image = cv2.resize(image, (224, 224))  # Redimensionnez l'image à la taille d'entrée du modèle
    resized_image = resized_image.astype(np.float32) / 255.0  # Normalisez les valeurs de pixel entre 0 et 1
    preprocessed_image = np.transpose(resized_image, (2, 0, 1))  # Réorganisez les dimensions de l'image

    # Ajoutez une dimension de lot (batch dimension)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    return preprocessed_image

# Fonction de détection d'objets
def detect_objects(image):
    # Chargez le modèle ONNX
    session = ort.InferenceSession('models/best.onnx')
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # Affichez les noms des sorties du modèle (pour le débogage)
    st.write(output_names)

    # Prétraitez l'image
    preprocessed_image = preprocess_image(image)

    # Effectuez les prédictions sur l'image
    outputs = session.run(output_names, {input_name: preprocessed_image})

    st.write(outputs)

    # Vérifiez la structure de la sortie du modèle
    if outputs is not None and len(outputs) > 0:
        output_shape = outputs[0].shape
        st.write(f"Output shape: {output_shape}")
        if len(output_shape) > 2:
            num_boxes = output_shape[1]
            num_attributes = output_shape[2]
            st.write(f"Number of boxes: {num_boxes}")
            st.write(f"Number of attributes per box: {num_attributes}")

            # Obtenez les résultats de détection (classes) à partir des sorties du modèle
            predicted_classes = np.argmax(outputs[0], axis=2)
            st.write(predicted_classes)

            # Obtenez les bounding boxes à partir des sorties du modèle
            bounding_boxes = outputs[0][:, :, :4]  # Assurez-vous que les coordonnées des bounding boxes sont dans les 4 premières colonnes
            st.write(bounding_boxes)

            # Obtenez les scores prédits à partir des sorties du modèle
            predicted_scores = outputs[0][:, :, 4]  # Assurez-vous que les scores prédits sont dans la 5e colonne
            st.write(predicted_scores)

            # Retournez les résultats de détection
            return predicted_classes, predicted_scores, bounding_boxes[:, :, 0]

    # Si aucune sortie valide n'est trouvée, retournez des résultats vides
    return None, None, None



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
            
            # Vérifiez si le bouton "Détecter" est cliqué
            if st.button("Détecter"):
                # Effectuez la détection d'objets sur l'image
                predicted_classes, predicted_scores, bounding_boxes = detect_objects(image)
                
                # Dessinez les bounding boxes et les annotations sur l'image
                for i in range(bounding_boxes.shape[0]):
                    x, y, w, h = bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][2], bounding_boxes[i][3]

                    class_name = predicted_classes[i]
                    score = predicted_scores[i]
                    
                    # Dessinez la bounding box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Ajoutez l'étiquette de classe et le score à la bounding box
                    label = f"{class_name}: {score:.2f}"
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                # Affichez l'image avec les bounding boxes et les annotations
                st.image(image, channels="BGR", caption="Image avec détection d'objets", use_column_width=True)
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
