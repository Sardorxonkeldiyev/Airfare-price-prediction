import streamlit as st 
from fastai.vision.all import *
import pathlib
import plotly.express as px 
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


st.title('Classifikatsiya model: (Fruit, Fish, Flower, Tree)')


#rasmni joylash:
file = st.file_uploader('Rasm yuklash', type=['png','svg','jpeg','gif'])
if file:
    st.image(file)
    #PIL:
    img = PILImage.create(file)

    #model:
    model  = load_learner('Classifikatsiya_model.pkl')

    #predict:
    pred,pred_id,probs = model.predict(img)
    # Rasmda 4 ta asosiy classdan boshqa classni aniqlash
    # Rasmda 4 ta asosiy classdan boshqa classni aniqlash
    if pred == 'Fish' or pred == 'Flower' or pred == 'Tree' or pred == 'Fruit':
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik foizi: {probs[pred_id]*100:.1f}%")
            # plotly:
        fig = px.bar(x= probs*100,y=model.dls.vocab)
        st.plotly_chart(fig)
    else:
        st.warning("Uzur, bu rasm bizning modelda yo'q")
   
