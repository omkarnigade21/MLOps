import streamlit as st
import random
import pandas as pd
from io import StringIO
from PIL import Image
import joblib

df = pd.DataFrame()

train_features = joblib.load("/Users/harish/Desktop/Human Acivity Recognition/Notebooks/model_features/K12_train_features.joblib")

model = joblib.load("/Users/harish/Desktop/Human Acivity Recognition/Notebooks/model_registry/K12-tuned-random_forest.joblib")

data = pd.read_csv('/Users/harish/Desktop/Human Acivity Recognition/Data/new_data.csv')

le = joblib.load("/Users/harish/Desktop/Human Acivity Recognition/Notebooks/model_features/encoder_weights.joblib")

def model_real_time_predict(df,model,le):
    #print("shape = ", df.shape)
    #print(df)
    prediction = model.predict(df)
    prediction = le.inverse_transform(prediction)
    return prediction
# streamlit run .app/app.y

def model_batch_predict(df,train_features,model,le):
    df.dropna(inplace = True)
    new_data_features = df[train_features]
    prediction = model.predict(new_data_features)
    prediction = le.inverse_transform(prediction)
    df['Prediction_label'] = prediction
    return df

tab1, tab2 = st.tabs(["Batch Prediction", "Real Time Prediction"])
df_transformed = data[train_features]

with tab1:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # To ensure the uniformity of the file we will carry out the below process
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        #st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        #st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        #st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        df = dataframe
        st.write("Data uploaded successfully")
        #st.write(dataframe)

    if st.button('Batch Predict'):
        st.write('Model Prediction')
        prediction = model_batch_predict(df,train_features,model,le)
        # print(prediction.head()['Prediction_label'])
        pred_df = pd.DataFrame(prediction['Prediction_label'].value_counts())
        pred_df.columns = ['minutes']
        st.bar_chart(pred_df)

        st.write(pred_df)

with tab2:
    st.header("Real Time Prediction")
    st.markdown("### Generate Sensor Value")
    global val
    val = [0,0,0,0,0,0,0,0,0,0,0,0]
    col1, col2 = st.columns(2)

    with col1:
        # Adding the image from the folder
        image = Image.open("/Users/harish/Desktop/Human Acivity Recognition/Notebooks/App/phone.jpeg")
        st.image(image, caption='Smartphone Activity Tracker')
        
        random_int = st.slider('Select a range of Random Sensor value',0, len(df_transformed))
        val = df_transformed.iloc[random_int].values
    
    with col2:

        feature1 = st.text_input(label = 'tGravityAcc-energy()-X' ,value = val[0])
        feature2 = st.text_input(label ='tGravityAcc-mean()-X',value = val[1])
        feature3 = st.text_input(label ='tGravityAcc-max()-X',value = val[2])
        feature4 = st.text_input(label ='tGravityAcc-min()-X',value = val[3])
        feature5 = st.text_input(label ='angle(X,gravityMean)',value = val[4])
        feature6 = st.text_input(label ='tGravityAcc-min()-Y',value = val[5])
        feature7 = st.text_input(label ='tGravityAcc-mean()-Y',value = val[6])
        feature8 = st.text_input(label ='tGravityAcc-max()-Y',value = val[7])
        feature9 = st.text_input(label ='angle(Y,gravityMean)',value = val[8])
        feature10 = st.text_input(label ='tBodyAcc-max()-X',value = val[9])
        feature11 = st.text_input(label ='tGravityAcc-energy()-Y',value = val[10])
        feature12 = st.text_input(label ='fBodyAcc-entropy()-X',value = val[11])

        data_dict = {       'tGravityAcc-energy()-X':      feature1,
                            'tGravityAcc-mean()-X':   feature2,
                           'tGravityAcc-max()-X':      feature3,
                           'tGravityAcc-min()-X':       feature4,
                           'angle(X,gravityMean)':      feature5,
                           'tGravityAcc-min()-Y':       feature6,
                           'tGravityAcc-mean()-Y':       feature7,
                           'tGravityAcc-max()-Y':     feature8,
                           'angle(Y,gravityMean)':     feature9,
                           'tBodyAcc-max()-X':     feature10,
                           'tGravityAcc-energy()-Y': feature11,
                           'fBodyAcc-entropy()-X': feature12

                     }
        
        test_df = pd.DataFrame(val.reshape(1, -1))
        print(test_df.shape)
        #if st.button('Predict'):
        print(test_df)
        prediction = model_real_time_predict(test_df,model,le)
        st.write('Model Prediction is : ', prediction)
           #st.write(prediction.T)


        
        