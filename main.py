from timeit import default_timer as timer
from flask import Flask, request, render_template, jsonify

import io
import base64
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Input, Masking, Lambda, TimeDistributed, LSTM, LeakyReLU, Bidirectional, RepeatVector, Concatenate

########################################################################

NUM_MIN_SAMPLES_SWIPE = 5
LEN_SEQUENCES_SWIPE = 30
LEN_FEATURES_SWIPE = 2
NUM_MIN_SAMPLES_LACC = 10
LEN_SEQUENCES_LACC = 80
LEN_FEATURES_LACC = 3


app = Flask(__name__)


##-----------------------------------------------------------------------------

def load_swipe_classifier():
    #Carga del modelo del clasificador RUIDO vs REAL de SWIPE (NO el discriminador)
    Classifier_swipe_input = Input(shape = (LEN_SEQUENCES_SWIPE,LEN_FEATURES_SWIPE), dtype= 'float32')
    C = Masking(mask_value=0,input_shape=(LEN_SEQUENCES_SWIPE,LEN_FEATURES_SWIPE))(Classifier_swipe_input)
    C = LSTM(units = 64)(C)
    C = Dense(1, activation = 'sigmoid')(C)
    Classifier_swipe_model =  Model(Classifier_swipe_input, C)
    Classifier_swipe_model.compile(loss= 'binary_crossentropy',optimizer='nadam',metrics=['accuracy'])    
    
    #Carga de los pesos del modelo ya entrenado
    Classifier_swipe_model.load_weights('./GAN/swipe_classifier/SWIPE_classificador_ruido-real.h5')
    Classifier_swipe_model.load_weights('./GAN/swipe_classifier/SWIPE_classificador_ruido-real_solopesos.h5')

    return Classifier_swipe_model

def load_swipe_classifier2():
    #Carga del modelo del clasificador FAKE vs REAL de SWIPE (NO el discriminador)
    Classifier_swipe_model = load_model('./GAN/swipe_classifier2/SWIPE_classificador_fake-real.h5')
    Classifier_swipe_model.load_weights('./GAN/swipe_classifier2/SWIPE_classificador_fake-real_solopesos.h5')

    return Classifier_swipe_model

def load_swipe_discriminator():
    #Carga del modelo del discriminador de SWIPE
    Discriminator_input = Input(shape = (LEN_SEQUENCES_SWIPE, LEN_FEATURES_SWIPE), dtype= 'float32')
    D = Masking(mask_value=0,input_shape=(LEN_SEQUENCES_SWIPE, LEN_FEATURES_SWIPE))(Discriminator_input)
    D = Bidirectional(LSTM(units = 64, recurrent_dropout = 0.2))(D)
    D = Dense(32)(D)
    D = LeakyReLU(alpha = 0.2)(D)
    D = Dense(16)(D)
    D = LeakyReLU(alpha = 0.2)(D)
    D = Dense(1, activation = 'sigmoid')(D)
    Discriminator_model =  Model(Discriminator_input, D, name='Discriminator')
    Discriminator_optimizer = keras.optimizers.Nadam(lr=0.0002, beta_1=0.5)
    Discriminator_model.compile(loss= 'binary_crossentropy', optimizer=Discriminator_optimizer)      
    
    #Carga de los pesos del modelo ya entrenado
    Discriminator_model.load_weights('./GAN/swipe_discriminator/solo_swipe_dis_step550.h5')

    return Discriminator_model

def load_swipe_generator():
    #Cargar modelo del generador de SWIPE
    Generator_model = load_model('./GAN/swipe_generator/solo_swipe_gen_step550.h5')
    Generator_optimizer = keras.optimizers.Nadam(lr=0.0002, beta_1=0.5,epsilon=1e-8)    
    Generator_model.compile(optimizer=Generator_optimizer, loss='mse')
    
    return Generator_model

def generate_fake_sequence_swipe(Generator_model):
    #Generar secuencia a partir de ruido aleatorio ----------------------------
    rand_length = LEN_SEQUENCES_SWIPE
    while rand_length >=LEN_SEQUENCES_SWIPE or rand_length <= NUM_MIN_SAMPLES_SWIPE:
        rand_length = int(np.random.normal(loc=18.97, scale=4.91, size=None)) 
    
    random_latent_vectors=  np.random.normal(size=(1, rand_length,LEN_FEATURES_SWIPE))    
    generated_data = Generator_model.predict(random_latent_vectors)
    
    alfa = np.arctan2(generated_data[0][-1,1]-generated_data[0][0,1], generated_data[0][-1,0]-generated_data[0][0,0])
    angulo = (np.arctan(-generated_data[0][-1,1]/generated_data[0][-1,0])+alfa)
    rotacion = angulo-alfa
    dato_xrot= generated_data[0][:,0]*np.cos(rotacion)-generated_data[0][:,1]*np.sin(rotacion)
    dato_yrot= generated_data[0][:,0]*np.sin(rotacion)+generated_data[0][:,1]*np.cos(rotacion)
    generated_data[0] = np.array([dato_xrot, dato_yrot]).T
    
    sequence = np.hstack((generated_data,np.zeros((1,LEN_SEQUENCES_SWIPE-rand_length,LEN_FEATURES_SWIPE))))
    
    return sequence

##-----------------------------------------------------------------------------

def load_lacc_classifier():
    #Carga del modelo del clasificador RUIDO vs REAL de LACC (NO el discriminador)
    Classifier_lacc_input = Input(shape = (LEN_SEQUENCES_LACC,LEN_FEATURES_LACC), dtype= 'float32')
    C = Masking(mask_value=0,input_shape=(LEN_SEQUENCES_LACC,LEN_FEATURES_LACC))(Classifier_lacc_input)
    C = LSTM(units = 128)(C)
    C = Dense(1, activation = 'sigmoid')(C)
    Classifier_lacc_model =  Model(Classifier_lacc_input, C)
    Classifier_lacc_model.compile(loss= 'binary_crossentropy',optimizer='nadam',metrics=['accuracy']) 
    
    #Carga de los pesos del modelo ya entrenado
    Classifier_lacc_model.load_weights('./GAN/lacc_classifier/LACC_classificador_ruido-real.h5')
    Classifier_lacc_model.load_weights('./GAN/lacc_classifier/LACC_classificador_ruido-real_solopesos.h5')

    return Classifier_lacc_model

def load_lacc_classifier2():
    #Carga del modelo del clasificador RUIDO vs REAL de LACC (NO el discriminador)
    Classifier_lacc_model = load_model('./GAN/lacc_classifier2/LACC_classificador_fake-real.h5')
    Classifier_lacc_model.load_weights('./GAN/lacc_classifier2/LACC_classificador_fake-real_solopesos.h5')

    return Classifier_lacc_model

def load_lacc_discriminator():
    #Carga del modelo del discriminador de LACC
    Discriminator_input = Input(shape = (LEN_SEQUENCES_LACC, LEN_FEATURES_LACC), dtype= 'float32')
    D = Masking(mask_value=0,input_shape=(LEN_SEQUENCES_LACC, LEN_FEATURES_LACC))(Discriminator_input)
    D = Bidirectional(LSTM(units = 64, recurrent_dropout = 0.2))(D)
    D = Dense(32)(D)
    D = LeakyReLU(alpha = 0.2)(D)
    D = Dense(16)(D)
    D = LeakyReLU(alpha = 0.2)(D)
    D = Dense(1, activation = 'sigmoid')(D)
    Discriminator_model =  Model(Discriminator_input, D, name='Discriminator')
    Discriminator_optimizer = keras.optimizers.Nadam(lr=0.0002, beta_1=0.5)
    Discriminator_model.compile(loss= 'binary_crossentropy', optimizer=Discriminator_optimizer) 
    
    #Carga de los pesos del modelo ya entrenado
    Discriminator_model.load_weights('./GAN/lacc_discriminator/solo_lacc_dis_step350.h5')

    return Discriminator_model

def load_lacc_generator():
    Generator_input = Input(shape =(None,LEN_FEATURES_LACC), dtype= 'float32')
    G = Bidirectional(LSTM(units = 64, return_sequences = True))(Generator_input) 
    G = TimeDistributed(Dense(LEN_FEATURES_LACC))(G)   
    G = Lambda(lambda x: x-x[:,0:1,:])(G)
    Generator_model =  Model(Generator_input, G, name='Generator')
    Generator_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)    
    Generator_model.compile(optimizer=Generator_optimizer, loss='mse')
    #Carga de los pesos del modelo ya entrenado
    Generator_model.load_weights('./GAN/lacc_generator/solo_lacc_gen_step350.h5')

    return Generator_model

def generate_fake_sequence_lacc(Generator_model):
    #Generar secuencia a partir de ruido aleatorio ----------------------------
    rand_length = LEN_SEQUENCES_LACC
    while rand_length >=LEN_SEQUENCES_LACC or rand_length <= NUM_MIN_SAMPLES_LACC:
        rand_length = int(np.random.normal(loc=47.25, scale=22.79, size=None)) 
    
    random_latent_vectors=  np.random.normal(size=(1, rand_length,LEN_FEATURES_LACC))    
    generated_data = Generator_model.predict(random_latent_vectors)
    sequence = np.hstack((generated_data,np.zeros((1,LEN_SEQUENCES_LACC-rand_length,LEN_FEATURES_LACC))))

    return sequence
    
##-----------------------------------------------------------------------------

def load_2path2sensor_discriminator():

    Discriminator_input = Input(shape = (LEN_SEQUENCES_LACC, 5), dtype= 'float32')
    D = Masking(mask_value=0,input_shape=(LEN_SEQUENCES_LACC, 5))(Discriminator_input)
    D = Bidirectional(LSTM(units = 64))(Discriminator_input)
    D = Dense(1, activation = 'sigmoid')(D)
    Discriminator_model =  Model(Discriminator_input, D, name='Discriminator')
    Discriminator_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    Discriminator_model.compile(loss= 'binary_crossentropy', optimizer=Discriminator_optimizer)    

    Discriminator_model.load_weights('./GAN/2path2sensor_discriminator/2s2path_dis_step220.h5') 
    
    return Discriminator_model

def repeat_vector(args):
    # Para poder tener batches fake de longitud variable usando capas RepeatVector
    layer_to_repeat = args[0]
    sequence_layer = args[1]
    return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)

def load_2path2sensor_generator():

    Generator_model = load_model('./GAN/2path2sensor_generator/2s2path_gen_step220.h5')
    Generator_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5) 
    Generator_model.compile(optimizer=Generator_optimizer, loss='mse')

    return Generator_model

##-----------------------------------------------------------------------------

def predict_on_model(model,sequence):  
    # Devuelve un score para la secuencia de entrada
    score = model.predict(sequence)
    return score

def get_swipe_gesture(clean_db):
    # Obtener el ultimo gesto de swipe
    last_conn_id = list(clean_db.items())[-1][0]
    swipedata = clean_db[last_conn_id]['SwipeGesture']
        
    swX = swipedata['X'][-1]
    swY = swipedata['Y'][-1]
    swT = swipedata['Timestamp'][-1]

    return swX,swY,swT


def botorhuman_img(score, threshold):
    #Muestra imagen de bot o humano dependiendo del umbral en funcion del score
    if score <= threshold:
        res_img = '<img src="/static/human.png" alt="Humano">'
    else:
        res_img = '<img src="/static/bot.png" alt="Bot">'

    return res_img

###############################################################################
# Precarga de modelos con la web
classifier_swipe = load_swipe_classifier()
classifier_swipe2 = load_swipe_classifier2()
generator_swipe = load_swipe_generator()
discriminator_swipe = load_swipe_discriminator()
classifier_lacc2 = load_lacc_classifier2()
classifier_lacc = load_lacc_classifier()
generator_lacc = load_lacc_generator()
discriminator_lacc = load_lacc_discriminator()

generator_2path2sensor = load_2path2sensor_generator()
discriminator_2path2sensor = load_2path2sensor_discriminator()

###############################################################################
    
@app.route("/", methods=["GET","POST"])
def index():
    return (render_template("index.html"))

@app.route("/interaccion-tactil", methods=["GET","POST"])
def interaccion_tactil():
    return (render_template("interaccion-tactil.html"))
    
@app.route("/rcv_swipeGANrequest", methods=["POST"])
def receive_swipe_GAN_request():
    if request.method == 'POST':
        
        sequence = generate_fake_sequence_swipe(generator_swipe)
        
        #Eliminar la "vuelta a cero" de los plots
        x2_gen = sequence[0][:,0].copy()
        y2_gen = sequence[0][:,1].copy()
        for e in range(len(sequence[0][:,0])-1,1,-1):
            if x2_gen[e] == 0.00000000e+00 and y2_gen[e] == 0.00000000e+00:
                x2_gen = np.delete(x2_gen, e)
                y2_gen = np.delete(y2_gen, e)
        
        # Generate plot
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title("Secuencia de interacción táctil generada por la red GAN")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid()
        axis.plot(x2_gen, y2_gen, "ro-")
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        
        #Obtener scores
        score1 = predict_on_model(classifier_swipe,sequence)
        score2 = predict_on_model(discriminator_swipe,sequence)
        score3 = predict_on_model(classifier_swipe2,sequence)

        img_score1 = botorhuman_img(score1,0.5)
        img_score2 = botorhuman_img(score2,0.3)
        img_score3 = botorhuman_img(score3,0.55)
        
        return (render_template("image.html", image=pngImageB64String)
                    + '<script> var x = document.getElementById("ganbox"); x.style.display = "none"; </script>'
                    + '<script> var x = document.getElementById("swipebox"); x.style.display = "none"; </script>'
                    + '<ul><li>Score obtenido empleando el <strong>clasificador entre ruido y secuencias reales</strong>:</li><ul class="no-bullets"><li>'
                    + str(score1[0])+img_score1
                    + '</li></ul><ul><li style="color:blue;">Resultado deseable: Humano.</li><li style="color:blue;">Interpretación: ¿Es la GAN capaz de generar secuencias muy similares a las humanas?</li></ul>'
                    +'<li>Score obtenido empleando el <strong>discriminador de la red GAN</strong>:</li><ul class="no-bullets"><li>'
                    + str(score2[0])+img_score2
                    + '</li></ul><ul></li><li style="color:blue;">Resultado deseable: Bot.</li><li style="color:blue;">Interpretación: ¿Es el discriminador de la GAN capaz de distinguir entre bot y humano?</li></ul>'
                    +'<li>Score obtenido empleando el <strong>clasificador entre secuencias reales y generadas por la GAN</strong>:</li><ul class="no-bullets"><li>'
                    + str(score3[0])+img_score3
                    + '</li></ul><ul></li><li style="color:blue;">Resultado deseable: Bot.</li><li style="color:blue;">Interpretación: ¿Es el clasificador capaz de distinguir entre bot y humano?</li></ul></ul>'
        )
    else:
        return ('',204)
    
@app.route("/rcv_swipe", methods=['POST'])
def receive_swipe():
    #Leer el swipe generado en el cuadro blanco y mostrarlo por pantalla
        
    if request.method == 'POST':
        
        seq = request.json
        x = np.array(seq['x'])
        y = np.array(seq['y'])
        sequence = np.vstack((x,y)).T
        
        # Generate plot
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title("Secuencia de interacción táctil generada")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid()
        axis.plot(x, y, "ro-")
        
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        
        #Ajustar el numero de muestras de las secuencias a 30. 
        subswipe = sequence
        if len(sequence[:,0]) > LEN_SEQUENCES_SWIPE: 
            #Trunco si tengo mas muestras
            subswipe = subswipe[:LEN_SEQUENCES_SWIPE,:]
        elif len(subswipe[:,0]) < LEN_SEQUENCES_SWIPE: 
            #Relleno con ceros si tengo menos muestras
            numzeros = LEN_SEQUENCES_SWIPE - len(subswipe[:,0])
            subswipe = np.vstack((subswipe,np.zeros((numzeros,2))))

        subswipe = np.reshape(subswipe,(1,LEN_SEQUENCES_SWIPE,LEN_FEATURES_SWIPE))

        #Obtener scores
        score1 = predict_on_model(classifier_swipe,subswipe)
        score2 = predict_on_model(discriminator_swipe,subswipe)
        score3 = predict_on_model(classifier_swipe2,subswipe)

        img_score1 = botorhuman_img(score1,0.5)
        img_score2 = botorhuman_img(score2,0.6)
        img_score3 = botorhuman_img(score3,0.55)
        
        return (render_template("image.html", image=pngImageB64String)
                    + '<script> var x = document.getElementById("ganbox"); x.style.display = "none"; </script>'
                    + '<ul><li>Score obtenido empleando el <strong>clasificador entre ruido y secuencias reales</strong>:</li><ul class="no-bullets"><li>'
                    + str(score1[0])+img_score1
                    + '</li></ul><ul><li style="color:blue;">Resultado deseable: Humano.</li><li style="color:blue;">Interpretación: ¿Se ha registrado una secuencia realista, o solo ruido?</li></ul>'
                    +'<li>Score obtenido empleando el <strong>discriminador de la red GAN</strong>:</li><ul class="no-bullets"><li>'
                    + str(score2[0])+img_score2
                    + '</li></ul><ul></li><li style="color:blue;">Resultado deseable: Humano.</li><li style="color:blue;">Interpretación: ¿Es el discriminador de la GAN capaz de distinguir entre bot y humano?</li></ul>'
                    +'<li>Score obtenido empleando el <strong>clasificador entre secuencias reales y generadas por la GAN</strong>:</li><ul class="no-bullets"><li>'
                    + str(score3[0])+img_score3
                    + '</li></ul><ul></li><li style="color:blue;">Resultado deseable: Humano.</li><li style="color:blue;">Interpretación: ¿Es el clasificador capaz de distinguir entre bot y humano?</li></ul></ul>'
        )
    else:
        return ('',204)

@app.route("/acelerometro", methods=["GET","POST"])
def acelerometro():
    return  render_template("acelerometro.html")

@app.route("/rcv_laccGANrequest", methods=["POST"])
def receive_lacc_GAN_request():
    if request.method == 'POST':
        
        sequence = generate_fake_sequence_lacc(generator_lacc)
        score1 = predict_on_model(classifier_lacc,sequence)
        score2 = predict_on_model(discriminator_lacc,sequence)
        score3 = predict_on_model(classifier_lacc2,sequence)
        
        #Eliminar la "vuelta a cero" de los plots
        x2_gen = sequence[0][:,0].copy()
        y2_gen = sequence[0][:,1].copy()
        z2_gen = sequence[0][:,2].copy()
        for e in range(len(sequence[0][:,0])-1,1,-1):
            if x2_gen[e] == 0.00000000e+00 and y2_gen[e] == 0.00000000e+00 and z2_gen[e] == 0.00000000e+00:
                x2_gen = np.delete(x2_gen, e)
                y2_gen = np.delete(y2_gen, e)
                z2_gen = np.delete(z2_gen, e)
                
        # ACELEROMETRO vs NUM MUESTRAS
        fig, axs = plt.subplots(3, 1)
        #fig.set_title("Secuencia de aceleración generada por la red GAN")
        fig.tight_layout(pad=2)
        axs[0].set_title('X')
        axs[0].plot(x2_gen,color='r',ls='--')
        axs[1].set_title('Y')
        axs[1].plot(y2_gen,color='g',ls='--')
        axs[2].set_title('Z')
        axs[2].plot(z2_gen,color='b',ls='--')

        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

        img_score1 = botorhuman_img(score1,0.5)
        img_score2 = botorhuman_img(score2,0.6)
        img_score3 = botorhuman_img(score3,0.5)
        
        return (render_template("image.html", image=pngImageB64String)
                    + '<script> var x = document.getElementById("ganbox"); x.style.display = "none"; </script>'
                    + '<script> var x = document.getElementById("laccbox"); x.style.display = "none"; </script>'
                    + '<ul><li>Score obtenido empleando el <strong>clasificador entre ruido y secuencias reales</strong>:</li><ul class="no-bullets"><li>'
                    + str(score1[0])+img_score1
                    + '</li></ul><ul><li style="color:blue;">Resultado deseable: Humano.</li><li style="color:blue;">Interpretación: ¿Es la GAN capaz de generar secuencias muy similares a las humanas?</li></ul>'
                    +'<li>Score obtenido empleando el <strong>discriminador de la red GAN</strong>:</li><ul class="no-bullets"><li>'
                    + str(score2[0])+img_score2
                    + '</li></ul><ul></li><li style="color:blue;">Resultado deseable: Bot.</li><li style="color:blue;">Interpretación: ¿Es el discriminador de la GAN capaz de distinguir entre bot y humano?</li></ul>'
                    +'<li>Score obtenido empleando el <strong>clasificador entre secuencias reales y generadas por la GAN</strong>:</li><ul class="no-bullets"><li>'
                    + str(score3[0])+img_score3
                    + '</li></ul><ul></li><li style="color:blue;">Resultado deseable: Bot.</li><li style="color:blue;">Interpretación: ¿Es el clasificador capaz de distinguir entre bot y humano?</li></ul></ul>'
        )
    else:
        return ('',204)

@app.route("/rcv_lacc", methods=['POST'])
def receive_lacc():
    #Ver aceleracion asociada al swipe anterior (si lo hubo)
        
    if request.method == 'POST':
            
            seq = request.json
            x = np.array(seq['x'])
            y = np.array(seq['y'])
            z = np.array(seq['z'])

            lacc_available = np.array(seq['couldread'])
            
            if str(lacc_available)=='1' and str(x)!='null' and str(x)!=None and str(x)!='None':

                sequence = np.vstack((x,y,z)).T
                
                # ACELEROMETRO vs NUM MUESTRAS
                fig, axs = plt.subplots(3, 1)
                #fig.set_title("Secuencia de aceleración generada por la red GAN")
                fig.tight_layout(pad=2)
                axs[0].set_title('X')
                axs[0].plot(x,color='r',ls='--')
                axs[1].set_title('Y')
                axs[1].plot(y,color='g',ls='--')
                axs[2].set_title('Z')
                axs[2].plot(z,color='b',ls='--')
                
                # Convert plot to PNG image
                pngImage = io.BytesIO()
                FigureCanvas(fig).print_png(pngImage)
                
                # Encode PNG image to base64 string
                pngImageB64String = "data:image/png;base64,"
                pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
                
                #Ajustar el numero de muestras de las secuencias a 80. 
                sublacc = sequence
                if len(sequence[:,0]) > LEN_SEQUENCES_LACC: 
                    #Trunco si tengo mas muestras
                    sublacc = sublacc[:LEN_SEQUENCES_LACC,:]
                elif len(sublacc[:,0]) < LEN_SEQUENCES_LACC: 
                    #Relleno con ceros si tengo menos muestras
                    numzeros = LEN_SEQUENCES_LACC - len(sublacc[:,0])
                    sublacc = np.vstack((sublacc,np.zeros((numzeros,3))))
        
                sublacc = np.reshape(sublacc,(1,LEN_SEQUENCES_LACC,LEN_FEATURES_LACC))
        
                #Obtener scores
                score1 = predict_on_model(classifier_lacc,sublacc)
                score2 = predict_on_model(discriminator_lacc,sublacc)
                score3 = predict_on_model(classifier_lacc2,sublacc)

                img_score1 = botorhuman_img(score1,0.5)
                img_score2 = botorhuman_img(score2,0.6)
                img_score3 = botorhuman_img(score3,0.5)
            
                return (render_template("image.html", image=pngImageB64String)
                            + '<script> var x = document.getElementById("ganbox"); x.style.display = "none"; </script>'
                            + '<script> var x = document.getElementById("laccbox"); x.style.display = "none"; </script>'
                            + '<ul><li>Score obtenido empleando el <strong>clasificador entre ruido y secuencias reales</strong>:</li><ul class="no-bullets"><li>'
                            + str(score1[0])+img_score1
                            + '</li></ul><ul><li style="color:blue;">Resultado deseable: Humano.</li><li style="color:blue;">Interpretación: ¿Se ha registrado una secuencia realista, o solo ruido?</li></ul>'
                            +'<li>Score obtenido empleando el <strong>discriminador de la red GAN</strong>:</li><ul class="no-bullets"><li>'
                            + str(score2[0])+img_score2
                            + '</li></ul><ul></li><li style="color:blue;">Resultado deseable: Humano.</li><li style="color:blue;">Interpretación: ¿Es el discriminador de la GAN capaz de distinguir entre bot y humano?</li></ul>'
                            +'<li>Score obtenido empleando el <strong>clasificador entre secuencias reales y generadas por la GAN</strong>:</li><ul class="no-bullets"><li>'
                            + str(score3[0])+img_score3
                            + '</li></ul><ul></li><li style="color:blue;">Resultado deseable: Humano.</li><li style="color:blue;">Interpretación: ¿Es el clasificador capaz de distinguir entre bot y humano?</li></ul></ul>'
                )
            else:
                if (str(x)==None or str(x)=='null' or str(lacc_available)=='1'):
                    return '<script> var x = document.getElementById("laccbox"); x.style.display = "none"; </script> No se ha registrado interaccion táctil. Debe registrarse un movimiento previamente para capturar la aceleración producida durante el mismo. Puede capturarse <a href=/interaccion-tactil>aquí</a>.'
                else:
                    return '<script> var x = document.getElementById("laccbox"); x.style.display = "none"; </script> No se puede detectar el acelerómetro. Pruebe desde Google Chrome o asegúrese de que su dispositivo cuenta con acelerómetro incorporado.'
    else:
        return ('',204)

@app.route("/rcv_2path2sensorGANrequest", methods=["POST"])
def receive_2path2sensor_GAN_request():
    if request.method == 'POST':
        
        sequence = predict_on_model(generator_2path2sensor,np.random.normal(size=(1,LEN_SEQUENCES_LACC,5)))
        score = predict_on_model(discriminator_2path2sensor,sequence)

        # Swipe -----------------
        #Eliminar la "vuelta a cero" de los plots
        x2_gen = sequence[0][:,0].copy()
        y2_gen = sequence[0][:,1].copy()
        for e in range(len(sequence[0][:,0])-1,1,-1):
            if x2_gen[e] == 0.00000000e+00 and y2_gen[e] == 0.00000000e+00:
                x2_gen = np.delete(x2_gen, e)
                y2_gen = np.delete(y2_gen, e)
        
        # Generate plot
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title("Secuencia de interacción táctil generada por la red GAN")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid()
        axis.plot(x2_gen, y2_gen, "ro-")
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        # Encode PNG image to base64 string
        pngImageB64String_swipe = "data:image/png;base64,"
        pngImageB64String_swipe += base64.b64encode(pngImage.getvalue()).decode('utf8')

        # Lacc -----------------
        #Eliminar la "vuelta a cero" de los plots
        x2_gen = sequence[0][:,2].copy()
        y2_gen = sequence[0][:,3].copy()
        z2_gen = sequence[0][:,4].copy()
        for e in range(len(sequence[0][:,0])-1,1,-1):
            if x2_gen[e] == 0.00000000e+00 and y2_gen[e] == 0.00000000e+00 and z2_gen[e] == 0.00000000e+00:
                x2_gen = np.delete(x2_gen, e)
                y2_gen = np.delete(y2_gen, e)
                z2_gen = np.delete(z2_gen, e)
                
        # ACELEROMETRO vs NUM MUESTRAS
        fig, axs = plt.subplots(3, 1)
        #fig.set_title("Secuencia de aceleración generada por la red GAN")
        fig.tight_layout(pad=2)
        axs[0].set_title('X')
        axs[0].plot(x2_gen,color='r',ls='--')
        axs[1].set_title('Y')
        axs[1].plot(y2_gen,color='g',ls='--')
        axs[2].set_title('Z')
        axs[2].plot(z2_gen,color='b',ls='--')

        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        # Encode PNG image to base64 string
        pngImageB64String_lacc = "data:image/png;base64,"
        pngImageB64String_lacc += base64.b64encode(pngImage.getvalue()).decode('utf8')
        
        img_score = botorhuman_img(score,0.5)

        return (render_template("image.html", image=pngImageB64String_swipe)
            + render_template("image.html", image=pngImageB64String_lacc)
            + '<script> var x = document.getElementById("ganbox"); x.style.display = "none"; </script>'
            + '<ul><li>Score obtenido empleando el discriminador de la red GAN:</li><ul class="no-bullets"><li>'
            + str(score[0])+img_score
            + '</li></ul><ul></li><li style="color:blue;">Resultado deseable: Bot.</li><li style="color:blue;">Interpretación: ¿Es el discriminador de la GAN capaz de distinguir entre bot y humano?</li></ul></ul>'
        )
    else:
        return ('',204)

@app.route("/ambos-sensores", methods=["GET","POST"])
def ambos_sensores():
    return  render_template("ambos-sensores.html")

if __name__ == "__main__":
    app.run()
    #app.run(host="0.0.0.0", port=8080, debug=True, ssl_context='adhoc')