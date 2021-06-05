from getDatabase import download_db
from read_json import clean_database

from timeit import default_timer as timer

from flask import Flask, request, render_template, jsonify

import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import keras
from keras.models import Model, load_model
from keras.layers import Dense, Input, Masking, Lambda, TimeDistributed, LSTM, LeakyReLU, Bidirectional

import numpy as np

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
    Generator_input = Input(shape =(None,LEN_FEATURES_SWIPE), dtype= 'float32')
    G = LSTM(units = 32, activation = 'relu',return_sequences = True)(Generator_input)
    #G = Lambda(repeat_vector, output_shape=(None, LEN_FEATURES_SWIPE)) ([G, Generator_input])
    G = LSTM(units = 16, activation = 'relu', return_sequences = True)(G)
    G = TimeDistributed(Dense(LEN_FEATURES_SWIPE))(G)   
    G = Lambda(lambda x: x-x[:,0:1,:])(G)
    Generator_model =  Model(Generator_input, G, name='Generator')
    Generator_optimizer = keras.optimizers.Nadam(lr=0.0002, beta_1=0.5,epsilon=1e-8)    
    Generator_model.compile(optimizer=Generator_optimizer, loss='mse')
    
    #Cargar pesos del generador ya entrenado ----------------------------------
    stp=550
    Generator_model = load_model('./GAN/swipe_generator/solo_swipe_gen_step{}.h5'.format(stp))
    
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

def predict_on_model(model,sequence):  
    # Devuelve un score para la secuencia de entrada
    score = model.predict(sequence)
    return score

def get_database():
    # Descarga y limpieza de BD
    
    db_name = 'database.json'
    start = timer()
    # Descargar BD desde Google
    download_db(db_name)
    end = timer()
    p1 = 'Tiempo lectura de la ultima conexion: {:.2f} segundos'.format(end-start)
    
    start = timer()
    # Limpiar IDs aleatorios de google
    clean_db = clean_database(db_name)
    end = timer()
    p2 = 'Tiempo procesado de la ultima conexion: {:.2f} segundos'.format(end-start)
    
    return p1,p2,clean_db


def get_swipe_gesture(clean_db):
    # Obtener el ultimo gesto de swipe
    last_conn_id = list(clean_db.items())[-1][0]
    swipedata = clean_db[last_conn_id]['SwipeGesture']
        
    swX = swipedata['X'][-1]
    swY = swipedata['Y'][-1]
    swT = swipedata['Timestamp'][-1]

    return swX,swY,swT

def plotView(sequence,sensor):
    # Mostrar un plot por pantalla

    if sequence != "":
        
        if sensor=="swipe":
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
            
        elif sensor=="lacc":
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
        
        return render_template("image.html", image=pngImageB64String)
    else:
        return ""
    
###############################################################################
# Precarga de modelos con la web
classifier_swipe = load_swipe_classifier()
generator_swipe = load_swipe_generator()
discriminator_swipe = load_swipe_discriminator()
classifier_lacc = load_lacc_classifier()
generator_lacc = load_lacc_generator()
discriminator_lacc = load_lacc_discriminator()
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
        
        return (render_template("image.html", image=pngImageB64String)
                    + "Score obtenido empleando el clasificador entre ruido y secuencias reales: "
                    + str(score1) +"<br>"
                    + "Score obtenido empleando el discriminador de la red GAN: "
                    + str(score2) +"<br>"
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
        
        return (render_template("image.html", image=pngImageB64String)
                    + "Score obtenido empleando el clasificador entre ruido y secuencias reales: "
                    + str(score1) +"<br>"
                    + "Score obtenido empleando el discriminador de la red GAN: "
                    + str(score2) +"<br>"
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
        
        return (render_template("image.html", image=pngImageB64String)
                    + "Score obtenido empleando el clasificador entre ruido y secuencias reales: "
                    + str(score1) +"<br>"
                    + "Score obtenido empleando el discriminador de la red GAN: "
                    + str(score2) +"<br>"
        )
    else:
        return ('',204)

@app.route("/rcv_lacc", methods=['POST'])
def receive_lacc():
    #Ver aceleracion asociada al swipe anterior (si lo hubo)
        
    if request.method == 'POST':
            
            # SACAR SEQS DE LACC A PARTIR DEL SUBSWIPE: CON LOS TIMESTAMPS!!
            # NECESITO RECIBIRLOS DE JS!
            # Y GUARDAR EL LACC Y ENVIARLO TAMBIEN
            # MEJOR DESDE JS TODO (?)
            
            seq = request.json
            x = np.array(seq['x'])
            y = np.array(seq['y'])
            z = np.array(seq['z'])
            lacc_available = np.array(seq['couldread'])
            
            print(lacc_available)
            
            if lacc_available==1:
            
                print("............")
                print(x)
                print("............")
                print(y)
                print("............")
                print(z)
                print("............")
                
                sequence = np.vstack((x,y)).T
                
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
                score1 = predict_on_model(classifier_swipe,sublacc)
                score2 = predict_on_model(discriminator_swipe,sublacc)
            
                return (render_template("image.html", image=pngImageB64String)
                            + "Score obtenido empleando el clasificador entre ruido y secuencias reales: "
                            + str(score1) +"<br>"
                            + "Score obtenido empleando el discriminador de la red GAN: "
                            + str(score2) +"<br>"
                )
            else:
                print("No se puede detectar el acelerometro")
                return ('',204)
    else:
        return ('',204)

@app.route("/ambos-sensores", methods=["GET","POST"])
def ambos_sensores():
    return  render_template("acelerometro.html")

if __name__ == "__main__":
    #app.run(host="127.0.0.1", port=8080, debug=True)
    #app.run(host="0.0.0.0", port=443)#, debug=True, ssl_context='adhoc')
    app.run()