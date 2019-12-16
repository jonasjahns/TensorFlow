import tensorflow as tf
from Metrics import custom_metrics as cm

# Carga do dataset mnist http://yann.lecun.com/exdb/mnist/
mnist = tf.keras.datasets.mnist

# Divisão entre base de teste e base de treino
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Conversão do valor dos pixeis para preto e branco
x_train, x_test = x_train/255.0, x_test/255.0

# Criação do modelo Keras Sequencial
model = tf.keras.models.Sequential([
    # Transforma a primeira camada de  28x28 para um array de 784 pixels
    # Essa camada não aprende nada, apenas realiza a reformatação dos dados
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Criação de uma camada densa com 128 neuônios
    # Função de ativação Relu, retorna o max(x,0)
    tf.keras.layers.Dense(128, activation='relu'),
    # Criação de uma camada de Droupout para 20% dos casos serem alterados para 0, evitando overfitting
    tf.keras.layers.Dropout(0.2),
    # Criação de uma camada densa com saida de tamanho 10, uma para cada classe de classificação do problema
    # A saída da camada são probabilidades que somam 100%
    tf.keras.layers.Dense(10, activation='softmax')
])
# Estapa de compilação do modelo
# Loss function trata o quão acertivo o modelo é durante o treinamento
# O modelo usuado no exemplo é recomendado para classes mutuamente exclusivas
# Ou seja, uma imagem pode pertencer apenas a uma das classes do problema
# Optimizer é utilizado para otimizar o aprendizado do problema
# O modelo Adam, publicado em 2014, utiliza adaptative learning
# Para que cada parâmetro do modelo tenha taxas de aprendizado individuais
# Por fim, são incluidas as métricas do modelo apresentadas durante cada passo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', cm.f1_m, cm.recall_m])


# Criação do Modelo de Deep Learning, com 5 iteraçoes no dataset de treino
model.fit(x_train, y_train, epochs=5)

# Avaliação do modelo treinado com os dados separados para teste com versobse=2
model.evaluate(x_test, y_test, verbose=2)

