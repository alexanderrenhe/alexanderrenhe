import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy
from tensorflow_core.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow_core.python.keras.regularizers import l2, l1, l1_l2
from tensorflow_core.python.keras.utils.np_utils import to_categorical
from keras_multi_head import MultiHeadAttention


def build_data_arr(data_fra):
    a1 = data_fra
    max_a = a1.a.max() + 1
    y_list = []
    x_list = []
    drop_list = ['b', 'a', 'a_list']
    # y_list.append(1)
    for l in range(1, max_a):
        b1 = a1.loc[a1['a'] == l]
        bb1 = b1
        y_list.append(bb1['b'].iloc[0])
        b1 = b1.drop(drop_list, axis=1)
        c1 = b1.values.tolist()
        x_list.append(c1)
        y_list.append(bb1['b'].iloc[0])
        c2 = c1[::-1]
        x_list.append(c2)
    ####将少于6张样本逐行重复一遍，提升下变化特征
    for i in range(0, len(x_list)):
        if len(x_list[i]) < 7:
            mm = []
            for j in range(0, len(x_list[i])):
                mm.append(x_list[i][j])
                mm.append(x_list[i][j])
                mm.append(x_list[i][j])
                mm.append(x_list[i][j])

            x_list[i] = mm

    return (y_list, x_list)


#################
def dataadd(x_train, y_train):
    x_train_add = []
    y_train_add = []
    for l in range(0, len(x_train)):
        h = len(x_train[l])
        w = len(x_train[l][0])
        addnum = int(0.3 * h * w)
        addlist = np.array(x_train[0])
        addlist2 = copy.copy(addlist)  
        for i1 in range(addnum):
            x2 = np.random.randint(0, h - 1)
            y2 = np.random.randint(11, w - 1)
            random_num2 = random.random()
            addlist2[:, y2] = addlist2[:, y2] * 1.3 if random_num2 >= 0.5 else addlist2[:, y2] * 0.7
        x_train_add.append(addlist2.tolist())
        y_train_add.append(y_train[l])


    return y_train_add, x_train_add


data_1 = pd.read_csv('***********')

data_100_1 = pd.read_csv('************', index_col=0)

feature_data100_1 = data_100_1.columns.values.tolist()
num_1 = 100

feature_data100 = feature_data100_1
data_100 = data_100_1
listnum = num_1
feature_a1 = feature_data100[0:listnum + 3]
a1 = data_100[feature_a1]
initial_datalist = build_data_arr(a1)
# 阶段性暂存 x y
y_Temp_deposit = initial_datalist[0]
x_Temp_deposit = initial_datalist[1]
#######
# 填补矩阵到listnum*25
len1 = len(x_Temp_deposit)
for m in range(0, len1):
    len2 = len(x_Temp_deposit[m])
    CZ = 25 - len2
    if CZ > 0:
        ZZ = np.zeros((CZ, listnum))
        x_Temp_deposit[m].extend(ZZ.tolist())

        ##########################################################
X_train, X_test, y_train, y_test = train_test_split(x_Temp_deposit, y_Temp_deposit, test_size=0.2, random_state=12)

data_add_list = dataadd(X_train, y_train)
X_addtrain1 = data_add_list[1]
Y_addtrain1 = data_add_list[0]
x_train = np.array(X_train + X_addtrain1)
y_train = np.array(y_train + Y_addtrain1)
x_test = np.array(X_test)
y_test = np.array(y_test)



np.random.seed(117)
np.random.shuffle(x_train)
np.random.seed(117)
np.random.shuffle(y_train)
tf.random.set_seed(117)

x_train = np.reshape(x_train, (len(x_train), 25, num_1))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), 25, num_1))
y_test = np.array(y_test)

y_train = to_categorical(y_train - 1, num_classes=3)
y_test = to_categorical(y_test - 1, num_classes=3)




def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


SINGLE_ATTENTION_VECTOR = True
TIME_STEPS = 25
INPUT_DIM = num_1


def model_attention_applied_after_lstm():
    K.clear_session()
    inputs1 = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, activation='sigmoid'))(inputs1)
    att_layer = MultiHeadAttention(head_num=8, name='Multi-Head')(lstm_out)
    attention_mul = Flatten()(att_layer)
    cc = Dropout(0.5)(attention_mul)
    aa = BatchNormalization()(cc)
    # d_d = Dense(256, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(aa)
    # aa = LeakyReLU(alpha=0.01)(d_d)
    # cc = Dropout(0.5)(aa)
    #aa = BatchNormalization()(cc)
    d_d = Dense(128, kernel_regularizer=l1_l2(l1=0.006, l2=0.01))(aa)
    aa = LeakyReLU(alpha=0.01)(d_d)
    cc = Dropout(0.5)(aa)
    aa = BatchNormalization()(cc)
    d_d = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.006, l2=0.01))(aa)
    cc = Dropout(0.5)(d_d)

    aa = BatchNormalization()(cc)
    d_d = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.005, l2=0.01))(aa)
    cc = Dropout(0.5)(d_d)
    aa = BatchNormalization()(cc)
    output = Dense(3, activation='softmax', kernel_regularizer=l1_l2(l1=0.005, l2=0.01))(aa)
    m1 = Model(inputs=inputs1, outputs=output)
    return m1

model = model_attention_applied_after_lstm()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

myReduce_lr = LearningRateScheduler(myScheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
history = model.fit(x_train, y_train, batch_size=16, epochs=500, validation_data=(x_test, y_test), validation_freq=1,
                    shuffle=True)

model.summary()
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
