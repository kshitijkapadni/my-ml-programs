 
import numpy as np
from gensim import models 

from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
import theano
from keras.models import Sequential

model = models.KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)


lines = open('movie_lines.txt',encoding="utf-8",errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt',encoding="utf-8",errors='ignore').read().split('\n')

id2line = {}

for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]

convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))

questions = 'a' 
answers ='a'

for conv in convs:
    for i in range(len(conv)-1):
        questions+=(id2line[conv[i]])
        answers+=(id2line[conv[i+1]])

def clean_text(text):
    text = text.lower()

    from nltk.tokenize import word_tokenize
    tokens  = word_tokenize(text)
    
    import string
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table)for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    del words[20000:]
    return words

tok_questions = clean_text(questions)

tok_answers = clean_text(answers)
sentend=np.ones((300,),dtype=np.float32)
vec_que= []
for sent in tok_questions:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_que.append(sentvec)

vec_ans = []
for sent in tok_answers:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_ans.append(sentvec)

for tok_sent in vec_que:
    tok_sent[19:]=[]
    tok_sent.append(sentend)

for tok_sent in vec_que:
    if len(tok_sent)<20:
        for i in range(20-len(tok_sent)):
            tok_sent.append(sentend)

for tok_sent in vec_ans:
    tok_sent[19:]=[]
    tok_sent.append(sentend)

for tok_sent in vec_ans:
    if len(tok_sent)<20:
        for i in range(20-len(tok_sent)):
            tok_sent.append(sentend)

#with open('chatbotQA.h5','wb')as f:
 #   pickle.dump([vec_que,vec_ans],f,protocol=pickle.HIGHEST_PROTOCOL)
print(len(vec_que))
print(len(vec_ans))
minl = min(len(vec_que),len(vec_ans))
del vec_ans[minl:]
print(len(vec_ans))
theano.config.optimizer = "None"
model_vec_que  = np.array(vec_que)
model_vec_ans  = np.array(vec_ans)

que_train,que_test,ans_train,ans_test = train_test_split(model_vec_que,model_vec_ans,test_size=0.2,random_state = 1)

cmodel = Sequential()
cmodel.add(LSTM(input_shape= que_train.shape[1:],return_sequences=True,units=300,kernel_initializer='glorot_normal'))
cmodel.add(LSTM(input_shape= que_train.shape[1:],return_sequences=True,units=300,kernel_initializer='glorot_normal'))
cmodel.add(LSTM(input_shape= que_train.shape[1:],return_sequences=True,units=300,kernel_initializer='glorot_normal'))
cmodel.add(LSTM(input_shape= que_train.shape[1:],return_sequences=True,units=300,kernel_initializer='glorot_normal'))
cmodel.compile(loss='cosine_proximity',optimizer='adam',metrics=['accuracy'])

cmodel.fit(que_train,ans_train,epochs=50,validation_data=(que_test,ans_test),steps_per_epoch=500,validation_steps=125)

cmodel.save('chatbot.h5');           
