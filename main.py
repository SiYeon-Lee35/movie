import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    #https://github.com/e9t/nsmc/에서 훈련데이터 다운로드
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
    #훈련 데이터와 테스터 데이터 저장
    train_data = pd.read_table('ratings_train.txt')
    test_data = pd.read_table('ratings_test.txt')

    #데이터 전처리 함수
    def data_preprocessing(data):
    
        print("데이터 전처리 전 데이터 개수 : {0}.", len(data))
        
        #중복된 샘플 제거
        data.drop_duplicates(subset=['document'], inplace=True)

        #정규식을 이용해 한글과 공백을 제외하고 모두 제거
        data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

        #빈값을 가지는 행을 Null값으로 변경 (한글로 작성된 리뷰가 아니라면 빈값이 됨)
        data['document'] = data['document'].str.replace('^ +', "")
        data['document'].replace('', np.nan, inplace=True)
        data = data.dropna()

        print("데이터 전처리 후 데이터 개수 : {0}.", len(data))

    #train data 전처리
    data_preprocessing(train_data)
    train_data = train_data.dropna()
    #test data 전처리
    data_preprocessing(test_data)
    test_data = test_data.dropna()


    #토큰화 : 불용어 제거
    def tokenization(data):
    #불용어
        stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
        okt = Okt()
        output= []
        for sentence in tqdm(data['document']):
            tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
            output.append(stopwords_removed_sentence)
        return output   
    X_train = tokenization(train_data)
    X_test = tokenization(test_data)   

    #정수인코딩
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    threshold = 3
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value
            
    # 빈도수가 2이하인 단어들의 수를 제외
    vocab_size = total_cnt - rare_cnt + 1

    tokenizer = Tokenizer(vocab_size) 
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    #빈샘플 제거
    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)

    #패딩
    max_len = 30
    #샘플의 길이를 30으로 맞춤
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    #LSTM
    from tensorflow.keras.layers import Embedding, Dense, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import load_model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    embedding_dim = 100
    hidden_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation='sigmoid'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)   

    #정확도 측정
    loaded_model = load_model('best_model.h5')
    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

    #네이버 영화 리뷰 크롤링
    import requests as req
    from bs4 import BeautifulSoup as bs
    def move_crawling(site, n):
        data = []

        #1~n페이지 리뷰 수집    
        for i in range(1,n+1):
            #웹페이지 정보 요청
            res = req.get(site[:len(site)-1] + str(i))

            #HTML 형식으로 변환
            soup = bs(res.text, 'lxml')

            #관락객 요소 선택
            viewer = soup.select('span.ico_viewer')

            #관람객 요소 추출 및 제거
            for i in viewer:
                i.extract()

            # 관람객 요소가 삭제된 해당 요소 추출     
            review = soup.select("div.score_reple > p")

            #개행 문자 삭제
            for i in review:
                data.append(i.text.strip())
        return data

    #리뷰 예측
    
    def sentiment_predict(new_sentence):
        okt = Okt()
        stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
        new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
        new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
        encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
        score = float(loaded_model.predict(pad_new)) # 예측
        if(score > 0.6):#긍정
            return 1
        elif(score<0.5):#부정
            return -1
        else:#중립
            return 0

    score = []
    review_data = move_crawling('https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=74977&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page=1', 10)
    for i in review_data:
        score.append(sentiment_predict(i))
        
    b_type = [score.count(-1), score.count(0), score.count(1)]
    b_name = ['부정', '중립', '긍정']
    plt.rc('font', family='Malgun Gothic')
    plt.pie(b_type, labels=b_name, autopct='%1.1f%%')
    plt.show()
    
if __name__=='__main__':
    main()
