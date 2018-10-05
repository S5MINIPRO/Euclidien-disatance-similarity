import matplotlib.pyplot as plt
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import wordnet,stopwords
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tag import pos_tag
import math,re
from math import sqrt
from collections import Counter
import numpy as np
from numpy import sqrt




input_text=open("model.txt","r")
input_text1=open("ans1.txt","r")

text=input_text.read()
text1=input_text1.read()

input_text.close()
input_text1.close()

sentences=sent_tokenize(text)
sentences1=sent_tokenize(text1)

N=len(sentences)
N1=len(sentences1)



ps=PorterStemmer()
lemmatizer=WordNetLemmatizer()
stop_words=stopwords.words('english')
special=['.',',','\'','"','-','/','*','+','=','!','@','$','%','^','&','``','\'\'','We','The','This']






def normalise(word):
    word = word.lower()
    word = ps.stem(word)
    return word


def euclidean_distance(x,y):
 
    return float(sqrt(sum(pow(x[a]-y[b],2) for a in x for b in y if(a==b))))

def text_to_vector(text):
     words = word_tokenize(text)
     vec=[]
     for word in words:
         if(word not in stop_words):
             if(word not in special):
                 w=normalise(word);
                 vec.append(w);
     #print Counter(vec)
     return Counter(vec)



def docu_to_vector(sent):
     vec=[]
     for text in sent:
         words = word_tokenize(text)
         for word in words:
             if(word not in stop_words):
                 if(word not in special):
                     w=normalise(word);
                     vec.append(w);
     #print Counter(vec)
     return Counter(vec)




def f_s_to_s(sent):
    cosine_mat=np.zeros(N+1)
   
    row=0
    for text in sentences:
        maxi=0
        vector1 = text_to_vector(text)
        for text1 in sent:
            vector2 = text_to_vector(text1)
            cosine = euclidean_distance(vector1, vector2)

        for text2 in sent:
            vector3 = text_to_vector(text2)
            cosine = euclidean_distance(vector1, vector3)

        for text3 in sent:
            vector4 = text_to_vector(text3)
            cosine = euclidean_distance(vector1, vector4)


        for text4 in sent:
            vector5 = text_to_vector(text4)
            cosine = euclidean_distance(vector1, vector5)
       
        for text5 in sent:
            vector6 = text_to_vector(text5)
            cosine = euclidean_distance(vector1, vector6)

        for text6 in sent:
            vector7 = text_to_vector(text6)
            cosine = euclidean_distance(vector1, vector7)

        for text7 in sent:
            vector8 = text_to_vector(text7)
            cosine = euclidean_distance(vector1, vector8)


        for text8 in sent:
            vector9 = text_to_vector(text4)
            cosine = euclidean_distance(vector1, vector9)

        for text9 in sent:
            vector10 = text_to_vector(text9)
            cosine = euclidean_distance(vector1, vector10)
            
            if(maxi<cosine):
                maxi=cosine
               
        cosine_mat[row]=maxi
             
        row+=1
        
    return cosine_mat   
    
def main():
    mat = f_s_to_s(sentences1)
    print (mat)
    point1= sum(mat)
    print (point1)
    temp1=[]
    n=len(mat)
    for i in range(1,n+1):
    	temp1.append(i)
    
	
    plt.scatter(temp1,mat)
    plt.xlabel('line')
    plt.ylabel('points')
    plt.title('euclidean similarity graph')
    plt.show()  
  
    
if __name__ == '__main__':
    main()
    
   


