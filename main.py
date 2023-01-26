'''**************************************** Law *********************************************'''
# N = number of doc
# df = doc frequency (number of the term showed up in all the doc)
# tf = term frequency (number of the term showed up in a certain doc)
# idf = log (N / df)
# tf weight = 1+log(tf)
# tf-idf = tf weight * idf
# doc length = root (the sum of each tf-idf power by 2) 
# normalized = tf-idf / doc length
# similarity = sum(normalized of each query term * normalized of each doc term)
'''**************************************** import liprary *********************************************'''
import os
import nltk
nltk.download('stopwords')
from nltk.stem import lancaster
from nltk.tokenize import TweetTokenizer
from natsort import natsorted
from nltk.corpus import stopwords
import math
stop_words = set(stopwords.words('english'))
import string
from termcolor import colored,cprint
print_red=lambda x:cprint(x,'red')
print_green=lambda x:cprint(x,'green')

# read file
def read_file(filename):
    with open(filename, 'r', encoding="utf-8", errors="surrogateescape") as f:
        stuff = f.read()
    f.close()
    return stuff

# Preprocessing all terms in files
def preprocessing(final_string):
    '''***************** Tokenize ************************'''
    tokenizer = TweetTokenizer()
    token_list = tokenizer.tokenize(final_string)
    '''***************** Remove punctuations ************************'''
    table = str.maketrans('', '', '\t')
    token_list = [word.translate(table) for word in token_list]
    punctuations = (string.punctuation).replace("'", "")
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in token_list]
    token_list = [str for str in stripped_words if str]
    '''*********** Change to lowercase ************************'''
    token_list = [word.lower() for word in token_list]
    freqForEachFolder = len(token_list)  
    return token_list,freqForEachFolder

    '''***************** create folder of files that will br in positional index ************************'''

# initializing
folder_names = ["ir"]
fileno = 0
pos_index = {}
file_map = {}
freq = {} 
counter = 0
alltokens = []
stop_words.remove('in')
stop_words.remove('to')
stop_words.remove('where')

for folder_name in folder_names:

    file_names = natsorted(os.listdir( folder_name))
    for file_name in file_names:

        stuff = read_file( folder_name + "/" + file_name)
        final_token_list,freq[counter] = preprocessing(stuff)
        # Filtered_pharseQuery = [w for w in final_token_list if not w in stop_words]
        for pos, term in enumerate(final_token_list):
            # term = stemmer.stem(term)
            if term in pos_index:
                pos_index[term][0] = pos_index[term][0] + 1
                if fileno in pos_index[term][1]:
                    pos_index[term][1][fileno].append(pos)
                else:
                    pos_index[term][1][fileno] = [pos]
            else:
                pos_index[term] = []
                pos_index[term].append(1)
                pos_index[term].append({})
                pos_index[term][1][fileno] = [pos]
                alltokens.append(term)
        file_map[fileno] = folder_name + "/" + file_name
        fileno += 1
        counter +=1
print_red("********************pos_index********************")  
print(pos_index)
print_red("********************************************************")  

'''********************** Inserting query /stopWord and filter it with stemmer on query *****************************'''
pharseQuery=input("Enter your pharse Query To search for it :")

stop_words = set(stopwords.words('english'))
pharseQuery, freq2 = preprocessing(pharseQuery)
stop_words.remove('in')
stop_words.remove('to')
stop_words.remove('where')

Filtered_pharseQuery = [w for w in pharseQuery if not w in stop_words]
print(Filtered_pharseQuery)
print_red("********************************************************")  

'''************************* Query part *****************************'''
# Filtered_pharseQuery = list(dict.fromkeys(Filtered_pharseQuery))
x=len(Filtered_pharseQuery)
i=0
arr = []
tf_idf = [[0 for c in range(len(file_names))]for r in range(len(Filtered_pharseQuery))]
tf = 0
sum_of_tf_idf = 0
doc_len_query = 0
for i in range(0,x):
    try:
        sample_pos_idx = pos_index[Filtered_pharseQuery[i]]
        print('\nThe word is :',Filtered_pharseQuery[i])
        print("Positional Index --> "+str(i+1)+" word")
        #print(sample_pos_idx)
        frequancey=sample_pos_idx[0]
        print("Frequancey :",frequancey)
        file_list = sample_pos_idx[1]
        print("Filename, [Positions]")
        counter=0
        for fileno, positions in file_list.items():
            counter+= len(positions)
            print(file_map[fileno], positions)

        tf= Filtered_pharseQuery.count(Filtered_pharseQuery[i])
        print("df :",counter)
        df=counter
        idf=math.log10(10/df)
        TfWeight= 1 + math.log10(tf)
        Tf_idf = TfWeight*idf
        doc_len_query += Tf_idf*Tf_idf 
        if len(Filtered_pharseQuery) == i+1 :
            doc_len_query = math.sqrt(doc_len_query)
        print("tf Weight : ",TfWeight)
        print("IDF : ",idf)
        print("tf*idf : ",TfWeight*idf)
    except KeyError:
        print("\n\nSorry!Your "+str(i+1)+" word"+" Not Found" )

print_red("***********************************")
print("doc length of this query is : ",doc_len_query)
print("normalized of this query : ")
for i in range(0,x):
    try:
        tf= Filtered_pharseQuery.count(Filtered_pharseQuery[i])
        df=counter
        idf=math.log10(10/df)
        TfWeight= 1 + math.log10(tf)
        Tf_idf = TfWeight*idf
        normalizatee =  Tf_idf/doc_len_query
        print(i+1,"- normlizate =" , normalizatee)
    except KeyError:
        print("\n\nSorry!Your "+str(i+1)+" word"+" Not Found" )    
print_red("***********************************")

'''************************* Tables (Task 3) *****************************'''

normalizate = [[0 for c in range(len(file_names))]for r in range(len(alltokens))]
doc_len =[0 for r in range(len(file_names))]
idf = []
tf_idf = [[0 for c in range(len(file_names))]for r in range(len(alltokens))]
tf_weights = [[0 for c in range(len(file_names))]for r in range(len(alltokens))]
word = 0
display = [[0 for c in range(len(file_names))]for r in range(len(alltokens))]


for term in alltokens:
    try:
        index_word= alltokens.index(term)
        sample_pos_idx = pos_index[term]
        file_list = sample_pos_idx[1]
        counter=0
        FFreq = 0
        for fileno, positions in file_list.items():
            FFreq=len(positions)
            counter += FFreq 
            tf_weights[index_word][fileno]= 1 + math.log10(FFreq) 
            display[index_word][fileno]=1
            
        df=counter
        idf.append(math.log10(10/df))
        word +=1
    except KeyError:
        print("\nSorry!Your "+str(i+1)+" word"+" Not Found" )

for i in range(len(file_names)):
    for j in range(len(idf)):
        tf_idf[j][i] = tf_weights[j][i]*idf[j]
        doc_len[i] += tf_idf[j][i]*tf_idf[j][i] 
    doc_len[i] = math.sqrt(doc_len[i])

for i in range(len(file_names)):
    for j in range(len(idf)):
        normalizate[j][i] = tf_idf[j][i]/doc_len[i]

         
print_green("\ndisplay TF table\n")
print('\n'.join([''.join([ '{:4}'.format(item) for item in row]) 
    for row in display ]))

print_green("\nprint idf table\n")
print(idf)

print_green("\nprint TF-weights table\n")
print('\n'.join([''.join(['{:15}'.format(item) for item in row]) 
      for row in tf_weights ]))

print_green("\nprint tf-idf table\n")
print('\n'.join([''.join(['{:}\t'.format(item) for item in row]) 
      for row in tf_idf ]))

print_green("\nprint doc-len\n")
print(doc_len)

print_green("\nprint normalization table\n")
print('\n'.join([''.join(['{:}\t'.format(item) for item in row]) 
      for row in normalizate ]))


'''************************* Similarity *****************************'''

print_red("********************************************************")  

details={}   
sumofproducts = 0    
for word in alltokens:
    for x in normalizate:
        details[word] = x
        normalizate.remove(x)
        break  

for i in range(len(file_names)):
    for word in alltokens:
        if word in Filtered_pharseQuery:
            for z in range(0,len(Filtered_pharseQuery)):
                try:
                    tf= Filtered_pharseQuery.count(Filtered_pharseQuery[z])
                    df=counter
                    idf=math.log10(10/df)
                    TfWeight= 1 + math.log10(tf)
                    Tf_idf = TfWeight*idf
                    normalizatee =  Tf_idf/doc_len_query
                    sumofproduct = details[word][i] * normalizatee
                    sumofproducts = sumofproduct + sumofproducts

                except KeyError:
                    print("\n\nSorry!Your "+str(z+1)+" word"+" Not Found" )   

    print("the similarity of doc",i,",with the query is : " , sumofproducts)                