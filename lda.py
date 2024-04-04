import pandas as pd
from dependencies import description
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from pprint import pprint
from nltk.stem import WordNetLemmatizer

#test

covid = pd.read_csv("Data/metadata_April10_2020.csv")
covid = covid[["title", "abstract", "doi"]]

#Creates an indicator attribute for the formatting of the abstract, makes extracting conclusions much simpler
def hasConclusion(p):
    text = p.split(".")
    for sentence in text:
        if sentence.split(":")[0] == " CONCLUSION" or  sentence.split(":")[0] == " CONCLUSIONS":
            return True
    return False

#Functiion to extract conclusion from abstract, grabbing only the last sentence of abstracts with no designated conclusion
def extractConclusions_1Sentence(row):
    if row["Has Conclusion"] == True:
        return row["abstract"].split("CONCLUSION: ")[-1]
    else:
        return row["abstract"].split(". ")[-1]
    
#Copy of above function except it grabs the last 2 sentences
def extractConclusions_2Sentence(row):
    if row["Has Conclusion"] == True:
        return row["abstract"].split("CONCLUSION: ")[-1]
    else:
        return ". ".join(row["abstract"].split(". ")[-2:])

#Remove unecessary observations with no abstracts and create indicator attribute
covid.dropna(subset=["abstract"], inplace=True)
covid["Has Conclusion"] = covid["abstract"].apply(hasConclusion)

#Perform function to extract both forms of conclusions
covid["conclusion1"] = covid.apply(extractConclusions_1Sentence, axis=1)
covid["conclusion2"] = covid.apply(extractConclusions_2Sentence, axis=1)

#Tokenizing and 
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
"""
#1 Sentence Conclusions
covid["c1"] = covid["conclusion1"].apply(lambda x: [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])
dictionary1 = corpora.Dictionary(covid['c1'])

# Create a corpus
corpus1 = [dictionary1.doc2bow(doc) for doc in covid['c1']]

# Train the LDA model
lda_model1 = gensim.models.LdaModel(corpus1, num_topics=5, id2word=dictionary1, passes=15)

# Print the topics
print("Top words by Topic with 1 Sentence Conclusions")
pprint(lda_model1.print_topics())
"""

###############################################################################
print(description(covid))

#2 Sentence Conclusions
covid["c2"] = covid['conclusion2'].apply(lambda x: [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])
dictionary2 = corpora.Dictionary(covid['c2'])

# Create a corpus
corpus2 = [dictionary2.doc2bow(doc) for doc in covid['c2']]

# Train the LDA model
lda_model2 = gensim.models.LdaModel(corpus2, num_topics=10, id2word=dictionary2, passes=15)

# Print the topics
print("Top words by Topic with 2 Sentence Conclusions")
pprint(lda_model2.print_topics())

topDocs = {0:[0, ""], 1:[0, ""], 2:[0, ""], 3:[0, ""], 4:[0, ""], 5:[0, ""], 6:[0, ""], 7:[0, ""], 8:[0, ""], 9:[0, ""]}
for i in range(covid.size[0]):
    doc_topics = lda_model2.get_document_topics(corpus2[i])
    for tup in doc_topics:
      if tup[1] > topDocs[tup[0]][0]:
        topDocs[tup[0]][0] = tup[1]
        topDocs[tup[0]][1] = covid.iloc[i]["doi"]

print(topDocs)
        

"""
Output of LDA(n=5):

Top words by Topic with 1 Sentence Conclusions
[(0,
  '0.016*"health" + 0.013*"disease" + 0.008*"control" + 0.007*"outbreak" + '
  '0.007*"patient" + 0.007*"public" + 0.007*"risk" + 0.006*"care" + '
  '0.006*"infection" + 0.005*"research"'),
 (1,
  '0.015*"method" + 0.014*"assay" + 0.011*"model" + 0.010*"detection" + '
  '0.010*"result" + 0.009*"diagnostic" + 0.008*"rapid" + 0.008*"tool" + '
  '0.008*"using" + 0.007*"used"'),
 (2,
  '0.028*"cell" + 0.015*"response" + 0.015*"infection" + 0.012*"result" + '
  '0.012*"immune" + 0.011*"may" + 0.010*"suggest" + 0.007*"role" + '
  '0.006*"mouse" + 0.006*"effect"'),
 (3,
  '0.017*"virus" + 0.014*"protein" + 0.012*"viral" + 0.012*"vaccine" + '
  '0.011*"study" + 0.009*"may" + 0.009*"development" + 0.008*"review" + '
  '0.008*"result" + 0.008*"potential"'),
 (4,
  '0.021*"respiratory" + 0.020*"virus" + 0.020*"infection" + 0.017*"patient" + '
  '0.010*"clinical" + 0.008*"conclusion" + 0.008*"child" + 0.008*"study" + '
  '0.008*"associated" + 0.007*"case"')]
Top words by Topic with 2 Sentence Conclusions
[(0,
  '0.024*"patient" + 0.020*"infection" + 0.019*"respiratory" + '
  '0.012*"conclusion" + 0.011*"clinical" + 0.010*"virus" + 0.009*"treatment" + '
  '0.008*"disease" + 0.008*"case" + 0.008*"associated"'),
 (1,
  '0.020*"protein" + 0.012*"virus" + 0.012*"cell" + 0.011*"viral" + '
  '0.008*"rna" + 0.008*"activity" + 0.007*"antiviral" + 0.007*"result" + '
  '0.007*"mechanism" + 0.006*"may"'),
 (2,
  '0.013*"health" + 0.013*"disease" + 0.007*"control" + 0.006*"outbreak" + '
  '0.006*"review" + 0.006*"research" + 0.006*"public" + 0.006*"risk" + '
  '0.005*"system" + 0.005*"infectious"'),
 (3,
  '0.016*"virus" + 0.011*"assay" + 0.010*"method" + 0.009*"result" + '
  '0.009*"detection" + 0.009*"study" + 0.008*"sample" + 0.007*"analysis" + '
  '0.007*"using" + 0.006*"test"'),
 (4,
  '0.023*"cell" + 0.018*"virus" + 0.018*"infection" + 0.015*"vaccine" + '
  '0.014*"response" + 0.013*"antibody" + 0.010*"immune" + 0.010*"mouse" + '
  '0.009*"result" + 0.007*"may"')]

"""

"""

Top words by Topic with 2 Sentence Conclusions
[(0,
  '0.022*"assay" + 0.016*"method" + 0.016*"detection" + 0.013*"result" + '
  '0.012*"sample" + 0.012*"test" + 0.010*"using" + 0.009*"virus" + '
  '0.009*"diagnostic" + 0.008*"sensitivity"'),
 (1,
  '0.017*"transmission" + 0.016*"infection" + 0.012*"calf" + 0.010*"disease" + '
  '0.009*"case" + 0.009*"may" + 0.008*"animal" + 0.007*"outbreak" + '
  '0.007*"number" + 0.007*"virus"'),
 (2,
  '0.045*"respiratory" + 0.038*"virus" + 0.035*"infection" + 0.024*"influenza" '
  '+ 0.016*"child" + 0.014*"viral" + 0.013*"conclusion" + 0.010*"patient" + '
  '0.010*"acute" + 0.010*"severe"'),
 (3,
  '0.046*"cell" + 0.020*"infection" + 0.018*"viral" + 0.017*"virus" + '
  '0.012*"role" + 0.011*"expression" + 0.011*"response" + 0.011*"replication" '
  '+ 0.010*"result" + 0.010*"may"'),
 (4,
  '0.038*"vaccine" + 0.021*"antibody" + 0.014*"antiviral" + 0.013*"response" + '
  '0.012*"activity" + 0.011*"development" + 0.011*"virus" + 0.010*"infection" '
  '+ 0.010*"compound" + 0.009*"drug"'),
 (5,
  '0.047*"strain" + 0.024*"cat" + 0.020*"ibv" + 0.016*"dog" + 0.016*"virus" + '
  '0.015*"material" + 0.013*"chicken" + 0.012*"feline" + 0.010*"supplementary" '
  '+ 0.010*"type"'),
 (6,
  '0.026*"health" + 0.011*"public" + 0.010*"control" + 0.009*"risk" + '
  '0.009*"care" + 0.009*"outbreak" + 0.007*"disease" + 0.007*"conclusion" + '
  '0.007*"measure" + 0.007*"epidemic"'),
 (7,
  '0.018*"disease" + 0.017*"review" + 0.011*"new" + 0.011*"development" + '
  '0.010*"study" + 0.010*"research" + 0.009*"also" + 0.008*"approach" + '
  '0.008*"understanding" + 0.007*"discus"'),
 (8,
  '0.035*"protein" + 0.017*"rna" + 0.016*"virus" + 0.014*"sequence" + '
  '0.011*"gene" + 0.009*"structure" + 0.008*"domain" + 0.007*"region" + '
  '0.007*"genome" + 0.007*"viral"'),
 (9,
  '0.047*"patient" + 0.022*"treatment" + 0.018*"clinical" + 0.013*"disease" + '
  '0.012*"conclusion" + 0.011*"p" + 0.010*"associated" + 0.009*"group" + '
  '0.009*"lung" + 0.008*"may"')]

"""

covid.to_csv("Data/covidFiltered.csv")