import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag, pos_tag_sents
import string

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("tagsets")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")

text = """Nós somos feitos de poeira de estrelas. Nós somos uma maneira de o cosmos se autoconhecer. 
A imaginação nos leva a mundos que nunca sequer existiram. Mas sem ela não vamos a lugar algum."""

sentences = sent_tokenize(text, language="portuguese")
print(sentences)

words = word_tokenize(text, language="portuguese")
print(words)

stops = stopwords.words("Portuguese") #words that have no semantic value
print(stops)

words_with_no_stop_words = [w for w in words if w not in stops]
words_with_no_punctuation = [w for w in words_with_no_stop_words if w not in string.punctuation]

print(words_with_no_punctuation)

word_frequency = nltk.FreqDist(words_with_no_punctuation)
most_common = word_frequency.most_common(2) #2 most comon words
print(most_common)

frequency = []
for w in word_frequency:
    frequency.append(w)
print(frequency) #it excludes repeated words

stemmer = PorterStemmer()
stem1 = [stemmer.stem(word) for word in words_with_no_punctuation]
print(stem1)

stemmer2 = SnowballStemmer(language="portuguese")
stem2 = [stemmer2.stem(word) for word in words_with_no_punctuation]
print(stem2)

stemmer3 = LancasterStemmer()
stem3 = [stemmer3.stem(word) for word in words_with_no_punctuation]
print(stem3)

nltk.help.upenn_tagset() #dictionary

pos_tag = nltk.pos_tag(words_with_no_punctuation)
print(pos_tag)


lemma = WordNetLemmatizer()
result = [lemma.lemmatize(word) for word in words_with_no_stop_words]
print(result)

text_en = "Barack Obama foi um presidente dos EUA."
words_en = word_tokenize(text_en)
pos_tag_en = nltk.pos_tag(words_en)
en = nltk.ne_chunk(pos_tag_en)
print(en)
