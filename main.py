import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math

from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = open('read.txt', encoding='utf-8').read()
lower_case = text.lower()
print(lower_case)

print(string.punctuation)

cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
print(cleaned_text)

token_words = cleaned_text.split()
print(token_words)

tokenized_word = word_tokenize(cleaned_text, "English")

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

final_words = []
for word in token_words:
    if word not in stop_words:
        final_words.append(word)

for word in tokenized_word:
    if word not in stopwords.words('English'):
        final_words.append(word)
print(final_words)

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        #print("Word :" + word + " " + "Emotion :" + emotion)

        if word in final_words:
            emotion_list.append(emotion)

print(emotion_list)
w = Counter(emotion_list)
print(w)


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    print(score)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        print("Negative Sentiment")
    elif pos > neg:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")


sentiment_analyse(cleaned_text)

#bar-graph
fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()

#line-graph
w = {'Sad': 4.0, 'Happy': 2.0, 'Attached': 1.0, 'Fearful': 1.0, 'loved': 1.0}
fig, ax1 = plt.subplots()
ax1.plot(w.keys(), w.values())
plt.xticks(rotation=45)
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('Line Graph')
plt.savefig('line_graph.png')
plt.show()

#pie-chart
labels = w.keys()
values = w.values()
fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct="%1.1f%%")
ax.axis('equal')
plt.title('Emotion Analysis\n')
plt.savefig('pie_chart.png')
plt.show()


entropy = -sum(p(i) * math.log2(p(i)))
data = {'emotion': emotion_list, 'neg': sentiment_scores['neg'], 'pos': sentiment_scores['pos'], 'neu': sentiment_scores['neu']}
df = pd.DataFrame(data)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df[['neg', 'pos', 'neu']])

cluster_labels = kmeans.labels_

cluster_counts = df['emotion'].value_counts()
total_data = len(df)
probabilities = cluster_counts / total_data

entropy = -sum(probabilities * np.log2(probabilities))
print("Estimated Entropy:", entropy)