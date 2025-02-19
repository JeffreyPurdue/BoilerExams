import polars as pl
import datetime as dt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import LeaveOneOut


# note: clean up pre-classifier code up later
# somewhat roundabout since it took a while to figure out
# what parquet contained what

#pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

questionsDf = pl.read_parquet("Question.parquet")
questionsToTopicDf = pl.read_parquet("QuestionToTopic.parquet")
topicsDf = pl.read_parquet("Topic.parquet")
courseDf = pl.read_parquet("Course.parquet")

# courseDf Processing
courseDf = courseDf.with_columns(
    pl.concat_str(
        [
            pl.col("abbreviation"),
            pl.col("number")
        ],
        separator=" ",
    ).alias("courseName"),
)
courseDf = courseDf.select("id", "courseName")
courseDf = courseDf.rename({"id": "courseId"})
courseDf = courseDf.with_columns(pl.format("({})", pl.col("courseName")).alias("courseName"))

print(courseDf)
print(questionsDf)


# merge questions + topic df
#print(topicsDf)
questionsDf = questionsDf.rename({"id": "questionId"})
topicsDf = topicsDf.rename({"name": "topicName"})
topicsDf = topicsDf.rename({"id": "topicId"})

finalDf = questionsDf.join(questionsToTopicDf, on="questionId")


# merge final + course df
finalDf = finalDf.join(courseDf, on="courseId", how="left").select(
    [col for col in finalDf.columns] + [pl.col("courseName").alias("courseName")]
)

finalDf = finalDf.with_columns(pl.col("courseName").alias("courseId"))

finalDf = finalDf.drop("courseId")

finalDf = finalDf.join(topicsDf, on="topicId", how="left")
finalDf = finalDf.with_columns(pl.col("topicName").alias("topicId"))
finalDf = finalDf.drop("topicId")


# join topic and course names in final df
finalDf = finalDf.with_columns(
    pl.concat_str(
        [
            pl.col("topicName"),
            pl.col("courseName")
        ],
        separator=" ",
    ).alias("topicName"),
)
finalDf = finalDf.drop('courseName')

#
#
# classifier
#
#

#print(fields)
finalDf = finalDf.with_columns(pl.col("data").struct.field("body").alias("data"))


questionTopic = finalDf["topicName"].to_list()
questionText = finalDf["data"].to_list()
topics = finalDf["topicName"].unique().to_list()

mapping = {topic: i for i, topic in enumerate(topics)}

finalDf = finalDf.with_columns(
    pl.col("topicName").replace_strict(mapping).alias("questionTopicEncoded")
)

# convert text to vector form
vectorizer = TfidfVectorizer(stop_words='english', max_features=350)
X = vectorizer.fit_transform(questionText).toarray()
y = np.array(finalDf["questionTopicEncoded"].to_list())

# split data into training and validation set
trainSize = len(X) - 100  # training size
XTrain, XVal = X[:trainSize], X[trainSize:]
yTrain, yVal = y[:trainSize], y[trainSize:]

# training
model = LogisticRegression()
model.fit(XTrain, yTrain)

# testing on validation set
yPred = model.predict(XVal)

# classification report
print("Classification Report:")
print(classification_report(yVal, yPred, labels=np.unique(yVal), target_names=[topics[i] for i in np.unique(yVal)]))
print("Accuracy:", accuracy_score(yVal, yPred))

# classifier
def classify(text):
    processedText = vectorizer.transform([text]).toarray()
    predictedIndex = model.predict(processedText)[0]
    return topics[predictedIndex]

#example
questionTranscript = "The derivative of this function is 2x."
prediction = classify(questionTranscript)
print(f"Predicted Topic: {prediction}")

#print(finalDf)


'''
# convert text to vector form
vectorizer = TfidfVectorizer(stop_words='english', max_features=135)
X = vectorizer.fit_transform(questionText).toarray()
y = np.array(finalDf["questionTopicEncoded"].to_list())

# leave one out cross validation:

loo = LeaveOneOut()
yTrue = []
yPred = []

model = LogisticRegression()

i = 0
for trainIndex, testIndex in loo.split(X):
    # split data into training and test
    XTrain, XTest = X[trainIndex], X[testIndex]
    yTrain, yTest = y[trainIndex], y[testIndex]

    # model training
    model.fit(XTrain, yTrain)

    # model testing
    yPred.append(model.predict(XTest)[0])
    yTrue.append(yTest[0])

    i = i+1
    print(i)

# classification report:
print("Classification Report:")
print(classification_report(yTrue, yPred, target_names=topics))

print("Accuracy:", accuracy_score(yTrue, yPred))


# classifier
def classify(text):
    processedText = vectorizer.transform([text]).toarray()
    predictedIndex = model.predict(processedText)[0]
    return topics[predictedIndex]

#print(yTrue, len(yTrue))
#print(yPred, len(yPred))

#example:
questionTranscript = "The derivative of this function is 2x." # <-- transcript of question you want classified
prediction = classify(questionTranscript)
print(f"Predicted Topic: {prediction}")

print(finalDf)
'''