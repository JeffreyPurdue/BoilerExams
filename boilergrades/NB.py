import polars as pl
import datetime as dt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut

df = pl.read_parquet("C:/Users/main/OneDrive/Desktop/study/cs/vscode/boilergrades/subtitles.parquet")

df = df.filter(pl.col("file_name").str.contains("MA 161"))

questionTopic = []
questionText = []
topics = [
    "BasicDerivatives", "Continuity", "Domains", "Graphing", "Inverses", "LimitDefinition", 
    "LimitsExam1", "LogarithmProperties", "TrigProperties"
]

for topic in topics:
    # extract problem data from csvs

    csvFile = f"C:/Users/v3rmi/OneDrive/Desktop/study/cs/vscode/boilergrades/{topic}.csv"
    topicDf = pl.read_csv(csvFile, columns=["Problem", "Start", "End"])

    # Process each problem in the topic's CSV file
    for row in topicDf.iter_rows():
        # CSV format: Problem, Start Timestamp, End Timestamp
        # example: "Question 11 Fall 2018 Exam 1, 2034, 2449"
        problem = row[0]
        startTimestamp = row[1]
        endTimestamp = row[2]

        # search for relevant keywords (exam name, date, etc).
        problemParts = problem.split()
        season = problemParts[2]
        year = int(problemParts[3])
        examId = "Final" if problemParts[4] == "Final" else "Exam " + problemParts[5]

        filteredDf = df.filter(
                        pl.col("file_name").str.contains(f"{season}") &
                        pl.col("file_name").str.contains(f"{year}") &
                        pl.col("file_name").str.contains(f"{examId}") &
                        (pl.col("elapsed_seconds") >= int(startTimestamp)) &
                        (pl.col("elapsed_seconds") <= int(endTimestamp))
                        )

        # insert text transcript from problem and the problem type into the relevant lists
        currentProblemText = " ".join(filteredDf["word"].to_list())
        questionTopic.append(topic)
        questionText.append(currentProblemText)

questions = pl.DataFrame({"questionTopic": questionTopic, "questionText": questionText})

mapping = {topic: idx for idx, topic in enumerate(topics)}

questions = questions.with_columns(
    pl.col("questionTopic").replace_strict(mapping).alias("questionTopicEncoded")
)

XText = questions["questionText"].to_list()
y = np.array(questions["questionTopicEncoded"].to_list())

# leave one out cross validation:

loo = LeaveOneOut()
yTrue = []
yPred = []

model = make_pipeline(TfidfVectorizer(stop_words='english', max_features=243), MultinomialNB())

for trainIndex, testIndex in loo.split(XText):
    # split data into training and test
    XTrain = [XText[i] for i in trainIndex]
    XTest = [XText[i] for i in testIndex]
    yTrain, yTest = y[trainIndex], y[testIndex]

    # model training
    model.fit(XTrain, yTrain)

    # model testing
    yPred.append(model.predict(XTest)[0])
    yTrue.append(yTest[0])

# classification report:
print("Classification Report:")
print(classification_report(yTrue, yPred, target_names=topics))

#print(yTrue)
#print(yPred)

print("Accuracy:", accuracy_score(yTrue, yPred))

# classifier
def classify(text):
    predictedIndex = model.predict([text])[0]
    return topics[predictedIndex]

# example
questionTranscript = "The derivative of this function is 2x." # <-- transcript of question you want classified
prediction = classify(questionTranscript)
print(f"Predicted Topic: {prediction}")