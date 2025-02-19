import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pathlib
pathlib.Path(__file__).parent.resolve()


df = pl.read_parquet("C:/Users/main/OneDrive/Desktop/study/cs/vscode/boilergrades/subtitles.parquet")
df = df.filter(pl.col("file_name").str.contains("MA 161") & pl.col("file_name").str.contains("Exam 1"))

# filter words with less than 6 uses from data
useCount = df.group_by("word").agg(pl.n_unique("file_name").alias("video_count"))
filteredDf = useCount.filter(pl.col("video_count") >= 6)

df = df.join(filteredDf, on="word", how="inner")
df = df.drop('video_count')


# split database into 5 minute bins where each bin corresponds to a question number
# '//' corresponds to division + floor
df = df.with_columns(
    (pl.col("elapsed_seconds") // 300 + 1).alias("bin")  
)

# count word frequency per time bin
binFreq = df.group_by(["bin", "word"]).agg(pl.len().alias("word_count"))
totalFreq = df.group_by("word").agg(pl.len().alias("total_word_count"))

# merge total frequency and bin frequency, then calculate their ratio
ratioDf = binFreq.join(totalFreq, on="word")

ratioDf = ratioDf.with_columns(
    (pl.col("word_count") / pl.col("total_word_count")).alias("bin_total_ratio")
)

print(ratioDf)

# find all words that are concentrated around a specific timeframe / bin
highRatioDf = ratioDf.filter(
    pl.col("bin_total_ratio") > 0.5
)

print(highRatioDf)

minMaxBins = ratioDf.group_by("word").agg(
    pl.col("bin_total_ratio").max().alias("max_bin_ratio"),
    pl.col("bin_total_ratio").min().alias("min_bin_ratio"),
)

highRatioDf = highRatioDf.select(["word", "bin", "bin_total_ratio"])

# create column 'word (bin)' which will be used for graph x-axis
highRatioDf = highRatioDf.with_columns(
    ( pl.col("word") + " (Question " + pl.col("bin").cast(pl.Int64).cast(pl.Utf8) + ")" ).alias("word_bin")
)

print(highRatioDf)

# graph
graphDf = highRatioDf

plt.figure(figsize=(10, 6))
plt.bar(graphDf['word_bin'], graphDf['bin_total_ratio'])
plt.xlabel('Word (Question number)')
plt.ylabel('Cluster rate')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
plt.savefig('figure.png')