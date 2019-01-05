'''
    Collects comments from two different subreddits and uses naive Bayes to classify
    whether a comment is more likely to belong to one or the other
'''

import praw # Python Reddit API

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Establish contact
reddit = praw.Reddit (
    client_id = 'Kb_ngMdlcLrIlA',
    client_secret = 'z-pTnxd0Bj5SYvmn8UbDfrvYPRg',
    user_agent = 'testing'
)

if reddit.read_only: 
    print("Log: contact established")
else:
    print("Log: contact failed")

# Read comments from first subreddit
lp_submission_list = list(reddit.subreddit('learnpython').hot(limit=10)) # Top 10 submissions
print("Log: obtained submissions from /r/learnpython")
lp_submission = lp_submission_list[0] # A submission
print("Log: Top /r/learnpyton submission is '%s'" % lp_submission.title)
lp_comments = lp_submission.comments.list() # Comments from the above submission
print("Log: obtained %d comments from /r/learnpython" %len(lp_comments))

# Read comments from second subreddit
aww_submission_list = list(reddit.subreddit('aww').hot(limit=10))
print("Log: obtained submssions from /r/aww")
aww_submission = aww_submission_list[3]
print("Log:Top /r/aww submission is '%s'" % aww_submission.title)
aww_comments = aww_submission.comments.list()
print("Log: obtained %d comments from /r/aww" %len(aww_comments))

# Create corpus with /r/learnpython as 0 and /r/aww as 1
corpus = [comment.body for comment in (lp_comments[:50] +aww_comments[:50])]
y_train=[0]*len(lp_comments[:50]) + [1] * len(aww_comments[:50])

# Vectorize corpus
vectorizer = CountVectorizer()
vectorizer.fit(corpus)
x_train = vectorizer.transform(corpus)

# Train the naive Bayes model
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

# Prepare a testing data set
test_lp_comments = lp_submission_list[2].comments.list()[:10]
test_aww_comments = aww_submission_list[4].comments.list()[:10]
test_comments = test_lp_comments + test_aww_comments

# Tag testing corpus with those in /r/learnpython as 0 and /r/aww as 1
test_corpus = [comment.body for comment in test_comments]
y_test = [0]*len(test_lp_comments) + [1]*len(test_aww_comments)

# Vectorize testing data
x_test = vectorizer.transform(test_corpus)

# Print results
print(classifier.predict(x_test))
print(y_test)