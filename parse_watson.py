import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EmotionOptions
import csv
import re

natural_language_understanding = NaturalLanguageUnderstandingV1(
	version='2018-11-16',
	iam_apikey='Qi2ddC8XJZi33XHUS1zqU7fejqhXUFwnUZrFWUybTucT',
	url='https://gateway.watsonplatform.net/natural-language-understanding/api'
)

meyersToNumbersDict = {
    "ISTJ": 0,
    "INTP": 1,
    "ISFJ": 2,
    "INFJ": 3,
    "ISTP": 4,
    "ISFP": 5,
    "INFP": 6,
    "INTJ": 7,
    "ESTP": 8,
    "ESTJ": 9,
    "ESFJ": 10,
    "ENFJ": 11,
    "ESFP": 12,
    "ENTJ": 13,
    "ENTP": 14,
    "ENFP": 15
}


with open('data/mbti_5_1_emotion.csv', mode='w') as write_file:
	employee_writer = csv.writer(write_file, delimiter=',')
	employee_writer.writerow(['mbti', 'text', 'sadness', 'joy', 'fear', 'disgust', 'anger'])
	with open('data/mbti_5_1.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			response = natural_language_understanding.analyze(
				text=row[1],
				features=Features(emotion=EmotionOptions())).get_result()
			y_label = meyersToNumbersDict[row[0]]
			emotion_dict = response["emotion"]["document"]["emotion"]
			sadness = emotion_dict['sadness']
			joy = emotion_dict['joy']
			fear = emotion_dict['fear']
			disgust = emotion_dict['disgust']
			anger = emotion_dict['anger']
			instanceNoBars = row[1].replace("|||", " ").replace("\"", "'")
			instanceNoUrls = re.sub(r'http\S+', "", instanceNoBars)
			instanceNoMeyers = re.sub(r'([E,I,e,i][N,S,n,s][F,T,f,t][P,J,p,j])', "abcd", instanceNoUrls)
			employee_writer.writerow([y_label, instanceNoMeyers, sadness, joy, fear, disgust, anger])
