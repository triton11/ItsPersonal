import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EmotionOptions
import csv

natural_language_understanding = NaturalLanguageUnderstandingV1(
	version='2018-11-16',
	iam_apikey='Qi2ddC8XJZi33XHUS1zqU7fejqhXUFwnUZrFWUybTucT',
	url='https://gateway.watsonplatform.net/natural-language-understanding/api'
)


with open('mbti_w_emotion.csv', mode='w') as write_file:
	employee_writer = csv.writer(write_file, delimiter=',')
	employee_writer.writerow(['mbti', 'text', 'sadness', 'joy', 'fear', 'disgust', 'anger'])
	with open('mbti_test.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			response = natural_language_understanding.analyze(
				text=row[1],
				features=Features(emotion=EmotionOptions())).get_result()
			emotion_dict = response["emotion"]["document"]["emotion"]
			sadness = emotion_dict['sadness']
			joy = emotion_dict['joy']
			fear = emotion_dict['fear']
			disgust = emotion_dict['disgust']
			anger = emotion_dict['anger']
			employee_writer.writerow([row[0], row[1], sadness, joy, fear, disgust, anger])