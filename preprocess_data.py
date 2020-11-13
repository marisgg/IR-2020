from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import nltk
#from nltk.tokenize import word_tokenize 

"""
Compute TF-IDF, which consists of the following two components:
1. Term frequency: measures the frequency of a word in a document, normalize.
    tf(t,d) = count of t in d / number of words in d
2. Inverse document frequency: measures the informativeness of term t.
    idf(t) = log(N / (df + 1)               (df = occurences of t in documents)

The resulting formula: tf-idf(t,d) = tf(t,d)*log(N/(df+1))

INPUT:		Dictionary, with for each file a sub-dictionary containing the title, abstract, and introduction.
OUTPUT:		
""" 

test_input = {
	'3bmaswia': {'title': 'Infectious diseases: a call for manuscripts in an interdisciplinary era', 'abstract': '', 'introduction': []},
	'iwx9tyj3': {'title': 'Spinal anesthesia for Cesarean delivery in women with COVID-19 infection: questions regarding the cause of hypotension', 'abstract': '', 'introduction': []},
	'vs1zu7z5': {'title': 'Public activities preceding the onset of acute respiratory infection syndromes in adults in England - implications for the use of social distancing to control pandemic respiratory infections.', 'abstract': 'Background: Social distancing measures may reduce the spread of emerging respiratory infections however, there is little empirical data on how exposure to crowded places affects risk of acute respiratory infection. Methods: We used a case-crossover design nested in a community cohort to compare self-reported measures of activities during the week before infection onset and baseline periods. The design eliminates the effect of non-time-varying confounders. Time-varying confounders were addressed by exclusion of illnesses around the Christmas period and seasonal adjustment. Results: 626 participants had paired data from the week before 1005 illnesses and the week before baseline. Each additional day of undertaking the following activities in the prior week was associated with illness onset: Spending more than five minutes in a room with someone (other than a household member) who has a cold (Seasonally adjusted OR 1·15, p=0·003); use of underground trains (1·31, p=0·036); use of supermarkets (1·32, p<0·001); attending a theatre, cinema or concert (1·26, p=0·032); eating out at a café, restaurant or canteen (1·25, p=0·003); and attending parties (1·47, p<0·001). Undertaking the following activities at least once in the previous week was associated with illness onset: using a bus, (aOR 1.48, p=0.049), shopping at small shops (1.9, p<0.002) attending a place of worship (1.81, p=0.005). Conclusions: Exposure to potentially crowded places, public transport and to individuals with a cold increases risk of acquiring circulating acute respiratory infections. This suggests social distancing measures can have an important impact on slowing transmission of emerging respiratory infections.', 'introduction': []},
	'7adncyih': {'title': 'Public health implications of complex emergencies and natural disasters', 'abstract': 'BACKGROUND: During the last decade, conflict or natural disasters have displaced unprecedented numbers of persons. This leads to conditions prone to outbreaks that imperil the health of displaced persons and threaten global health security. Past literature has minimally examined the association of communicable disease outbreaks with complex emergencies (CEs) and natural disasters (NDs). METHODS: To examine this association, we identified CEs and NDs using publicly available datasets from the Center for Research on the Epidemiology of Disasters and United Nations Flash and Consolidated Appeals archive for 2005–2014. We identified outbreaks from World Health Organization archives. We compared findings to identify overlap of outbreaks, including their types (whether or not of a vaccine-preventable disease), and emergency event types (CE, ND, or Both) by country and year using descriptive statistics and measure of association. RESULTS: There were 167 CEs, 912 NDs, 118 events linked to ‘Both’ types of emergencies, and 384 outbreaks. Of CEs, 43% were associated with an outbreak; 24% NDs were associated with an outbreak; and 36% of ‘Both’ types of emergencies were associated with an outbreak. Africa was disproportionately affected, where 67% of total CEs, 67% of ‘Both’ events (CE and ND), and 46% of all outbreaks occurred for the study period. The odds ratio of a vaccine-preventable outbreak occurring in a CE versus an ND was 4.14 (95% confidence limits 1.9, 9.4). CONCLUSIONS: CEs had greater odds of being associated with outbreaks compared with NDs. Moreover, CEs had high odds of a vaccine-preventable disease causing that outbreak. Focusing on better vaccine coverage could reduce CE-associated morbidity and mortality by preventing outbreaks from spreading.', 'introduction': []},
	'hfvcv2dw': {'title': 'Detection of Viral and Bacterial Pathogens in Hospitalized Children With Acute Respiratory Illnesses, Chongqing, 2009–2013', 'abstract': "Acute respiratory infections (ARIs) cause large disease burden each year. The codetection of viral and bacterial pathogens is quite common; however, the significance for clinical severity remains controversial. We aimed to identify viruses and bacteria in hospitalized children with ARI and the impact of mixed detections. Hospitalized children with ARI aged ≤16 were recruited from 2009 to 2013 at the Children's Hospital of Chongqing Medical University, Chongqing, China. Nasopharyngeal aspirates (NPAs) were collected for detection of common respiratory viruses by reverse transcription polymerase chain reaction (RT-PCR) or PCR. Bacteria were isolated from NPAs by routine culture methods. Detection and codetection frequencies and clinical features and severity were compared. Of the 3181 hospitalized children, 2375 (74.7%) were detected with ≥1 virus and 707 (22.2%) with ≥1 bacteria, 901 (28.3%) with ≥2 viruses, 57 (1.8%) with ≥2 bacteria, and 542 (17.0%) with both virus and bacteria. The most frequently detected were Streptococcus pneumoniae, respiratory syncytial virus, parainfluenza virus, and influenza virus. Clinical characteristics were similar among different pathogen infections for older group (≥6 years old), with some significant difference for the younger. Cases with any codetection were more likely to present with fever; those with ≥2 virus detections had higher prevalence of cough; cases with virus and bacteria codetection were more likely to have cough and sputum. No significant difference in the risk of pneumonia, severe pneumonia, and intensive care unit admission were found for any codetection than monodetection. There was a high codetection rate of common respiratory pathogens among hospitalized pediatric ARI cases, with fever as a significant predictor. Cases with codetection showed no significant difference in severity than those with single pathogens.", 'introduction': []}
}

#nltk.download('stopwords')
def filter_stopwords(data):
	stop_words = set(nltk.corpus.stopwords.words('english')) 
	filtered_sentence = [w for w in data if not w in stop_words] 
	return filtered_sentence

def remove_punctuation(data):
	symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
	new_data = []
	for word in data:
		for i in symbols:
			between_word = np.char.replace(word, i, '')
			new_word = np.char.replace(between_word, "'", "")
		new_data.append(new_word)
	return new_data

def remove_single_characters(data):
	new_data = []
	for word in data:
		if len(str(word)) > 1:
			new_data.append(str(word))
	return new_data

#nltk.download('wordnet')
def lemmatize(data):
	lemmatizer = WordNetLemmatizer()
	new_data = []
	for word in data:
		new_data.append(lemmatizer.lemmatize(word))
	return (new_data)

def stem(data):
	new_data = []
	stemmer = PorterStemmer()
	for word in data:
		new_data.append(stemmer.stem(word))
	return(new_data)

def remove_numbers(data):
	new_data = []
	for word in data:
		result = ''.join(i for i in word if not i.isdigit())
		new_data.append(result)
	return new_data

def preprocessing(data):
	"""
	Preprocessing steps help us have a look at the distribution of our data. Some standard steps are:
		* change characters to lowercase letters
		* remove stop words
		* remove punctuation
		* remove apostrophe
		* remove stopwords
		* remove single characters
		* stemming (PorterStemmer())
		* lemmatisation (WordNetLemmatizer)
		* converting numbers (num2word) --> remove punctuation and stopwords again 
			--> maybe remove numbers in general, since we are mostly dealing with medical texts
	"""
	
	# Create new dictionary with processed input
	processed_input = {}
	for docuid, body in data.items():		
		for seg, text in body.items():
			if not text or text=="":
				continue
			split_value = re.split('[.|:|?|!| | - ]', text)
			lower_value = np.char.lower(split_value)
			removed_stopwords = filter_stopwords(lower_value)

			removed_numbers = remove_numbers(removed_stopwords)
			# TODO:  check punctuation removal order
			removed_punctuation = remove_punctuation(removed_numbers)
			removed_single_characters = remove_single_characters(removed_punctuation)

			lemmatized_data = lemmatize(removed_single_characters)
			stemmed_data = stem(lemmatized_data)

			print(stemmed_data)
			


			#processed_input[key2] = 


		#processed_input[key] = value
	# change characters to lowercase letters

	return processed_input

processed_data = preprocessing(test_input)
