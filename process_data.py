import csv
import json
import timeit
import os.path

def process_data():
    all_texts = []
    with open('metadata.csv', encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            #print(row)  
        
            # access some metadata
            cord_uid_to_text = {}
            cord_uid = row['cord_uid']
            title = row['title']
            abstract = row['abstract']
            authors = row['authors'].split('; ')

            # access the full text (if available) for Intro
            introduction = []
            if row['pdf_json_files']:
                for json_path in row['pdf_json_files'].split('; '):
                    if os.path.exists(json_path):
                        #print(json_path)
                        with open(json_path) as f_json:
                            full_text_dict = json.load(f_json)
                            
                            # grab introduction section from *some* version of the full text
                            for paragraph_dict in full_text_dict['body_text']:
                                paragraph_text = paragraph_dict['text']
                                section_name = paragraph_dict['section']
                                if 'intro' in section_name.lower():
                                    introduction.append(paragraph_text)

                            # stop searching other copies of full text if already got introduction
                            if introduction:
                                break                  
            
            # save for later usage
            cord_uid_to_text[cord_uid] = {
                'title': title,
                'abstract': abstract,
                'introduction': introduction
            }
    print(cord_uid_to_text)
    return cord_uid_to_text

print(timeit.timeit(process_data, number=1))