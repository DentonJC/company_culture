# Culture match

## About

I collected information about companies and applied machine learning methods to find the one that best suits your company culture. The first part of project just compares your cover letter with companies descriptions on Wikipedia and returns the most relevant ones. You can use naive cosine similarity (it may well be the most reliable tool) or use a model that predicts your corporate culture and compares with the corporate culture of the company according to the [Culture 500](https://sloanreview.mit.edu/culture500/).
Culture 500 is the central dataset for both approaches. 

### Structure
#### wiki.py
This script takes the name of the company and searches for the corresponding Wikipedia page, then fills in the table data/companies.csv.
#### utils.py
Do basic preprocessing, tokenize text, stem words and sanitize output.
#### cosine_similarity.py
Gets features from TfidfVectorizer and looks for the best match by cosine similarity between the corpus of Wikipedia articles about companies and the user's cover letter.
#### culture_extraction.py
I use information about cultural values of companies (here is the [EDA](https://krutsylo.neocities.org/Reports/hackme21/Culture500_EDA.html) of preprocessed dataset) as ground truth for training my own XGBRegressor. My hypothesis is that simple but supervised model can approximate solution of the complex task of the cultural values extraction from text and could be applied to user's cover letter.

Original model works this way: huge number of reviews about companies was classified by more than 90 culture-related topics. Then 9 central cultural values was extracted. 

My model trained to predict this values using company descriptions from Wikipedia. When applied to a user's cover letter, it returns an approximate measurement of the values of the company culture. Then the program searches for companies with similar values. 

## Requirements
- Linux
- Python 3

## Install 
> pip install requirements.txt

## Usage
1. Put you cover letter in data/letter.txt
2. Run cosine_similarity.py for best matches by cosine similarity
3. Run culture_extraction.py for best matches by extracted culture values

## Data sources
**User**: Since we have more control over the data that the user provides, it is required to write a text (cover letter) that most fully reveals person's cultural values.

**Company**:

- [ ] Website
    - [ ] text
    - [ ] images OCR
- [x] Wikipedia
- [ ] Reviews
- [ ] Vacancies
- [ ] Youtube top-10 videos
- [ ] Events sponsored by the company
- [ ] News

## References
- English stopwords: https://countwordsfree.com/stopwords
- Culture 500 dataset: https://sloanreview.mit.edu/culture500/ scrapped with https://github.com/DarianNwankwo/culture500
- Tested on cover letters from: https://www.indeed.com/career-advice/cover-letter-samples
- Descriptions of companies scrapped from Wikipedia

