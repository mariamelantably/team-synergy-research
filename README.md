# Team Synergy & Team-Idea-Fit Feature Engineering & Scoring
This project aims to determine whether a business will succeed based on various features to do with the founders, particularly focused on their synergy and how well they fit the idea as a team. The random chance of choosing a successful group of founders is approximately 1.9% - this model improves that percentage [insert number] times, by expanding the features with the highest correlation with business success and considering a variety of features to do with past experience, education, and market barriers. 

## Features
### The Dataset
A dataset (founders_cleaned_dataset.csv) of 8000 unsuccessful founders & 800 successful founders, with the following information included:
- *Identification Information*: foudnder_uuid, name, org_name, org_uuid
- *Social Media Links*: twitter_url, cb_url, linkedin_url
- *Social Media JSONs*: linkedin_json, cb_json, clean_cb_profile, clean_linkedin_profile
- *Business Information*: idea, started_on, funds_range, success, company_description
I grouped the dataset by organisation, by using `dataset.groupby("org_name")`.

### LLM based feature engineering 
Uses OpenAI's structured output features, running them on each organisation, producing a dictionary with the following features, and then turning this into a dataframe for each organisation, with the following information included:
- *Prior Working Relationships*: 
    - org_name: the name of the organisation
    - group_size: the number of people in the founder group
    - **shared_companies**: the number of companies worked in by any 2 of the founders
    - **shared_big_companies**: the number of big tech companies worked in by any 2 of the founders (Facebook/Meta, Apple, Amazon, Netflix, Google)
    - **countries_worked_in_together**: the number of countries worked in by any 2 of the founders.
- *Duration of Collaboration*:
    - **worked_together_time**: the average time in years that any two founders worked together.
    - **worked_together_time_big**: the average time in years that any two founders worked together at a big-tech company.
- *Complimentary Skills*:
    - communication_skills: the number of founders indicating that they have good communication skills.
    - emotional_intelligence: the number of founders indicating that they have good emotional intelligence skills.
    - **shared_languages**: the number of languages spoken by at least two of the founders
    - **shared_skills**: the number of skills exhibited by at least 2 of the founders.
    - **endorsements**: the number of times any founder has endorsed another on linkedin. 
    - skill_synergy: the level at which the founders have business and/or technical experience. 
- *Shared Education*:
    - **years_studied_together**: the average number of years any two founders studied together
    - **level_studied_together**: the average level studied together (Bachelors, Masters, or PHD)
    - **shared_projects**: the average number of shared projects during university together
    - **shared_publications**: the number of publications published by at least 2 of the founders. 
- *Required Expertise for the Idea*:
    - prior_experience_scale: on a scale from 1 to 5, how experienced all the founders are in the industry of the business.
    - founders_with_prior_technical_experience: the number of foudners with previous tech experience.
    - founders_with_prior_business_experience: the number of founders with previous business/entrepreunerial experience
    - founders_with_good_network: the number of founders with a good network of individuals in the same industry of the business.
    - founders_with_marketing_experience: the number of founders with previous marketing expertise.
    - founders_with_finance_experience: the number of founders with previous financial experience.
    - founders_with_legal_experience: the number of founders with previous experience with legislation.
    - founders_resilient: the number of founders exhibiting resilience skills, such as conflict resolution. 
- *Market Barriers*:
    - rd_costs: on a scale from 1 to 10, how costly R&D is in this industry.
    - set_up_costs:  on a scale from 1 to 10, how costly business setup is in this industry.
    - brand_loyalty:  on a scale from 1 to 10, how monopolised and loyal to brands are consumers in the industry.
    - legal_barriers:  on a scale from 1 to 10, how difficult it is to establish a business in this industry with respect to legislation, courts etc.
    - capital_requirements: on a scale from 1 to 10, how much start-up capital is needed for this business, including factories, machines, technology etc.
    - years_to_breakeven: average number of years needed to breakeven for a company in this industry.
    - market_saturation: the number of already-established companies with a similar idea
    - companied_that_failed: the number of companies that experienced business failure in the last year in this business.
- *Founders Relevant Experience*:
    - prior_founding_experience: the number of founders who have previously founded a business. 
    - previous_big_tech_experience: the number of founders who have previously worked at a big-tech firm.
    - institution_level: the average level of institutions that the founders studied at, based on the Times Higher Education rankings.
    - average_publications: the average number of publications by founders in the business.
    - average_education: the average education level of the founders on a scale from 1 to 3, where 1 is Bachelors, 2 is Masters, 3 is PHD.
    - highest_education: the highest level of education any of the founders reached. 
    - average_extroversion: the average "extroversion" of the founders' linkedins, i.e. how complete each profile is. 
    - leadership_roles: the average number of leadership roles held by the foundersd
    - years_of_experience: the average number of years of experience for the founders. 
    - years_post_grad: the average number of years post completion of bachelors that the company was founded. 
    - countries_worked_in_average: the average number of countries each founder worked in.
    - average_GPA: the average GPA of the founders.
    - average_awards: the average number of awards won by the founders.
    - success: a boolean indicator variable to indicate whether the business was successful. 
Any collaboration related fields (indicated in bold) were filled by a None for single-founder teams. 
'success' is the target variable for this algorithm. 

### Data Transformation
I transformed the data by cleaning duplicates, using one-hot-encoding to encode categorical data into numerical variables, normalising the numerical data, and filling the None values occupied by the single-founders with a 0, a base value in this case.

I then split my data into training and testing, ensuring to have a sufficient number of successes. I experimented with the percentage of successes to keep in the training sample (See paper for more). 

### Model selection
For this project, I experimented with 5 different models:
- Logistic Regression
- Random Forests
- Gradient Boosting
- SVM 
- Ada Boost
Ada Boost had the highest recall, showing it is likely to be the most precise (See paper for more). 

### Precison & Accuracy Evaluation 
Firstly, I evaluated the Mean Squared Error for the model, to get an idea at how good it is at evaluating both successes and failures. Next, I focused on looking at the percentage of indicated successes who actually are successful, to get the percentage precision. This can be done using 'evaluate_precision' function at the end of the code. I also evaluated the recall, a measure of the ratio of true-positives to false-negatives, by the 'evaluate_recall' function at the end of the code. 

## Installation
Please ensure the following dependencies are installed:
- pandas
- scikitlearn
- pydantic
- openai
- matplotlib

these can be installed by running the following in your term
`pip install depedency`

Additionally, using OpenAI's structured features output requires an OpenAI key, which can be found on your account (), as well as the presence of a sufficient amount of OpenAI credit in the associated account. Please insert your key in the code variable 'api_key'. Please ensure your internet connection is sufficiently strong when running this to prevent delays. 

## Acknowledgements 
I would like to thank the team at Vela Partners for helping guide me through this project as a part of the micro-internship programme with the University of Oxford, and making this amazing opportunity available. 

To read the paper for this project, please see: 

## Contact

Email: mariamantably@gmail.com \n
Instagram: @mariamelantable \n 
Github: @mariamelantably \n
Linkedin: (www.linkedin.com/in/mariam-elantably-ab0559290)
