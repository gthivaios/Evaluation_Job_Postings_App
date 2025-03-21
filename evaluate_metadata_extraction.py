import pandas as pd
import mysql.connector

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'dm@lab172',
    'database': 'jobsposting_sample'
}

# Connect to MariaDB
conn = mysql.connector.connect(**db_config)


# SQL query to select data from the table
query = "SELECT * FROM Greek_Portals_RT"

# Load the data into a DataFrame
df2 = pd.read_sql(query, conn)
df2['scraped'].dt.date.unique()

# keep the following columns:
# id, foreign_key_id, source, location, scraped,  en_title, en_description, job_title, required_skills, qualifications, industry_category, required_education_level

df = df2[['id', 'foreign_key_id', 'source', 'location', 'scraped', 'en_title', 'en_description', 'job_title', 'required_skills', 'qualifications', 'industry_category', 'required_education_level']]
df.to_csv('./metadata_offline_data.csv', index=False)
# Define the schema for the output
class JPInfo(BaseModel):
    job_title: str = Field(..., description="The official title of the position as described in the job posting. This should clearly represent the specific role and level within the organization, including any relevant specializations or focus areas. Examples include 'Senior Data Scientist,' 'Marketing Manager,' or 'Junior Software Developer specializing in Frontend Development.'")
    required_skills: List[str] = Field(..., description="Abilities, proficiencies, or competencies required to perform a task, including language skills, soft skills (like communication) or technical skills (like programming).")
    qualifications: List[str] = Field(..., description="previous experience, working experience,degrees, certifications, licenses, or formal requirements (like a degree or driving license).")
    required_education_level: str = Field(..., description="State the minimum education requirement (e.g., High School Diploma, Bachelor’s Degree, Master’s Degree)")
    industry_category: str = Field(..., description="The industry or sector the job belongs to.")

def clean_text(text):
    try:
        warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'[•*]', '', text)
        text = re.sub(r'&', 'and', text)
        text = re.sub(r'[^a-zA-Z0-9\s/.-]', '', text)
        text = re.sub(r"(?<=\{|,)\s*'(\w+)':", r'"\1":', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    except Exception as e:
        logging.error(f"Error in clean_text: {e}")
        return text  # Return the original text if cleaning fails
    
def generate_prompt(schema: BaseModel, description: str) -> str:
    try:
        field_prompts = [
            f'  "{field_name}": "{field.description or ""}"'
            for field_name, field in schema.__fields__.items()
        ]
        schema_prompt = "{\n" + ",\n".join(field_prompts) + "\n}"
        
        prompt = (
            "<s>[INST] Please extract the job posting details from the following text and format the response strictly according to the schema provided below. "
            "Return only a JSON object in this structure, without any additional text or explanations:\n\n"
            f"{schema_prompt}\n\n"
            "Strictly return only a JSON object in this structure with accurate data extracted from the job description text. "
            + description + " [/INST]"
        )
        return prompt
    except Exception as e:
        logging.error(f"Error in generate_prompt: {e}")
        return ""

def process_description(description, client, model, temperature, max_retries=3):
    retries = 0

    while retries < max_retries:
        try:
            prompt = generate_prompt(JPInfo, description)
            messages = [
                {"role": "system", "content": "You are a helpful assistant that outputs JSON data for job posting comparisons."},
                {"role": "user", "content": prompt}
            ]

            chat_completion = client.beta.chat.completions.parse(
                model=model,
                temperature=temperature,
                response_format=JPInfo,
                messages=messages
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            response_data = json.loads(response_text)
            metadata = JPInfo(**response_data)
            
            return metadata
            
        except (ValidationError, json.JSONDecodeError) as e:
            retries += 1
        except Exception as e:
            return {'error': 'Unexpected error', 'details': str(e)}

    return {'error': 'Max retries reached'}

def extract_metadata(final_data, client, model, temperature):
    results = []
    
    for index, row in tqdm(final_data.iterrows(), total=len(final_data)):
        try:
            description = row['en_description']
            cleaned_description = clean_text(description)
            result = process_description(cleaned_description, client, model, temperature)
            
            if isinstance(result, JPInfo):
                result_dict = result.dict()
                result_dict['id'] = row['id']
                result_dict['source'] = row['source']
                results.append(result_dict)
            else:
                results.append({'id': row['id'], 'source': row['source'], 'error': result['error']})
                
        except Exception as e:
            logging.error(f"Error processing row {index}: {e}")
            results.append({'id': row['id'], 'source': row['source'], 'error': str(e)})
    
    # Convert extracted metadata to DataFrame
    results_df = pd.DataFrame(results)

    # Drop duplicate columns from final_data before merging
    columns_to_drop = [col for col in results_df.columns if col not in ['id', 'source']]
    final_data = final_data.drop(columns=columns_to_drop, errors='ignore')

    # Merge results_df back to final_data without causing duplicate column issues
    final_data = final_data.merge(results_df, on=['id', 'source'], how='left', suffixes=(None, '_drop'))

    # Remove any columns that got "_drop" suffix
    final_data = final_data[[col for col in final_data.columns if not col.endswith('_drop')]]

    return final_data