import pandas as pd
import mysql.connector
import streamlit as st
import ast  # For converting string representations of lists to actual lists

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'dm@lab172',
    'database': 'jobsposting_sample'
}

# Connect to MariaDB
conn = mysql.connector.connect(**db_config)


# SQL query to select data from the table
query1 = "SELECT * FROM Greek_Portals_RT"
query2 = "SELECT * FROM Greek_Skills_RT"
query3 = "SELECT * FROM Greek_Occupation_RT"
query4 = "SELECT * FROM Greek_Industry_RT"
query5 = "SELECT * FROM Greek_EQF_RT"
query6 = "SELECT * FROM Greek_Education_Level_RT"

df1 = pd.read_sql(query1, conn)
df2 = pd.read_sql(query2, conn)
df3 = pd.read_sql(query3, conn)
df4 = pd.read_sql(query4, conn)
df5 = pd.read_sql(query5, conn)
df6 = pd.read_sql(query6, conn)

# Function to safely convert string representations of lists to actual lists
def safe_literal_eval(value):
    if value is None or value == "":
        return []  # Return an empty list for None or empty strings
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []  # Return an empty list for invalid literals

# Convert string representations of lists to actual lists
df1["required_skills"] = df1["required_skills"].apply(safe_literal_eval)
df1["qualifications"] = df1["qualifications"].apply(safe_literal_eval)


# Streamlit App
st.title("Job Postings Metadata Explorer")

# Input for job posting ID
job_id = st.number_input("Enter Job Posting ID", min_value=1, max_value=100, value=1, key="job_id_input")

# Filter job posting by ID
job_posting = df1[df1["id"] == job_id]

if not df1.empty:
    # Display en_title and en_description
    st.subheader("Job Posting Details")
    st.write(f"**Foreign Key ID:** {job_posting['foreign_key_id'].values[0]}")
    st.write(f"**Source:** {job_posting['source'].values[0]}")
    # Display en_title and en_description
    st.write(f"**Title:** {job_posting['en_title'].values[0]}")
    st.write(f"**Description:** {job_posting['en_description'].values[0]}")

    # Button to show metadata fields
    if st.button("Show Metadata Fields"):
        st.subheader("Metadata Fields")
        st.write(f"**Job Title:** {job_posting['job_title'].values[0]}")
        st.write(f"**Required Skills:** {', '.join(job_posting['required_skills'].values[0])}")
        st.write(f"**Qualifications:** {', '.join(job_posting['qualifications'].values[0])}")
        st.write(f"**Industry Category:** {job_posting['industry_category'].values[0]}")
        st.write(f"**Required Education Level:** {job_posting['required_education_level'].values[0]}")
    
    # Button to show normalized skills (ESCO skills)
    if st.button("Show Normalized Skills"):
        # Get foreign_key_id and source for the selected job posting
        foreign_key_id = job_posting["foreign_key_id"].values[0]
        source = job_posting["source"].values[0]

        # Find normalized skills from skills_table
        normalized_skills = df2[
            (df2["foreign_key_id"] == foreign_key_id) &
            (df2["source"] == source)
        ][["foreign_key_id", "source", "required_skill", "skill_esco"]]
        normalized_skills["foreign_key_id"] = normalized_skills["foreign_key_id"].astype(str)
        st.subheader("Normalized Skills (ESCO)")
        st.dataframe(normalized_skills)  # Display as a table

        # Button to show normalized job title (ESCO skills)
    if st.button("Show Normalized Job Title"):
        # Get foreign_key_id and source for the selected job posting
        foreign_key_id = job_posting["foreign_key_id"].values[0]
        source = job_posting["source"].values[0]

        # Find normalized skills from skills_table
        normalized_job_title = df3[
            (df3["foreign_key_id"] == foreign_key_id) &
            (df3["source"] == source)
        ][["foreign_key_id", "source", "job_title", "occupation_title","isco_code","esco_code"]]
        normalized_job_title["foreign_key_id"] = normalized_job_title["foreign_key_id"].astype(str)
        st.subheader("Normalized Job Title (ESCO)")
        st.dataframe(normalized_job_title)  # Display as a table

        # Button to show normalized industry (ESCO skills)
    if st.button("Show Normalized Industry Category"):
        # Get foreign_key_id and source for the selected job posting
        foreign_key_id = job_posting["foreign_key_id"].values[0]
        source = job_posting["source"].values[0]

        # Find normalized skills from skills_table
        normalized_industry = df4[
            (df4["foreign_key_id"] == foreign_key_id) &
            (df4["source"] == source)
        ][["foreign_key_id", "source", "industry_category", "industry_nace"]]
        normalized_industry["foreign_key_id"] = normalized_industry["foreign_key_id"].astype(str)
        st.subheader("Normalized Industry (NACE)")
        st.dataframe(normalized_industry)  # Display as a table

        # Button to show normalized industry (ESCO skills)
    if st.button("Show Normalized Qualification Level based on EQF"):
        # Get foreign_key_id and source for the selected job posting
        foreign_key_id = job_posting["foreign_key_id"].values[0]
        source = job_posting["source"].values[0]

        # Find normalized skills from skills_table
        normalized_eqf = df5[
            (df5["foreign_key_id"] == foreign_key_id) &
            (df5["source"] == source)
        ][["foreign_key_id", "source", "qualification", "eqf_level"]]
        normalized_eqf["foreign_key_id"] = normalized_eqf["foreign_key_id"].astype(str)
        st.subheader("Normalized Qualification (ESCO)")
        st.dataframe(normalized_eqf)  # Display as a table

        # Button to show normalized industry (ESCO skills)
    if st.button("Show Normalized Education Level based on ISCED"):
        # Get foreign_key_id and source for the selected job posting
        foreign_key_id = job_posting["foreign_key_id"].values[0]
        source = job_posting["source"].values[0]

        # Find normalized skills from skills_table
        normalized_education_level = df6[
            (df6["foreign_key_id"] == foreign_key_id) &
            (df6["source"] == source)
        ][["foreign_key_id", "source", "required_education_level", "isced_level"]]
        normalized_education_level["foreign_key_id"] = normalized_education_level["foreign_key_id"].astype(str)
        st.subheader("Normalized Education Level (ISCED)")
        st.dataframe(normalized_education_level)  # Display as a table

        # New Feature: Search for required_skill by ESCO skill value
    st.subheader("Search Required Skills by ESCO Skill")
    esco_skill_input = st.text_input("Enter an ESCO Skill Value (e.g., 'work in an organised manner')")

    if esco_skill_input:
        # Filter the skills table for the input ESCO skill value
        filtered_skills = df2[df2["skill_esco"].str.contains(esco_skill_input, case=False, na=False)]

        if not filtered_skills.empty:
            # Get distinct required_skill values
            distinct_required_skills = filtered_skills["required_skill"].unique()

            # Display the distinct required_skill values
            st.write("**Distinct Required Skills:**")
            st.text_area(
                "Scrollable Required Skills",
                value="\n".join(distinct_required_skills),
                height=200,  # Set the height of the scrollable box
                key="required_skills_box"
            )
        else:
            st.warning("No matching required skills found for the given ESCO skill value.")
else:
    st.warning("No job posting found with the given ID.")