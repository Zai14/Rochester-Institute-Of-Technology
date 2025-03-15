import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fpdf import FPDF
import matplotlib.pyplot as plt
import os

# Load your dataset with the updated file path
file_path = r'C:\Users\zaids\Documents\PY\Rochester Institute Of Technology\Week-2 & Week-3\RIT(Week-2).csv'
df = pd.read_csv(file_path)

# Ensure columns are properly handled
df.columns = df.columns.str.strip()  # Strip any spaces around column names

# Step 1: Display the available countries in the dataset
print("Available Countries in the Dataset:")
available_countries = df['Country'].unique()
print(available_countries)

# Step 2: Allow user to input a specific country (e.g., "India")
selected_country = input("Enter the Country for filtering (from the list above): ")

# Filter the dataframe to include only opportunities for the specified country
filtered_df = df[df['Country'] == selected_country]

if filtered_df.empty:
    print(f"No opportunities found for the country: {selected_country}")
else:
    print(f"Available Opportunity Names in the Dataset for {selected_country}:")
    print(filtered_df['Opportunity Name'].unique())

    # PDF generation function
    def generate_pdf(opportunity_name, recommendations):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(200, 10, txt=f"Recommendations for '{opportunity_name}'", ln=True, align='C')

        # Table headers
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(80, 10, 'Opportunity Name', 1)
        pdf.cell(50, 10, 'Category', 1)
        pdf.cell(50, 10, 'Country', 1)
        pdf.ln()

        # Table content
        pdf.set_font('Arial', '', 12)
        for index, row in recommendations.iterrows():
            pdf.cell(80, 10, row['Opportunity Name'], 1)
            pdf.cell(50, 10, row['Opportunity Category'], 1)
            pdf.cell(50, 10, row['Country'], 1)
            pdf.ln()

        # Save PDF
        pdf_output_path = f"recommendations_{opportunity_name.replace(' ', '_')}.pdf"
        pdf.output(pdf_output_path)
        print(f"PDF saved as {pdf_output_path}")

    # Image generation function
    def generate_image(opportunity_name, recommendations):
        plt.figure(figsize=(10, 6))

        # Bar chart showing recommendations
        plt.barh(recommendations['Opportunity Name'], recommendations['Country'], color='skyblue')
        plt.xlabel('Country')
        plt.title(f"Top Recommendations for '{opportunity_name}'")

        # Save image
        image_output_path = f"recommendations_{opportunity_name.replace(' ', '_')}.png"
        plt.savefig(image_output_path)
        plt.close()
        print(f"Image saved as {image_output_path}")

    # Recommendation System Function
    def recommend_opportunities(opportunity_name, df, similarity_matrix, n_recommendations=5):
        # Get the index of the opportunity that matches the name
        idx = df[df['Opportunity Name'] == opportunity_name].index[0]

        # Get the pairwise similarity scores for the selected opportunity
        sim_scores = list(enumerate(similarity_matrix[idx]))

        # Sort opportunities based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the n most similar opportunities
        sim_scores = sim_scores[1:n_recommendations + 1]

        # Get the corresponding opportunities
        opportunity_indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[opportunity_indices]

        return recommendations

    # Create the TF-IDF matrix from the 'Opportunity Name' column
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Opportunity Name'])

    # Compute the cosine similarity matrix
    similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Ensure the new folder exists at the specified path
    output_dir = r'C:\Users\zaids\Documents\PY\Rochester Institute Of Technology\Week-4\Images & Report'
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Generate recommendations for all opportunities from the specified country
    for opportunity_name in filtered_df['Opportunity Name'].unique():
        try:
            recommendations = recommend_opportunities(opportunity_name, df, similarity_matrix, n_recommendations=5)

            # Output recommendations for each opportunity
            print(f"\nTop recommendations similar to '{opportunity_name}' in {selected_country}:")
            print(recommendations[['Opportunity Name', 'Opportunity Category', 'Country']])

            # Generate PDF and images for the recommendations
            generate_pdf(opportunity_name, recommendations)
            generate_image(opportunity_name, recommendations)

        except IndexError as e:
            print(f"Error processing {opportunity_name}: {e}")
