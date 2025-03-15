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

# List available opportunity names and countries in the dataset
print("Available Opportunity Names in the Dataset:")
print(df['Opportunity Name'].unique())
print("\nAvailable Countries in the Dataset:")
print(df['Country'].unique())

# Allow user input for opportunity name and country filter
selected_opportunity = input("Enter the Opportunity Name (from the available options): ")
selected_country = input("Enter the Country for filtering (or leave blank for no filter): ")

# Recommendation System Function
def recommend_opportunities(opportunity_name, df, similarity_matrix, n_recommendations=5, country=None):
    # Get the index of the opportunity that matches the name
    if opportunity_name not in df['Opportunity Name'].values:
        raise ValueError(f"Error: '{opportunity_name}' not found in the dataset.")
    
    idx = df[df['Opportunity Name'] == opportunity_name].index[0]
    
    # Get the pairwise similarity scores for the selected opportunity
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort opportunities based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the n most similar opportunities
    sim_scores = sim_scores[1:n_recommendations+1]
    
    # Get the corresponding opportunities
    opportunity_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[opportunity_indices]
    
    # Filter recommendations by country if provided
    if country:
        recommendations = recommendations[recommendations['Country'] == country]
    
    return recommendations

# Create the TF-IDF matrix from the 'Opportunity Name' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Opportunity Name'])

# Compute the cosine similarity matrix
similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Ensure the new folder exists at the specified path
output_dir = r'C:\Users\zaids\Documents\PY\Rochester Institute Of Technology\Week-4\Images & Report(Recommendation-Dynamic)'
os.makedirs(output_dir, exist_ok=True)

# Generate recommendations
try:
    recommendations = recommend_opportunities(selected_opportunity, df, similarity_matrix, n_recommendations=5, country=selected_country)

    # Output recommendations
    print(f"\nTop recommendations similar to '{selected_opportunity}' filtered by country '{selected_country}':")
    print(recommendations[['Opportunity Name', 'Opportunity Category', 'Country']])

    # Generate PDF report for recommendations
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Recommendations for: {selected_opportunity}", ln=True, align='C')

    for i, row in recommendations.iterrows():
        pdf.cell(200, 10, txt=f"{i+1}. {row['Opportunity Name']} - {row['Opportunity Category']} ({row['Country']})", ln=True)

    output_pdf_path = os.path.join(output_dir, f"Recommendation_Report_{selected_opportunity.replace(' ', '_')}.pdf")
    pdf.output(output_pdf_path)
    print(f"\nPDF report saved at: {output_pdf_path}")

    # Generate and save images of recommendations
    plt.figure(figsize=(10, 5))
    plt.barh(recommendations['Opportunity Name'], range(len(recommendations)), color='skyblue')
    plt.xlabel('Similarity Rank')
    plt.ylabel('Opportunity Name')
    plt.title(f"Top {len(recommendations)} Recommended Opportunities (Filtered by Country: {selected_country})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_img_path = os.path.join(output_dir, f"Recommendation_Image_{selected_opportunity.replace(' ', '_')}.png")
    plt.savefig(output_img_path)
    print(f"Recommendation image saved at: {output_img_path}")

except ValueError as e:
    print(e)
