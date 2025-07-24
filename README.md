US-China Decoupling Knowledge Graph
Analyzes geopolitical relationships using GDELT data and Graph Neural Networks.
Quick Start

Install dependencies:

bashpip install streamlit pandas torch torch-geometric folium streamlit-folium gdeltdoc

Run the app:

bashstreamlit run app/app.py

Use the app:

Select date range (1-3 days works best)
Click "Load Data"
View relationships and predictions on the map



Project Structure
app/app.py                    # Main application
data/fetch_gdelt_events.py    # GDELT data fetching
model/rgcn_embed.py          # Graph Neural Network
geo/get_country_centroids.py  # Country coordinates
Troubleshooting
GDELT data not loading?

Try dates from 3-7 days ago
Use shorter date ranges (1-3 days)
Check internet connection

Features

Real-time country relationship visualization
Link prediction using RGCN
Interactive map interface
Focus on US-China relations

Author
Tuvshin Selenge
