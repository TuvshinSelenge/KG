import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
import torch
from data.fetch_country_data import fetch_country_data
from data.fetch_gdelt_events import fetch_gdelt_events, get_edges_for
from geo.get_country_centroids import get_country_centroids
from model.rgcn_embed import RGCN
from torch_geometric.data import HeteroData

colors = {
    "USA": "blue",
    "CHN": "red",
}

if 'show_predictions' not in st.session_state:
    st.session_state['show_predictions'] = False

st.sidebar.header("Settings")

with st.sidebar.expander("⚠️ Time Range Guidelines"):
    st.markdown("""
    - **Too short** (< 1 day): May cause data loading errors
    - **Too long** (> 7 days): Will take longer to process
    - **Recommended**: 1-3 days for optimal performance
    """)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2025-07-15"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-07-16"))
num_records = st.sidebar.slider("Events per Day", 50, 1000, 250, step=50)

if st.sidebar.button("Load Data"):
    keyword = "trade"
    with st.spinner("Loading Country Data..."):
        st.session_state['df_countries'] = fetch_country_data()
    with st.spinner("Loading Geographic Coordinates..."):
        st.session_state['centroids'] = get_country_centroids()
    with st.spinner("Loading GDELT Events..."):
        try:
            ev, mn, _, _ = fetch_gdelt_events(str(start_date), str(end_date), keyword, [], num_records)
            
            if ev is None or mn is None or ev.empty or mn.empty:
                st.error(f"No GDELT data found for the period {start_date} to {end_date}. Please try a different date range.")
                st.stop()
            
            weights = (pd.merge(
                mn[['GLOBALEVENTID', 'MentionIdentifier']],
                ev[['GLOBALEVENTID', 'Actor1Code', 'Actor2Code']],
                on='GLOBALEVENTID', how='inner')
                .groupby(['Actor1Code', 'Actor2Code'])
                .size()
                .reset_index(name='weight')
            )
            
            if weights.empty:
                st.error("No interactions found in GDELT data. Please try a different date range or keyword.")
                st.stop()
                
            st.session_state['weights'] = weights

            actors = list({*weights['Actor1Code'], *weights['Actor2Code']})
            st.session_state['actors'] = actors
            st.session_state['actor2id'] = {code: i for i, code in enumerate(actors)}

            feat_dim = 3
            X = torch.randn(len(actors), feat_dim) 
            edge_index = torch.tensor([
                [st.session_state['actor2id'][a], st.session_state['actor2id'][b]]
                for a, b in zip(weights['Actor1Code'], weights['Actor2Code'])
            ], dtype=torch.long).t()

            data = HeteroData()
            data['actor'].num_nodes = len(actors)
            data['actor', 'interacts', 'actor'].edge_index = edge_index
            data['actor', 'interacts', 'actor'].edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
            data['actor'].x = X
            model = RGCN(X.size(1), 16, 1)
            with torch.no_grad():
                z = model(
                    data['actor'].x,
                    data['actor', 'interacts', 'actor'].edge_index,
                    data['actor', 'interacts', 'actor'].edge_type
                ).cpu()
            st.session_state['z'] = z
            
        except Exception as e:
            st.error(f"Error loading GDELT data: {str(e)}")
            st.info("Possible reasons: \n- No data available for this period\n- Time range too short\n- API connection issues")
            st.stop()

if all(x in st.session_state for x in ['df_countries', 'centroids', 'weights', 'actors', 'actor2id', 'z']):
    df_countries = st.session_state['df_countries']
    centroids = st.session_state['centroids']
    weights = st.session_state['weights']
    actors = st.session_state['actors']
    actor2id = st.session_state['actor2id']
    z = st.session_state['z']

    main_countries_iso3 = st.sidebar.multiselect(
        "Main Countries", 
        options=actors, 
        default=["USA", "CHN"] if "USA" in actors and "CHN" in actors else actors[:2]
    )
    k_pred = st.sidebar.slider("Predictions per Country", 1, 10, 5)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Predictions (Link Prediction)"):
            st.session_state['show_predictions'] = True
    with col2:
        if st.button("Show Only Real Edges"):
            st.session_state['show_predictions'] = False

    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')

    for ctr in main_countries_iso3:
        if ctr in centroids:
            folium.CircleMarker(
                location=centroids[ctr],
                radius=8,
                color=colors.get(ctr, "gray"),
                fill=True,
                fill_color=colors.get(ctr, "gray")
            ).add_to(m)

    if st.session_state['show_predictions']:
        def cosine_similarity(x, y):
            x_norm = x / (x.norm() + 1e-8)
            y_norm = y / (y.norm() + 1e-8)
            return torch.dot(x_norm, y_norm).item()

        preds_list = []
        for ctr in main_countries_iso3:
            if ctr not in actor2id:
                continue
            idx = actor2id[ctr]
            existing = set(weights[weights['Actor1Code'] == ctr]['Actor2Code']).union(
                        set(weights[weights['Actor2Code'] == ctr]['Actor1Code']))
            cands = [
                a for a in actors
                if a not in existing and a != ctr and a in centroids
            ]
            if not cands:
                continue
            cand_ids = [actor2id[a] for a in cands]
            scores = [cosine_similarity(z[idx], z[cid]) for cid in cand_ids]
            topk = sorted(zip(cands, scores), key=lambda x: -x[1])[:k_pred]
            for target, score in topk:
                if target in centroids and ctr in centroids:
                    folium.PolyLine(
                        locations=[centroids[target], centroids[ctr]],
                        color='purple',
                        weight=2,
                        opacity=0.8,
                        dash_array='5,5',
                        tooltip=f"Prediction: {target} ↔ {ctr}, score: {score:.2f}"
                    ).add_to(m)
                    preds_list.append({
                        "Central Country": ctr,
                        "Predicted Country": target,
                        "Score": round(score, 3)
                    })
        folium_static(m)
        st.success("Prediction map generated!")
        if preds_list:
            preds_df = pd.DataFrame(preds_list)
            st.markdown("### Link Prediction Results (Top-K per Country)")
            st.dataframe(preds_df, use_container_width=True)
        else:
            st.info("No predictions available for the selected countries.")

    else:
        for ctr in main_countries_iso3:
            edges = get_edges_for(ctr, weights)
            if edges.empty:
                st.write(f"No edges found for {ctr}")
                continue
            source = edges['source'].iloc[0]
            color = colors.get(ctr, "blue")
            for _, row in edges.iterrows():
                target = row['code']
                if pd.isna(target) or target not in centroids or source == target:
                    continue
                folium.PolyLine(
                    locations=[centroids[target], centroids[source]],
                    color=color,
                    weight=1.0,
                    opacity=0.5,
                    tooltip=f"{target} ↔ {source}: {int(row['weight'])} Events"
                ).add_to(m)
        folium_static(m)
        st.info("Click 'Show Predictions' to view link predictions.")

else:
    st.info("Please select a date range and click 'Load Data' to begin.")