import pandas as pd
from gdeltdoc import GdeltDoc, Filters
import gdelt as _gd
import country_converter as coco

def fetch_gdelt_events(start_date, end_date, keyword, countries, num_records=250):
    # Doc-API für Artikel & Timeline
    f = Filters(
        start_date  = start_date,
        end_date    = end_date,
        keyword     = keyword,
        country     = countries,
        num_records = num_records
    )
    gd_doc = GdeltDoc()
    articles = gd_doc.article_search(f)
    timeline = gd_doc.timeline_search("timelinevol", f)

    # Events/Mentions (tagweise)
    gd2 = _gd.gdelt(version=2)
    days = pd.date_range(start_date, end_date, freq='D')
    events_list, mentions_list = [], []

    for day in days:
        s = day.strftime("%Y %m %d")
        e = (day + pd.Timedelta(days=1)).strftime("%Y %m %d")
        ev = gd2.Search([s,e], table='events', coverage=True, output='pandas')
        mn = gd2.Search([s,e], table='mentions', coverage=False, output='pandas')
        events_list.append(ev)
        mentions_list.append(mn)

    events_df = pd.concat(events_list, ignore_index=True)
    mentions_df = pd.concat(mentions_list, ignore_index=True)

    return events_df, mentions_df, articles, timeline


def get_edges_for(code_iso3, weights):
    import country_converter as coco
    cc = coco.CountryConverter()
    # Code ist schon ISO3!
    df1 = weights[weights['Actor1Code'] == code_iso3][['Actor2Code', 'weight']].rename(
        columns={'Actor2Code': 'code'})
    df2 = weights[weights['Actor2Code'] == code_iso3][['Actor1Code', 'weight']].rename(
        columns={'Actor1Code': 'code'})
    edges = pd.concat([df1, df2], ignore_index=True)
    # Mappe Targets (falls es spezielle Entitäten gibt)
    edges['code'] = cc.convert(edges['code'].tolist(), to='ISO3')
    edges['source'] = code_iso3
    return edges


if __name__ == "__main__":
    ev, mn, _, _ = fetch_gdelt_events("2025-04-30", "2025-05-01", "trade", ["US", "CN"])
    print(ev.head(), mn.head())
