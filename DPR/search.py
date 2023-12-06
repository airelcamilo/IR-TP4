import pandas as pd
from build import dpr

# Membuat fungsi untuk mengonversi hasil pencarian ke dalam dataframe
def results_to_dataframe(results):
    data = {
        'document_title': [],
        'document_body': [],
        'faiss_dist': [],
        'reader_relevance': []
    }

    for result in results:
        data['document_title'].append(result['document']['title'])
        data['document_body'].append(result['document']['body'])
        data['faiss_dist'].append(result['scores']['faiss_dist'])
        data['reader_relevance'].append(result['scores']['reader_relevance'])

    df = pd.DataFrame(data)

    # Normalisasi min-max pada kolom 'reader_relevance'
    df['normalized_reader_relevance'] = (df['reader_relevance'] - df['reader_relevance'].min()) / (df['reader_relevance'].max() - df['reader_relevance'].min())
    
    return df

# Menggunakan fungsi untuk mengonversi hasil pencarian ke dalam dataframe
results = dpr.search('dengan proses apa hormon yang larut dalam lemak mempengaruhi sel target mereka? ')
results_dataframe = results_to_dataframe(results)
results_dataframe