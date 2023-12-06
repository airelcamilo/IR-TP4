# Information Retrieval

### Install Dependencies
1.  Create Python env
    ```python -m venv env```
2.  Activate env, for Windows use `env/Scripts/activate.bat`, for MacOS use `source env/bin/activate`
3.  Install Dependencies
    ```pip install -r requirements.txt```

### Dense Passage Retrieval
1.  Install ElasticSearch
    https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html
    ```
    curl -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.1-darwin-x86_64.tar.gz
    curl https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.1-darwin-x86_64.tar.gz.sha512 | shasum -a 512 -c - 
    tar -xzf elasticsearch-8.11.1-darwin-x86_64.tar.gz
    cd elasticsearch-8.11.1/ 
    ```
2.  Run ElasticSearch Host
    ```
    ./bin/elasticsearch
    ```

### Links
- Datasets: https://drive.google.com/drive/folders/1xwLmQdRet3NrIv0KYStrCVn57K-2YVlZ
- Dense Passage Retriever G-Collab: https://colab.research.google.com/drive/1FU38Lp7v36dlEgIuzq4ihpoc5BoromlR?usp=sharing
- Presentation Video: https://youtu.be/mBzVtqFgXOE