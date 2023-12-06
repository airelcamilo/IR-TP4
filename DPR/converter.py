import os
import glob

# Inisiasi documents
documents = []

# Tentukan direktori dasar
file_name = "/content/qrels-folder-for-dpr/train_docs.txt"
file_input = open(file_name)
file_content = file_input.read() 
file_input.close()

docs = file_content.split("\n")

for line in docs:
    doc_id = line.split()[0]
    start_content_index = 0

    for i in range(len(line)):
        if (line[i] == " "):
            start_content_index = i+1
            break

    new_document = {'title': doc_id,
                    'body': line[start_content_index:-1]}
    documents.append(new_document)

print(documents)