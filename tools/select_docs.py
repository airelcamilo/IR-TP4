# Koleksi document id yang ada di train_qrels for DPR
file_input = open("qrels-folder-for-dpr/train_qrels.txt")
dpr_train_qrels = file_input.read() 
file_input.close()

qrels = dpr_train_qrels.split()
qrels_clean = set()
for i in range(len(qrels)):
    if i % 3 == 1:
        qrels_clean.add(qrels[i])

doc_ids = list([int(i) for i in qrels_clean])
doc_ids.sort()

# Koleksi document yang ada di train_docs sesuai doc_id di atas
file_input = open("qrels-folder/train_docs.txt")
train_docs = file_input.read() 
file_input.close()

docs = train_docs.split("\n")
docs_map = dict()

for line in docs:
    doc_id = line.split()[0]
    start_content_index = 0

    for i in range(len(line)):
        if (line[i] == " "):
            start_content_index = i+1
            break

    docs_map[int(doc_id)] = line[start_content_index:-1]

# Buat file train_docs.txt baru untuk menampung docs yang ada di doc_ids
new_content = ""
for i in doc_ids:
    new_content += str(i) + " " + docs_map[i] + "\n"

my_file = open('train_docs.txt', mode='w')
print(new_content, file=my_file)
my_file.close()