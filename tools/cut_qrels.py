# Koleksi document id yang ada di train_qrels for DPR
file_input = open("qrels-folder-for-dpr/val_qrels.txt")
dpr_train_qrels = file_input.read() 
file_input.close()

qrels = dpr_train_qrels.split("\n")
counter = 5
new_qrels = ""

for line in qrels:
    query = line.split()[0]

    if (counter == 0 and query == saved_query):
        continue

    if (counter == 0 and query != saved_query):
        counter = 5
        continue

    # Process
    new_qrels += line + "\n"

    saved_query = query    
    counter -= 1

my_file = open('val_qrels.txt', mode='w')
print(new_qrels, file=my_file)
my_file.close()
        
