import json, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
import argparse

def main():

    if not torch.cuda.is_available():
        print("Warning: No GPU found. Please add GPU to your notebook")

    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('-model', type=str, default= "sentence-transformers/all-MiniLM-L6-v2") #Name of language model to encode queries with
        parser.add_argument('-queries', type=str) #path to queries we want to index (TSV format)
        parser.add_argument('-output', type=str) #path to output folder
        args = parser.parse_args()

        model_name = args.model
        queries_filepath = args.queries
        output = args.output

        model = SentenceTransformer(model_name)
        embedding_dimension_size = model.get_sentence_embedding_dimension()

        queries=[]
        with open(queries_filepath, 'r', encoding='utf8') as fIn:
            for line in fIn:
                qid, query = line.strip().split("\t")
                qid = int(qid)
                queries.append(query)

        print("Number of Queries to be indexed:", len(queries))

        queries_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True, batch_size=128)
        torch.save(queries_embeddings, output + '/queries_tensor.pt')

        index = faiss.IndexFlatL2(embedding_dimension_size)
        print(index.is_trained)

        all_corpus=torch.load(output + '/queries_tensor.pt', map_location=torch.device('cuda')).detach().cpu().numpy()
        index.add(all_corpus)

        print(index.ntotal)
        faiss.write_index(index, output + '/train_quereis_faiss')

if __name__ == "__main__":
    main()
