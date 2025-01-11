import os 
import multiprocessing as mp
from tqdm import tqdm 
import time
import glob
import time
import argparse
import pandas as pd

class QuerySpacePerformanceCalculator:
    def __init__(self, experiment_dir, anserini_path, index_path, hits, chunk_size):
        self.query_files_folder = experiment_dir + '/query_files/'
        self.run_files_folder = experiment_dir + '/run_files/'
        self.retrieval_effectiveness_folder = experiment_dir + '/retrieval_effectiveness/'
        self.qrels_folder = experiment_dir + '/qrels/'
        self.anserini_path = anserini_path
        self.index_path = index_path
        self.hits = hits
        self.queries_chunk_size = chunk_size

    def create_required_directories(self):
        if not os.path.exists(self.query_files_folder):
            os.makedirs(self.query_files_folder)
        if not os.path.exists(self.run_files_folder):
            os.makedirs(self.run_files_folder)
        if not os.path.exists(self.retrieval_effectiveness_folder):
            os.makedirs(self.retrieval_effectiveness_folder)
        if not os.path.exists(self.qrels_folder):
            os.makedirs(self.qrels_folder)

    def parse_queries_file(self, queries_file, qrels_file):
        qrels = pd.read_csv(qrels_file, sep = " ", names = ["qid", "0", "pid", "1"])
        file = open(queries_file, 'r') 
        index = 0
        cnt = 0
        queries = []
        file_lines = file.readlines()
        num_lines = len(file_lines)
        line_cnt = 0
        for line in tqdm(file_lines):
            qid, query = line.split('\t')
            queries.append([qid, query])
            if cnt < self.queries_chunk_size - 1 and line_cnt < num_lines - 1:
                cnt += 1
            else:
                qids = []
                chunk_file = open(self.query_files_folder + "chunk_" + str(index) + ".tsv", 'w')
                for item in queries:
                    qids.append(int(item[0]))
                    chunk_file.write(item[0] + "\t" + item[1])
                chunk_file.close()
                qrels_chunk = qrels[qrels['qid'].isin(qids)]
                qrels_chunk.to_csv(self.qrels_folder + "chunk_" + str(index) + ".trec", sep = " ", index = False, header = None)
                cnt = 0
                index += 1
                queries = []
            line_cnt += 1

    def retrieve(self, queries_file_path):
        chunk_id = queries_file_path.split("/")[-1].split("_")[-1].split(".tsv")[0]
        run_output_tsv_format = self.run_files_folder + "run_file_chunk_" + chunk_id + ".tsv"
        run_output_trec_format = self.run_files_folder + "run_file_chunk_" + chunk_id + ".trec"
        retrieval_effectiveness_output = self.retrieval_effectiveness_folder + "chunk_" + chunk_id + ".txt"
        qrels_path = self.qrels_folder + "chunk_" + chunk_id + ".trec"
        cmd = "python " + self.anserini_path + "/tools/scripts/msmarco/retrieve.py --hits " + str(self.hits) + " --threads 1 " \
            "--index " + self.index_path + " " \
            "--queries " + queries_file_path + " " \
            "--output " + run_output_tsv_format
        os.system(cmd)
        cmd = "python " + self.anserini_path + "/tools/scripts/msmarco/convert_msmarco_to_trec_run.py  --input " + run_output_tsv_format + " --output " + run_output_trec_format
        os.system(cmd)
        os.remove(run_output_tsv_format)
        cmd = self.anserini_path + "/tools/eval/trec_eval.9.0.4/trec_eval -q -mmap " + qrels_path + " " + run_output_trec_format + " > " + retrieval_effectiveness_output
        os.system(cmd)
        # os.remove(run_output_trec_format)

    def retrieve_documents(self, nproc): 
        query_files = glob.glob(self.query_files_folder + "*")
        pool = mp.Pool(nproc)
        pool.map(self.retrieve, list(query_files))
        pool.close()   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-queries', type=str) #path to queries to retrieve (TSV format)
    parser.add_argument('-anserini', type=str) #path to anserini git repo folder
    parser.add_argument('-index', type=str) #path to index folder
    parser.add_argument('-qrels', type=str) #path to qrels
    parser.add_argument('-nproc', type=int) #number of CPU 
    parser.add_argument('-experiment_dir', type=str) #experiment folder
    parser.add_argument('-queries_chunk_size', type=int) #chunk_size to split queries 
    parser.add_argument('-hits', type=int) #number of docs to retrieve
    args = parser.parse_args()

    start_time = time.time()
    document_retriever = QuerySpacePerformanceCalculator(args.experiment_dir, args.anserini, args.index, args.hits, args.queries_chunk_size)
    document_retriever.create_required_directories()
    document_retriever.parse_queries_file(args.queries, args.qrels)
    document_retriever.retrieve_documents(args.nproc)
    print("--- Total Execution Time: %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
