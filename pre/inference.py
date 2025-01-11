
import argparse
import pandas as pd 
import pickle

def main(args):
    top_matched_queries = args.corpus_top_matched_queries
    train_queries = pd.read_csv(args.train_queries, sep = "\t", names= ['qid', 'query'])
    with open(args.train_queries_performance, 'rb') as file:
        train_queries_map = pickle.load(file)
    output = args.output 

    index_qid_train_qid = dict()
    cnt = 0
    for row in train_queries.values.tolist():
        index_qid_train_qid[cnt] = row[0]
        cnt += 1

    grps = top_matched_queries.groupby(['corpus_qid'])
    target_dataset = []
    qpp_map = []
    for name, group in grps:
        qid = int(name)
        map_score = -1
        for row in group.values.tolist():
            map_score = train_queries_map[int(index_qid_train_qid[int(row[1])])]
            target_dataset.append([qid, int(index_qid_train_qid[int(row[1])]), map_score])

    target_dataset_df = pd.DataFrame(target_dataset, columns = ['corpus_qid', 'train_qid', 'train_score'])
    average_score_list = []
    grps = target_dataset_df.groupby(['corpus_qid'])
    for name, group in grps:
        average_score_train = group.iloc['train_score'].mean()
        average_score_list.append([int(name), average_score_train])
    
    predicted_socre = pd.DataFrame(average_score_list, columns = ['corpus_qid', 'predicted_score'])
    predicted_socre.to_csv(output + "/predicted_score_per_query", sep = "\t", index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_matched_queries', type=str) #path to top matched queries for target queries
    parser.add_argument('--QuerySpace_queries', type=str) #path to QueryStor queries TSV format
    parser.add_argument('--QuerySpace_queries_performance', type=str) #path to the pickle file containing the MAP@1000 of QueryStor queries
    parser.add_argument('--output', type=str) #path to output
    args = parser.parse_args()

    main()
