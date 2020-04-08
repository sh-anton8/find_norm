import tools.pravoved_recognizer as pravoved_recognizer
import analiz


def read_predictions_from_file():
    predictions = []
    with open("tools/prediction_file.txt", "r") as f:
        data = f.readlines()
        for line in data:
            line = line.replace('(', '').replace(')', '').replace(' ', ''). replace('\'', '').split('\n')[0].split(',')
            predictions.append(((line[1], line[2]), float(line[0])))
    predictions_by_queries = []
    i = 0
    j = 0
    while i + 6322 <= len(predictions):
        predictions_by_queries.append([])
        predictions_by_queries[j].extend(list(sorted(predictions[i:i + 6322], key= lambda x:x[1], reverse=True)))
        j += 1
        i += 6322
    return predictions_by_queries


predictions_by_queries = read_predictions_from_file()
pravoved = pravoved_recognizer.norms_codexes_to_normal("codexes")

analiz.ndcg(pravoved[1251:], predictions_by_queries)
