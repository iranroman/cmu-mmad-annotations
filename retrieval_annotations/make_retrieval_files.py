import fire
import os
import re
import pandas as pd
import datetime

offset_dict = {
    "S07": 508,
    "S08": 300,
    "S09": 226,
    "S12": 400,
    "S13": 290,
    "S14": 386,
    "S16": 168,
    "S17": 236,
    "S18": 316,
    "S19": 354,
    "S20": 212,
    "S22": 262,
    "S24": 360,
}

Verbs = [
    "none",
    "take",
    "put",
    "open",
    "fill",
    "crack",
    "beat",
    "stir",
    "pour",
    "clean",
    "switch_on",
    "read",
    "spray",
    "close",
    "walk",
    "twist_on",
    "twist_off",
]

Objects = [
    "none",
    "brownie_box",
    "brownie_bag",
    "egg_box",
    "egg",
    "egg_shell",
    "salt",
    "pepper",
    "water    ",
    "oil",
    "pam",
    "cap",
    "knife",
    "fork",
    "fork2",
    "spoon",
    "scissors",
    "cupboard",
    "drawer",
    "fridge",
    "stove",
    "oven",
    "counter",
    "sink",
    "baking_pan",
    "frying_pan",
    "measuring_cup_big",
    "measuring_cup_small",
    "big_bowl",
    "small_bowl",
    "paper_towel",
    "water",
]

Prepositions = [
    "into",
    "from",
    "with",
    "on",
    "to",
]

def load_subject_labels(S, recipe, labels_dir):
    filename = os.path.join(labels_dir,f'{S}_{recipe}','labels.dat')
    labels = pd.read_csv(filename, sep=' ', header=None, names=['start_frame', 'stop_frame', 'label'])
    return labels

def remove_preps(labels, rps):
    return [l.replace(rps[0],'').replace(rps[1],'').replace(rps[2],'').replace(rps[3],'') for l in labels]

def label2narration(labels, rps=['_top_right','_top_left','_bottom_right','_bottom_left']):
    labs = labels['label'].tolist()
    labs = remove_preps(labs, rps)
    narrations = [re.sub(' +', ' ', l.replace('-',' ').replace('_',' ').rstrip()) if l!='none---' else 'unspecified' for l in labs]
    labels['label'] = narrations
    return labels

def get_verb_and_class(labels, rps=['_top_right','_top_left','_bottom_right','_bottom_left']):
    labs = labels['label'].tolist()
    labs = remove_preps(labs, rps)
    verbs = [l.split('-')[0] for l in labs]
    verb_classes = [Verbs.index(v) for v in verbs]
    return verbs, verb_classes

def get_nouns_and_classes(labels, rps=['_top_right','_top_left','_bottom_right','_bottom_left']):
    labs = labels['label'].tolist()
    labs = remove_preps(labs, rps)
    nouns = [l.split('-')[1:] for l in labs]
    nouns = [[n for n in nn if n!=''] for nn in nouns]
    nouns = [n if n!=[] else ['none'] for n in nouns]
    nouns_classes = [[Objects.index(n) for n in nn if n not in Prepositions] for nn in nouns]
    return nouns, nouns_classes

def main(subj_names=list(offset_dict.keys()), recipe='Brownie', labels_dir='../data/'):
    narration_ids = []
    participant_id = []
    video_id = []
    narration_timestamp = []
    start_timestamp = []
    stop_timestamp = []
    start_frame = []
    stop_frame = []
    narration = []
    verb = []
    verb_class = []
    noun = []
    noun_class = []
    all_nouns = []
    all_noun_classes = []
    for S in subj_names:
        labels = load_subject_labels(S, recipe, labels_dir)
        labels['start_frame'] += offset_dict[S]
        labels['stop_frame'] += offset_dict[S]
        verbs, verb_classes = get_verb_and_class(labels)
        nouns, noun_classes = get_nouns_and_classes(labels)
        labels = label2narration(labels)

        narration_ids.extend([f'{S}_{recipe}_{i}' for i in range(len(labels))])
        participant_id.extend([S]*len(labels))
        video_id.extend([f'{S}_{recipe}']*len(labels))
        narration_timestamp.extend([datetime.timedelta(seconds=s/30) for s in labels['start_frame']])
        start_timestamp.extend([datetime.timedelta(seconds=s/30) for s in labels['start_frame']])
        stop_timestamp.extend([datetime.timedelta(seconds=s/30) for s in labels['stop_frame']])
        start_frame.extend(labels['start_frame'].tolist())
        stop_frame.extend(labels['stop_frame'].tolist())
        narration.extend(labels['label'].tolist())
        verb.extend(verbs)
        verb_class.extend(verb_classes)
        noun.extend([n[0] if n!='' else '' for n in nouns])
        noun_class.extend(noun_classes)
        all_nouns.extend(nouns)
        all_noun_classes.extend(noun_classes)

    df = pd.DataFrame(
        {
            'narration_ids'      :     narration_ids,
            'participant_id'     :     participant_id,
            'video_id'           :     video_id,
            'narration_timestamp':     narration_timestamp,
            'start_timestamp'    :     start_timestamp,
            'stop_timestamp'     :     stop_timestamp,
            'start_frame'        :     start_frame,
            'stop_frame'         :     stop_frame,
            'narration'          :     narration,
            'verb'               :     verb,
            'verb_class'         :     verb_class,
            'noun'               :     noun,
            'noun_class'         :     noun_class,
            'all_nouns'          :     all_nouns,
            'all_noun_classes'   :     all_noun_classes,
        }
    )
    
    df = df.sort_values(by=['narration_ids'])
    df.to_csv('cmu_retrieval_test.csv')
    df.to_pickle('cmu_retrieval_test.pkl')

    df_sentence = df.drop_duplicates(subset=['narration'],keep='first')[['narration_ids','narration']]
    df_sentence.to_csv('cmu_retrieval_test_sentence.csv')
    df_sentence.to_pickle('cmu_retrieval_test_sentence.pkl')


if __name__ == "__main__":
    fire.Fire(main)
