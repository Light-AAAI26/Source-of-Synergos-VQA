# src/data.py (Refactored Version)
import torch
import json
import random
import numpy as np
from collections import Counter
from pathlib import Path


# =============================================================================
# 1. Data Manager: Handles all raw file loading and preprocessing.
# =============================================================================
class OKVQADataManager:
    def __init__(self, data_dir):
        """
        Manages loading all necessary OK-VQA data files from a base directory.
        All paths are relative to the provided data_dir.
        """
        self.data_dir = Path(data_dir)
        self._load_base_annotations()
        self._load_caption_and_feature_data()
        self._load_similar_question_data()

    def _load_base_annotations(self):
        """Loads question and annotation files for both splits."""
        self.annotations = {
            'train': json.load(open(self.data_dir / "mscoco_train2014_annotations.json", "r")),
            'val': json.load(open(self.data_dir / "mscoco_val2014_annotations.json", "r"))
        }
        self.questions = {
            'train': json.load(open(self.data_dir / "OpenEnded_mscoco_train2014_questions.json", "r")),
            'val': json.load(open(self.data_dir / "OpenEnded_mscoco_val2014_questions.json", "r"))
        }
        # Create a mapping from question_id to question_text for quick lookup
        self.qid_to_question = {
            'train': {str(q['question_id']): q['question'] for q in self.questions['train']['questions']},
            'val': {str(q['question_id']): q['question'] for q in self.questions['val']['questions']}
        }
        # Create a mapping from question_id to list of answers
        self.qid_to_answers = {
            'train': {str(ann['question_id']): [a['answer'] for a in ann['answers']] for ann in
                      self.annotations['train']['annotations']},
            'val': {str(ann['question_id']): [a['answer'] for a in ann['answers']] for ann in
                    self.annotations['val']['annotations']}
        }

    def _load_caption_and_feature_data(self):
        """Loads caption files and visual features."""
        # This demonstrates loading multiple caption sources as in your original code
        self.guided_captions = {
            'train': json.load(
                open(self.data_dir / "okvqa_train2014_caption_llava_patch_id_30_0_9009_10_32.json", "r")),
            'val': json.load(open(self.data_dir / "okvqa_val2014_caption_llava_patch_id_30_0_5046_10_32.json", "r"))
        }
        self.guided_captions_blip2 = {
            'train': json.load(open(self.data_dir / "train2014_caption_data_instructblip_v12.json", "r")),
            'val': json.load(open(self.data_dir / "val2014_caption_data_instructblip_v12.json", "r"))
        }
        self.image_features = {
            'train': np.load(self.data_dir / 'detr_encoded_train2014_dic.npy', allow_pickle=True).item(),
            'val': np.load(self.data_dir / 'detr_encoded_val2014_dic.npy', allow_pickle=True).item()
        }

    def _load_similar_question_data(self):
        """Loads data required for building in-context examples."""
        self.similar_qids = {
            'train': json.load(open(self.data_dir / "similar_train_qids.json", "r")),
            'val': json.load(open(self.data_dir / "similar_val_qids.json", "r"))
        }
        # This part of your logic seemed to combine two caption sources for the in-context examples
        train_captions_llava = json.load(
            open(self.data_dir / "okvqa_train2014_caption_llava_patch_id_30_0_9009_10_32.json", "r"))
        train_captions_blip2 = json.load(open(self.data_dir / "train2014_caption_data_instructblip_v12.json", "r"))
        self.in_context_captions = {}
        for key in train_captions_llava:
            id = str(key.split('<->')[1])
            self.in_context_captions[id] = train_captions_llava[key][:80] + train_captions_blip2[key][:40]

    def _build_in_context_passages(self, question_id, guided_captions, split, num_shots=5):
        """
        This function encapsulates your unique in-context example formatting logic.
        """
        line_prefix = "===\n"
        in_context_passages = []

        # The main captions for the current question
        for current_caption in guided_captions:
            prompt_text = ""
            # Get similar questions for few-shot examples
            similar_ids = self.similar_qids[split].get(str(question_id), [])
            if split == 'train':  # Your original code had this logic for training set
                similar_ids = similar_ids[1:]

            random.shuffle(similar_ids)

            # Build the few-shot part of the prompt
            for sim_qid in similar_ids[:num_shots]:
                sim_qid = str(sim_qid)
                # This assumes the first caption of the list is used for in-context examples
                context_caption = self.in_context_captions.get(sim_qid, [""])[0]
                question_text = self.qid_to_question['train'][sim_qid]
                answer_text = random.choice(self.qid_to_answers['train'][sim_qid])

                prompt_text += line_prefix + f'Context: {context_caption}\n'
                prompt_text += line_prefix + f'Question: {question_text}\n'
                prompt_text += line_prefix + f'Answer: {answer_text}\n\n'

            # Add the current example's context and question
            prompt_text += line_prefix + f'Context: {current_caption}\n'
            prompt_text += line_prefix + f'Question: {self.qid_to_question[split][str(question_id)]}\n'
            prompt_text += line_prefix + f'Answer: \n'

            in_context_passages.append(prompt_text)
        return in_context_passages

    def load_split_data(self, split='val'):
        """
        Loads and processes data for a specific split ('train' or 'val').
        This replaces your original `load_data` function.
        """
        processed_examples = []

        # Iterate through the questions for the target split
        for q_data in self.questions[split]['questions']:
            image_id = q_data['image_id']
            question_id = q_data['question_id']
            question_text = q_data['question']

            key = f"{image_id}<->{question_id}"

            # Combine captions from two sources, as in your original code
            guided_captions = self.guided_captions[split].get(key, [])[:80] + self.guided_captions_blip2[split].get(key,
                                                                                                                    [])[
                                                                              :40]

            # Build the complex in-context passages
            passages = self._build_in_context_passages(question_id, guided_captions, split)

            processed_examples.append({
                'id': f"{split}_{question_id}",
                'question': question_text,
                'answers_list': self.qid_to_answers[split][str(question_id)],
                'image_id': image_id,
                'vis_feat': self.image_features[split][image_id].squeeze(0),
                'passages': passages
            })
        return processed_examples


# =============================================================================
# 2. Dataset and Collator: These are now much simpler.
# =============================================================================
class VLT5Dataset(torch.utils.data.Dataset):
    def __init__(self, data, opt):
        self.data = data
        self.opt = opt

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        """Your original get_target logic, preserved."""
        eleCounts = Counter(example['answers_list'])
        top_one = eleCounts.most_common()
        true_answer = [s[0] for s in top_one if s[1] >= 3]

        if 'target' in example:
            return example['target'] + ' </s>'
        elif true_answer:
            return random.choice(true_answer) + ' </s>'
        else:
            return random.choice(example['answers_list']) + ' </s>'

    def __getitem__(self, index):
        example = self.data[index]
        question = "question: " + example['question']
        target = self.get_target(example)

        return {
            'index': index,
            'question': question,
            'target': target,
            'vis_feat': example["vis_feat"],
            'passages': example["passages"]
        }


def encode_passages(batch_text_passages, tokenizer, max_length):
    """Your original passage encoder, preserved."""
    passage_ids, passage_masks = [], []
    for text_passages in batch_text_passages:
        p = tokenizer(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'].unsqueeze(0))
        passage_masks.append(p['attention_mask'].unsqueeze(0))

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        """Your original collator logic, preserved with minor cleanup."""
        index = torch.tensor([ex['index'] for ex in batch])

        # Target processing
        target = [ex['target'] for ex in batch]
        target_encoding = self.tokenizer(
            target,
            max_length=self.answer_maxlength,
            padding='longest',
            return_tensors='pt',
            truncation=True,
        )
        target_ids = target_encoding["input_ids"]
        target_mask = target_encoding["attention_mask"].bool()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        # Visual features
        vis_feats = torch.from_numpy(np.array([ex['vis_feat'] for ex in batch]))

        # Passage processing
        text_passages = [ex['passages'] if ex['passages'] else [ex['question']] for ex in batch]
        passage_ids, passage_masks = encode_passages(text_passages, self.tokenizer, self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks, vis_feats)