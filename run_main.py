import argparse
import pickle
import os
import torch
from torch import optim
import random
import numpy as np
from criterion import Criterion, CrossEntropyCriterion
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, classification_report,accuracy_score
from collections import defaultdict
from data_utils import FewshotACSADataset,build_tokenizer,build_embedding_matrix,Tokenizer,AllAspectFewshotDataLoader, Tokenizer4Bert
from models.induction import FewShotInduction, AspectAwareInduction
from transformers import BertModel
from models.aspect_induction import AspectFewshot
from models.relation import FewShotRelation
from models.aspect_relation import AspectRelation
from models.cnn_relation import CNNRelation
from models.atae_lstm import ATAE_LSTM
from models.bert_spc import  BERT_SPC
from models.bert_aspect import BERT_ASPECT

class Instructor(object):
    def __init__(self,opt):
        self.opt = opt
        self.opt.supportset_size = opt.ways * opt.shots
        # total_tasks = self._get_tasks(os.path.join(opt.dataset_files['data_dir'],"total_tasks"))
        train_tasks = self._get_tasks(opt.dataset_files['train'])
        val_tasks = self._get_tasks(opt.dataset_files['val'])
        test_tasks = self._get_tasks(opt.dataset_files['test'])
        train_fnames = self._get_file_names(opt.dataset_files['data_dir'], train_tasks)
        val_fnames = self._get_file_names(opt.dataset_files['data_dir'], val_tasks)
        test_fnames = self._get_file_names(opt.dataset_files['data_dir'], test_tasks)

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=train_fnames + val_fnames + test_fnames,
                max_seq_len=self.opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=self.opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(300), self.opt.dataset))
            self.model = self.opt.model_class(embedding_matrix, opt).to(self.opt.device)
            print("tokenizer length: ", len(tokenizer.idx2word))
            print("embedding_matrix shape: ", embedding_matrix.shape)
        print("using model: ",opt.model_name)
        print("running dataset: ", opt.dataset)
        print("output_dir: ", opt.output_dir)
        print("shots: ", opt.shots)
        print("ways: ", opt.ways)
        data_dir = self.opt.dataset_files['data_dir']
        self.trainset = FewshotACSADataset(data_dir=data_dir, tasks=train_tasks,
                                      tokenizer=tokenizer,opt = self.opt)
        self.valset = FewshotACSADataset(data_dir=data_dir, tasks=val_tasks,
                                    tokenizer=tokenizer,opt = self.opt)
        self.testset = FewshotACSADataset(data_dir=data_dir, tasks=test_tasks,
                                     tokenizer=tokenizer,opt = self.opt)

        self.optimizer = self.opt.optim_class(self.model.parameters(), lr=self.opt.lr)


    def train(self,episode,criterion):
        self.model.train()
        batch = self.train_loader.get_batch()
        input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
        target = batch['polarity']
        num_support = batch['supportset_size'].item()
        # data, target = train_loader.get_batch()
        target = target.to(self.opt.device)
        self.optimizer.zero_grad()
        predict = self.model(input_features)
        _, loss, acc, f1 = criterion(predict, target,num_support=num_support)
        loss.backward()
        self.optimizer.step()

        writer.add_scalar('train_loss', loss.item(), episode)
        writer.add_scalar('train_acc', acc, episode)
        writer.add_scalar('train_f1', f1, episode)
        if episode % self.opt.log_step == 0:
            print('Train Episode: {} Loss: {} Acc: {} F1: {}'.format(episode, loss.item(), acc, f1))


    def dev_across_tasks(self,episode,criterion):
        tasks = self.valset.tasks
        self.model.eval()
        correct = 0.
        count = 0.
        record_target = defaultdict(list)  # task_id:result
        record_pred = defaultdict(list) # task_id: result
        for i in range(len(self.dev_loader)):
            batch = self.dev_loader.get_batch()
            num_support = batch['supportset_size'].item()
            input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
            # data = batch['text_indices']
            target = batch['polarity']
            task_ids = batch['task_id'].cpu().tolist()
            # data = data.to(self.opt.device)
            target = target.to(self.opt.device)
            predict = self.model(input_features)
            predict, _, _, _ = criterion(predict, target,num_support=num_support)
            true = np.ndarray.tolist(target[num_support:].data.cpu().numpy())
            preds = np.ndarray.tolist(predict.data.cpu().numpy())
            task_ids = task_ids[num_support:]
            assert len(task_ids)==len(preds)==len(true), "Wrong target len!\n"
            for example_index, task_id in enumerate(task_ids):
                record_target[task_id].append(true[example_index])
                record_pred[task_id].append(preds[example_index])
        if self.dev_loader.type !='train':
            self.dev_loader.reset_test_loader() ## dev loader iterration over reset dev loader for next
        score_dict = {}
        total_f1,total_acc = 0,0
        print(tasks)
        print(record_target.keys())
        assert len(tasks)==len(record_target),"dataloader tasks not equal"
        for task_id, targets in record_target.items():
            preds = record_pred[task_id]
            task_f1 = f1_score(targets,preds,average='macro')
            task_acc = accuracy_score(targets,preds)
            score_dict[tasks[task_id]] = {"f1":task_f1,"acc":task_acc}
            total_f1 += task_f1
            total_acc += task_acc
        acc = total_acc / len(record_target)
        f1 = total_f1/len(record_target)
        writer.add_scalar('dev_acc', acc, episode)
        writer.add_scalar('dev_f1',f1,episode)
        print('Dev Episode: {} Acc: {} F1: {}'.format(episode, acc, f1))
        return f1, acc



    def test_across_tasks(self,criterion):
        tasks = self.testset.tasks
        self.model.eval()
        correct = 0.
        count = 0.
        record_target = defaultdict(list)  # task_id:result
        record_pred = defaultdict(list)  # task_id: result
        for idx in range(len(self.test_loader)):
            batch = self.test_loader.get_batch()
            num_support = batch['supportset_size'].item()
            input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
            # data = batch['text_indices']
            target = batch['polarity']
            task_ids = batch['task_id'].cpu().tolist()
            # data = data.to(self.opt.device)
            target = target.to(self.opt.device)
            predict = self.model(input_features)
            predict, _, acc, _ = criterion(predict, target,num_support=num_support)
            true = np.ndarray.tolist(target[num_support:].data.cpu().numpy())
            preds = np.ndarray.tolist(predict.data.cpu().numpy())
            task_ids = task_ids[num_support:]
            assert len(task_ids) == len(preds) == len(true), "Wrong target len!\n"
            for example_index, task_id in enumerate(task_ids):
                record_target[task_id].append(true[example_index])
                record_pred[task_id].append(preds[example_index])
        if self.test_loader.type != 'train':
            self.test_loader.reset_test_loader() # test loader iteration over reset test loader for next test
        score_dict = {}
        total_f1, total_acc = 0, 0
        assert len(tasks) == len(record_target), "dataloader tasks not equal"
        for task_id, targets in record_target.items():
            preds = record_pred[task_id]
            task_f1 = f1_score(targets, preds, average='macro')
            task_acc = accuracy_score(targets, preds)
            cls_report = classification_report(preds,targets,target_names=self.opt.polarities)
            score_dict[tasks[task_id]] = {"f1": task_f1, "acc": task_acc,"report": cls_report}
            total_f1 += task_f1
            total_acc += task_acc
        acc = total_acc / len(record_target)
        f1 = total_f1 / len(record_target)
        print('Test Acc: {} F1: {}\n'.format(acc, f1))
        return f1,acc,score_dict


    def run(self):
        self.train_loader = AllAspectFewshotDataLoader(self.trainset, type='train', support_size=self.opt.shots, query_size=self.opt.query_size, shuffle=True)
        self.dev_loader = AllAspectFewshotDataLoader(self.valset, type='test', support_size=self.opt.shots, query_size=self.opt.query_size, shuffle=False)
        self.test_loader = AllAspectFewshotDataLoader(self.testset, type='test', support_size=self.opt.shots, query_size=self.opt.query_size, shuffle=False)
        print("train loader length: ",len(self.train_loader))
        print("dev loader length: ",len(self.dev_loader))
        print("test loader length: ",len(self.test_loader))
        criterion = self.opt.criterion_class(opt)
        best_episode, best_f1 = 0, 0.
        best_f1_episode, best_acc_episode = 0.,0.
        best_acc = 0.
        episodes = self.opt.num_episode
        print("total episodes: %d" % episodes)
        early_stop = self.opt.patience
        for episode in range(1, episodes + 1):
            self.train(episode,criterion)
            if episode % self.opt.dev_interval == 0:
                f1,acc = self.dev_across_tasks(episode,criterion)
                if f1 > best_f1: ##saving best f1 model
                    print('Better F1! Saving model!')
                    state_dict_dir = opt.output_dir+"/state_dict"
                    if not os.path.exists(state_dict_dir):
                        os.makedirs(state_dict_dir)
                    torch.save(self.model.state_dict(), os.path.join(state_dict_dir,"best_f1_model.bin"))
                    config_path = os.path.join(self.opt.output_dir,"config.bin")
                    with open(config_path,'wb') as out_config:
                        pickle.dump(self.opt,out_config)
                    best_episode, best_f1 = episode, f1
                    best_f1_episode = episode
                if acc > best_acc: ## saving best acc model
                    print('Better ACC! Saving model!')
                    state_dict_dir = opt.output_dir+"/state_dict"
                    if not os.path.exists(state_dict_dir):
                        os.makedirs(state_dict_dir)
                    torch.save(self.model.state_dict(), os.path.join(state_dict_dir,"best_acc_model.bin"))
                    config_path = os.path.join(self.opt.output_dir,"config.bin")
                    with open(config_path,'wb') as out_config:
                        pickle.dump(self.opt,out_config)
                    best_episode, best_acc = episode, acc
                    best_acc_episode = episode
                if episode - best_episode >= early_stop:
                    print('Early stop at episode', episode)
                    break
        print('\n\nReload the best model on episode', best_f1_episode, 'with best f1', best_f1,"from path {}\n\n".format(state_dict_dir))
        ckpt = torch.load(os.path.join(state_dict_dir,"best_f1_model.bin"))
        self.model.load_state_dict(ckpt)
        print(">"*40+"best f1 evaluation"+"<"*40+"\n")
        f1, acc, score_dict = self.test_across_tasks(criterion)
        self.save_evaluation_result(f1,acc,score_dict,"best_f1_result.txt")
        print('\n\nReload the best model on episode', best_acc_episode, 'with best acc', best_acc,
              "from path {}\n\n".format(state_dict_dir))
        print(">"*40+"best acc evaluation"+"<"*40+"\n")
        ckpt = torch.load(os.path.join(state_dict_dir, "best_acc_model.bin"))
        self.model.load_state_dict(ckpt)
        f1,acc,score_dict = self.test_across_tasks(criterion)
        self.save_evaluation_result(f1,acc,score_dict,"best_acc_result.txt")

    def save_evaluation_result(self,f1,acc,score_dict,file_name):
        result_path = os.path.join(self.opt.output_dir, file_name)
        with open(result_path, 'w', encoding='utf-8') as out_file:
            out_file.write('Test Acc: {} F1: {}\nReport:\n'.format(acc, f1))
            for k in sorted(score_dict.keys()):
                scores = score_dict[k]
                for meas_name, value in scores.items():
                    out_file.write("{} {}: {}\n".format(k, meas_name, value))
                out_file.write("\n")

    def _get_tasks(self,task_path):
        tasks = []
        with open(task_path) as file:
            for line in file.readlines():
                line = line.strip()
                tasks.append(line)
        return tasks

    def _get_file_names(self,data_dir,tasks):
        return [ os.path.join(data_dir,task) for task in tasks]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='relation', type=str,required=False)
    parser.add_argument('--dataset', default='fewshot_rest_3way_5', type=str, help='fewshot_rest,fewshot_mams',required=False)
    parser.add_argument('--output_par_dir',default='test_outputs',type=str)
    parser.add_argument('--polarities', default=["positive","neutral","negative"], nargs='+', help="if just two polarity switch to ['positive', 'negtive']",required=False)
    parser.add_argument('--optimizer', default='adam', type=str,required=False)
    parser.add_argument('--criterion', default='origin', type=str,required=False)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str,required=False)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5, 1e-3 for others',required=False)
    parser.add_argument('--dropout', default=0.1, type=float,required=False)
    parser.add_argument('--l2reg', default=0.01, type=float,required=False)
    parser.add_argument('--num_episode', default=5000, type=int, help='try larger number for non-BERT models',required=False)
    parser.add_argument('--log_step', default=10, type=int,required=False)
    parser.add_argument('--dev_interval', default=100, type=int,required=False)
    parser.add_argument('--log_path', default="./log", type=str,required=False)
    parser.add_argument('--embed_dim', default=300, type=int,required=False)
    parser.add_argument('--hidden_dim', default=128, type=int,required=False,help="lstm encoder hidden size")
    parser.add_argument('--feature_dim', default=2*128, type=int,required=False,help="feature dim after encoder depends on encoder")
    parser.add_argument('--output_dim', default=64, type=int,required=False)
    parser.add_argument('--relation_dim',default=100,type=int,required=False)
    parser.add_argument('--bert_dim', default=768, type=int,required=False)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str,required=False)
    parser.add_argument('--max_seq_len', default=85, type=int,required=False)
    parser.add_argument('--shots',default=5,type=int,required=False,help="set 1 shot; 5 shot; 10 shot")
    parser.add_argument('--query_size',default=10,type=int,required=False,help="set 20 for 1-shot; 15 for 5-shot; 10 for 10-shot")
    parser.add_argument('--iterations',default=3,type=int,required=False)
    # parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=3000, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0',required=False)
    parser.add_argument('--seed', default=124, type=int, help='set seed for reproducibility')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # seed
    if opt.seed:
        set_seed(opt.seed)
    model_classes = {
        'induction': FewShotInduction,
        'aspect':AspectFewshot,
        'relation': FewShotRelation,
        'cnn_relation': CNNRelation,
        'aspect_relation': AspectRelation,
        'atae-lstm': ATAE_LSTM,
        'bert-spc': BERT_SPC,
        'aspect_induction': AspectAwareInduction,
        'bert-aspect': BERT_ASPECT,
    }
    input_features = {
        'induction': ['text_indices','supportset_size'],
        'relation': ['text_indices','supportset_size'],
        'cnn_relation': ['text_indices','supportset_size'],
        'aspect': ['text_indices', 'masked_indices', 'seed_indices','supportset_size'],
        'aspect_relation': ['text_indices', 'masked_indices', 'seed_indices','supportset_size'],
        'atae-lstm': ['text_indices', 'aspect_indices','supportset_size'],
        'bert-spc': ['concat_bert_indices', 'concat_segments_indices', 'supportset_size'],
        'aspect_induction': ['text_indices', 'aspect_indices','supportset_size'],
        'bert-aspect': ['concat_bert_indices','masked_indices', 'seed_indices','concat_segments_indices','supportset_size'],
    }
    dataset_files = {
        'fewshot_mams_3way_1': {
            "data_dir": "./datasets/fewshot_mams_3way",
            'train': './tasks/mams/train_tasks',
            'val': "./tasks/mams/val_tasks",
            'test': './tasks/mams/test_tasks',
            'seed_files':['./seed_words/seed_words_list_entities_rest.csv','./seed_words/seed_words_list_attributes_rest.csv'],
        },'fewshot_mams_3way_2': {
            "data_dir": "./datasets/fewshot_mams_3way",
            'train': './tasks/mams/train_tasks_2',
            'val': "./tasks/mams/val_tasks_2",
            'test': './tasks/mams/test_tasks_2',
            'seed_files':['./seed_words/seed_words_list_entities_rest.csv','./seed_words/seed_words_list_attributes_rest.csv'],
        },'fewshot_mams_3way_3': {
            "data_dir": "./datasets/fewshot_mams_3way",
            'train': './tasks/mams/train_tasks_3',
            'val': "./tasks/mams/val_tasks_3",
            'test': './tasks/mams/test_tasks_3',
            'seed_files':['./seed_words/seed_words_list_entities_rest.csv','./seed_words/seed_words_list_attributes_rest.csv'],
        },'fewshot_mams_3way_4': {
            "data_dir": "./datasets/fewshot_mams_3way",
            'train': './tasks/mams/train_tasks_4',
            'val': "./tasks/mams/val_tasks_4",
            'test': './tasks/mams/test_tasks_4',
            'seed_files':['./seed_words/seed_words_list_entities_rest.csv','./seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_14rest_3way_1': {
            "data_dir": "./datasets/fewshot_14rest_3way",
            'train': './tasks/14rest/train_tasks',
            'val': "./tasks/14rest/val_tasks",
            'test': './tasks/14rest/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv','./seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_14rest_3way_2': {
            "data_dir": "./datasets/fewshot_14rest_3way",
            'train': './tasks/14rest/train_tasks_2',
            'val': "./tasks/14rest/val_tasks_2",
            'test': './tasks/14rest/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_14rest_3way_3': {
            "data_dir": "./datasets/fewshot_14rest_3way",
            'train': './tasks/14rest/train_tasks_3',
            'val': "./tasks/14rest/val_tasks_3",
            'test': './tasks/14rest/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_14rest_3way_4': {
            "data_dir": "./datasets/fewshot_14rest_3way",
            'train': './tasks/14rest/train_tasks_4',
            'val': "./tasks/14rest/val_tasks_4",
            'test': './tasks/14rest/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_3way_1': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks',
            'val': "./tasks/rest_3way/val_tasks",
            'test': './tasks/rest_3way/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_3way_2': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks_2',
            'val': "./tasks/rest_3way/val_tasks_2",
            'test': './tasks/rest_3way/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_3way_3': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks_3',
            'val': "./tasks/rest_3way/val_tasks_3",
            'test': './tasks/rest_3way/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_3way_4': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks_4',
            'val': "./tasks/rest_3way/val_tasks_4",
            'test': './tasks/rest_3way/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },'fewshot_rest_3way_5': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks_5',
            'val': "./tasks/rest_3way/val_tasks_5",
            'test': './tasks/rest_3way/test_tasks_5',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_lap_3way_1': {
            "data_dir": "./datasets/fewshot_lap_3way",
            'train': './tasks/lap_3way/train_tasks',
            'val': "./tasks/lap_3way/val_tasks",
            'test': './tasks/lap_3way/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_lap.csv','./seed_words/seed_words_list_attributes_lap.csv'],
        },'fewshot_lap_3way_2': {
            "data_dir": "./datasets/fewshot_lap_3way",
            'train': './tasks/lap_3way/train_tasks_2',
            'val': "./tasks/lap_3way/val_tasks_2",
            'test': './tasks/lap_3way/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_lap.csv','./seed_words/seed_words_list_attributes_lap.csv'],
        },
        'fewshot_lap_3way_3': {
            "data_dir": "./datasets/fewshot_lap_3way",
            'train': './tasks/lap_3way/train_tasks_3',
            'val': "./tasks/lap_3way/val_tasks_3",
            'test': './tasks/lap_3way/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_lap.csv',
                           './seed_words/seed_words_list_attributes_lap.csv'],
        },'fewshot_lap_3way_4': {
            "data_dir": "./datasets/fewshot_lap_3way",
            'train': './tasks/lap_3way/train_tasks_4',
            'val': "./tasks/lap_3way/val_tasks_4",
            'test': './tasks/lap_3way/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_lap.csv',
                           './seed_words/seed_words_list_attributes_lap.csv'],
        },
        'fewshot_mams_2way_1': {
            "data_dir": "./datasets/fewshot_mams_2way",
            'train': './tasks/mams/train_tasks',
            'val': "./tasks/mams/val_tasks",
            'test': './tasks/mams/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_mams_2way_2': {
            "data_dir": "./datasets/fewshot_mams_2way",
            'train': './tasks/mams/train_tasks_2',
            'val': "./tasks/mams/val_tasks_2",
            'test': './tasks/mams/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_mams_2way_3': {
            "data_dir": "./datasets/fewshot_mams_2way",
            'train': './tasks/mams/train_tasks_3',
            'val': "./tasks/mams/val_tasks_3",
            'test': './tasks/mams/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_mams_2way_4': {
            "data_dir": "./datasets/fewshot_mams_2way",
            'train': './tasks/mams/train_tasks_4',
            'val': "./tasks/mams/val_tasks_4",
            'test': './tasks/mams/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },'fewshot_rest_2way_1': {
            "data_dir": "./datasets/fewshot_rest_2way",
            'train': './tasks/rest_2way/train_tasks',
            'val': "./tasks/rest_2way/val_tasks",
            'test': './tasks/rest_2way/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_2way_2': {
            "data_dir": "./datasets/fewshot_rest_2way",
            'train': './tasks/rest_2way/train_tasks_2',
            'val': "./tasks/rest_2way/val_tasks_2",
            'test': './tasks/rest_2way/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_2way_3': {
            "data_dir": "./datasets/fewshot_rest_2way",
            'train': './tasks/rest_2way/train_tasks_3',
            'val': "./tasks/rest_2way/val_tasks_3",
            'test': './tasks/rest_2way/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_2way_4': {
            "data_dir": "./datasets/fewshot_rest_2way",
            'train': './tasks/rest_2way/train_tasks_4',
            'val': "./tasks/rest_2way/val_tasks_4",
            'test': './tasks/rest_2way/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
    }
    optimizers = {
        'adam':optim.Adam,
    }
    criterions = {
        'origin': Criterion,
        'ce':CrossEntropyCriterion,
    }
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_files = dataset_files[opt.dataset]
    opt.optim_class = optimizers[opt.optimizer]
    opt.criterion_class = criterions[opt.criterion]
    opt.input_features = input_features[opt.model_name]
    opt.output_dir = os.path.join(opt.output_par_dir,opt.model_name,opt.dataset) ##get output directory to save results
    opt.ways = len(opt.polarities)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    # writer
    writer = SummaryWriter(opt.log_path)
    ins = Instructor(opt)
    ins.run()
    writer.close()
