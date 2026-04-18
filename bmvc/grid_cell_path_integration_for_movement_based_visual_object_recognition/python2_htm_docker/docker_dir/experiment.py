import random
from tqdm import tqdm
import numpy as np
from dataloader import DataLoader, MMDataLoader, PseudoDataLoader
from GcCnn import ControlGCN
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
<<<<<<< HEAD
=======
from collections import namedtuple, defaultdict

>>>>>>> a7d644b5a99aa7d60a3eba94639d5b2ebe808732

class Experiment:
    def __init__(self, dataloader=DataLoader, use_train_in_testing=False, n_modals=1, n_modules=40, n_classes=10, dpc=1, T=25, recommendation=False, num_trainmovement=1, **kwargs):
        self.model = ControlGCN
        self.loader = dataloader
<<<<<<< HEAD
=======
        self.train = use_train_in_testing
        self.n_modals = n_modals
>>>>>>> a7d644b5a99aa7d60a3eba94639d5b2ebe808732
        self.dpc = dpc
        self.T = T
        self.recommendation = recommendation
        self.num_trainmovement = num_trainmovement
        self.config_model = {
            'n_modals': n_modals,
            'n_modules': n_modules,
            'n_classes': n_classes,
            'T': T,
            'recommendation': recommendation,
        }
        self.config_loader = {
            'n_modals': n_modals,
            'data_per_class': dpc,
            'n_classes': n_classes,
        }


    def run(self, seed_=0):
        random.seed(seed_)
        np.random.seed(seed_)
        model = self.model(**self.config_model)
        train_loader = self.loader(train=True, **self.config_loader)
        test_loader = self.loader(train=self.train, **self.config_loader)
        storage = defaultdict(list)

<<<<<<< HEAD
        # train
        for vecs, target in tqdm(train_loader, desc='Train'):
=======
        # --- train ---
        for iter, (vecs, target) in enumerate(tqdm(train_loader, desc='Train')):
>>>>>>> a7d644b5a99aa7d60a3eba94639d5b2ebe808732
            model.learn(vecs, target)
            if iter == 1:
                break
            #break
        
<<<<<<< HEAD
        # train movement
=======
        # --- train movement ---
>>>>>>> a7d644b5a99aa7d60a3eba94639d5b2ebe808732
        if self.recommendation:
            map = {1: 0, 5: 1, 10: 2, 20: 3}
            for i in range(self.num_trainmovement):
                random.seed(seed_ + i)
                np.random.seed(seed_ + i)
<<<<<<< HEAD
                for vecs, target in tqdm(train_loader, desc='Train Movement {}'.format(i+1)):
                    model.learn_movement(vecs, target, map[self.dpc])
                    
=======
                for iter, (vecs, target) in enumerate(tqdm(train_loader, desc='Train Movement {}'.format(i+1))):
                    results = model.learn_movement(vecs, target, map[self.dpc])
                    for k, v in results.items():
                        storage[k].append(v)
                    #break
        
>>>>>>> a7d644b5a99aa7d60a3eba94639d5b2ebe808732

        # eval
        random.seed(seed_)
        np.random.seed(seed_)
<<<<<<< HEAD
        brief_all = np.zeros((len(test_loader), self.T), dtype=np.float16)
        pred_hist = np.zeros(self.T, dtype=float)
        ent_all = np.zeros((len(test_loader), self.T), dtype=np.float16)
        mi_T_all = np.zeros((len(test_loader), self.T), dtype=np.int8)
        pi_T_all = np.zeros((len(test_loader), self.T), dtype=np.int8)
        target_all = np.zeros(len(test_loader), dtype=np.int8)

        mi_rate = 0.
        for iter, (vecs, target) in enumerate(tqdm(test_loader, desc='Eval')):
            dict = model.infer(vecs, target, data_idx=iter)
            brief_all[iter] = dict['brief_seq']
            ent_all[iter] = dict['ent_seq']
            mi_T_all[iter] = dict['mi_T']
            pi_T_all[iter] = dict['pi_T']
            mi_rate += sum(dict['mi_T']) / float(self.T)
            target_all[iter] = target
            if dict['pred_step'] is not None:
                pred_hist[dict['pred_step']] += 1
            #break
        print('brief {}'.format(brief_all.mean(axis=0)))
        #print('accuracy {}'.format(np.cumsum(pred_hist) / float(len(test_loader))))
        #print('modal_force {}'.format(mf))
        print('MI rate {}'.format(mi_rate / len(test_loader)))
        #print('brief mean & var {}, {}'.format(brief_all.mean(), brief_all.var()))
        print('ent mean & var {}, {}'.format(ent_all.mean(), ent_all.var()))

        return brief_all, mi_T_all, pi_T_all, target_all


=======
        pred_hist = np.zeros(self.T, dtype=np.float32)
        for iter, (vecs, target) in enumerate(tqdm(test_loader, desc='Eval')):
            results = model.infer(vecs, target, data_idx=iter)
            pred_t = results['pred_step_Eval']
            if pred_t is not None:
                pred_hist[pred_t] += 1
            for k, v in results.items():
                storage[k].append(v)
            if iter == 1:
                break

        print('brief : {}'.format(np.mean(storage['brief_Eval'], axis=0)))
        print('mi_rate : {}'.format(np.mean(storage['mi_T_Eval'])))
        #print('accuracy {}'.format(np.cumsum(pred_hist) / float(len(test_loader))))

        return storage


    def run_seeds(self, n_seeds=1,):
        results = defaultdict(list)

        for i in range(n_seeds):
            storage = self.run(i)
            for k, v in storage.items():
                results[k].append(v)

        results = {k: np.array(v) for k, v in results.items()}
        return results


>>>>>>> a7d644b5a99aa7d60a3eba94639d5b2ebe808732
def plot_brief(brief, name=''):
    avg = np.mean(brief, axis=1)
    med = np.median(avg, axis=0)
    plt.plot(range(1,25+1), med, marker='.', label='label')
    plt.savefig('results2/week0/brief_avg' + name + '.png')
    plt.clf()


if __name__ == '__main__':
    # setting
    config = {
        'dataloader': PseudoDataLoader0,
        'use_train_in_testing': True,  
        'n_modals': 2,
        'n_modules': 10,
        'n_classes': 10,
        'dpc': 10,
        'T': 25,
        'recommendation': False,
        'num_trainmovement': 1,
        'save': True,
<<<<<<< HEAD
        'n_seed': 1,
=======
        'n_seeds': 1,
>>>>>>> a7d644b5a99aa7d60a3eba94639d5b2ebe808732
    }
    data_size = config['dpc'] * config['n_classes']
    T = config['T']

    # run
    brief_all = np.zeros((config['n_seed'], data_size, T), dtype=np.float16)
    mi_T_all = np.zeros((config['n_seed'], data_size, T), dtype=np.int8)
    pi_T_all = np.zeros((config['n_seed'], data_size, T), dtype=np.int8)
    target_all = np.zeros((config['n_seed'], data_size), dtype=np.int8)
    for i in range(config['n_seed']):
        experiment = Experiment(**config)
        brief, mi_T, pi_T, target = experiment.run(i)
        brief_all[i] = brief
        mi_T_all[i] = mi_T
        pi_T_all[i] = pi_T
        target_all[i] = target
    

    # save
    japan_time = datetime.now(pytz.timezone("Asia/Tokyo"))
    now_ = japan_time.strftime("%m-%d-%H-%M")
    #plot_brief(brief_all, name=now_)
    if config['save']:
<<<<<<< HEAD
        results = {
            'brief': brief_all,
            'mi_T': mi_T_all,
            'pi_T': pi_T_all,
            'target': target_all,
        }
        config.pop('dataloader', None)
        results.update(config)
        np.save("results/week0/" + now_ + ".npy", results)
=======
        save_result(results, config)
    
    # ================================================

    # ================================================
>>>>>>> a7d644b5a99aa7d60a3eba94639d5b2ebe808732
    
