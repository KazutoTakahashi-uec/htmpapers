import random
from tqdm import tqdm
import numpy as np
from dataloader import DataLoader, MMDataLoader, PseudoDataLoader, PseudoDataLoader0, PseudoDataLoader1
from GcCnn import ControlGCN
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from collections import namedtuple


class Experiment:
    def __init__(self, dataloader=DataLoader, n_modals=1, n_modules=40, n_classes=10, dpc=1, T=25, recommendation=False, num_trainmovement=1, **kwargs):
        self.model = ControlGCN
        self.loader = dataloader
        self.n_modals = n_modals
        self.dpc = dpc
        self.T = T
        self.recommendation = recommendation
        self.num_trainmovement = num_trainmovement
        self.datasize = dpc * n_classes
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
        test_loader = self.loader(train=False, **self.config_loader)

        # --- train ---
        for vecs, target in tqdm(train_loader, desc='Train'):
            model.learn(vecs, target)
            #break
        
        # --- train movement ---
        LM_storage = {
            'mi_T_LM': [],
            'pi_T_LM': [],
            'target_LM': [],
            'ent_LM': [],
            'ss_LM': [],
        }
        if self.recommendation:
            map = {1: 0, 5: 1, 10: 2, 20: 3}
            for i in range(self.num_trainmovement):
                random.seed(seed_ + i)
                np.random.seed(seed_ + i)
                for iter, (vecs, target) in enumerate(tqdm(train_loader, desc='Train Movement {}'.format(i+1))):
                    results = model.learn_movement(vecs, target, map[self.dpc])
                    for key in LM_storage:
                        LM_storage[key].append(results[key])
                    #break
        for key in LM_storage:
            LM_storage[key] = np.array(LM_storage[key])
        

        # --- eval ---
        random.seed(seed_)
        np.random.seed(seed_)
        Eval_storage = {
            'brief_Eval': [],
            'ent_Eval': [],
            'pred_step_Eval': [],
            'mf_Eval': [],
            'mi_T_Eval': [],
            'pi_T_Eval': [],
            'target_Eval': [],
        }
        for iter, (vecs, target) in enumerate(tqdm(test_loader, desc='Eval')):
            results = model.infer(vecs, target, data_idx=iter)
            for key in Eval_storage:
                Eval_storage[key].append(results[key])
        for key in Eval_storage:
            Eval_storage[key] = np.array(Eval_storage[key])

        return Eval_storage, LM_storage


    def run_seeds(self, n_seeds=1,):
        results = {
            'mi_T_LM': [],
            'pi_T_LM': [],
            'target_LM': [],
            'ent_LM': [],
            'ss_LM': [],
            'brief_Eval': [],
            'ent_Eval': [],
            'pred_step_Eval': [],
            'mf_Eval': [],
            'mi_T_Eval': [],
            'pi_T_Eval': [],
            'target_Eval': [],
        }

        for i in range(n_seeds):
            Eval_storage, LM_storage = self.run(i)
            Eval_storage.update(LM_storage)
            for key in Eval_storage:
                results[key].append(Eval_storage[key])

        for key in results:
            results[key] = np.array(results[key])
        return results

def plot_brief(brief, name=''):
    avg = np.mean(brief, axis=1)
    med = np.median(avg, axis=0)
    plt.plot(range(1,25+1), med, marker='.', label='label')
    plt.savefig('results2/week0/brief_avg' + name + '.png')
    plt.clf()


def save_result(results={}, config={}):
    japan_time = datetime.now(pytz.timezone("Asia/Tokyo"))
    now_ = japan_time.strftime("%m-%d-%H-%M")
    
    config['dataloader'] = config['dataloader'].__name__
    results.update(config)
    np.save("results/week0/" + now_ + ".npy", results)


if __name__ == '__main__':
    # setting
    config = {
        'dataloader': PseudoDataLoader,
        'n_modals': 2,
        'n_modules': 10,
        'n_classes': 10,
        'dpc': 1,
        'T': 25,
        'recommendation': True,
        'num_trainmovement': 1,
        'save': True,
        'n_seeds': 2,
    }

    # run
    experiment = Experiment(**config)
    results = experiment.run_seeds(n_seeds=config['n_seeds'])

    # save
    if config['save']:
        save_result(results, config)
    
    # ================================================
    

    # ================================================
    
