import random
from tqdm import tqdm
import numpy as np
from dataloader import DataLoader, MMDataLoader, PseudoDataLoader, PseudoDataLoader0, PseudoDataLoader1
from GcCnn import ControlGCN
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

class Experiment:
    def __init__(self, dataloader=DataLoader, n_modals=1, n_modules=40, n_classes=10, dpc=1, T=25, recommendation=False, num_trainmovement=1, **kwargs):
        self.model = ControlGCN
        self.loader = dataloader
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

        # train
        for vecs, target in tqdm(train_loader, desc='Train'):
            model.learn(vecs, target)
            #break
        
        # train movement
        if self.recommendation:
            map = {1: 0, 5: 1, 10: 2, 20: 3}
            for i in range(self.num_trainmovement):
                random.seed(seed_ + i)
                np.random.seed(seed_ + i)
                for vecs, target in tqdm(train_loader, desc='Train Movement {}'.format(i+1)):
                    model.learn_movement(vecs, target, map[self.dpc])
                    

        # eval
        random.seed(seed_)
        np.random.seed(seed_)
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
    

    def run_seeds(self, n_seeds=1,):
        brief_all = np.zeros((n_seeds, self.datasize, self.T), dtype=np.float16)
        mi_T_all = np.zeros((n_seeds, self.datasize, self.T), dtype=np.int8)
        pi_T_all = np.zeros((n_seeds, self.datasize, self.T), dtype=np.int8)
        target_all = np.zeros((n_seeds, self.datasize), dtype=np.int8)

        for i in range(n_seeds):
            brief, mi_T, pi_T, target = self.run(i)

            brief_all[i] = brief
            mi_T_all[i] = mi_T
            pi_T_all[i] = pi_T
            target_all[i] = target
        
        results = {
            'brief': brief_all,
            'mi_T': mi_T_all,
            'pi_T': pi_T_all,
            'target': target_all,
        }
        return results


def plot_brief(brief, name=''):
    avg = np.mean(brief, axis=1)
    med = np.median(avg, axis=0)
    plt.plot(range(1,25+1), med, marker='.', label='label')
    plt.savefig('results2/week0/brief_avg' + name + '.png')
    plt.clf()


def save_result(results={},):
    japan_time = datetime.now(pytz.timezone("Asia/Tokyo"))
    now_ = japan_time.strftime("%m-%d-%H-%M")
    np.save("results/week0/" + now_ + ".npy", results)


if __name__ == '__main__':
    # setting
    config = {
        'dataloader': PseudoDataLoader0,
        'n_modals': 2,
        'n_modules': 10,
        'n_classes': 10,
        'dpc': 10,
        'T': 25,
        'recommendation': True,
        'num_trainmovement': 1,
        'save': True,
        'n_seeds': 1,
    }

    # run
    experiment = Experiment(**config)
    results = experiment.run_seeds(n_seeds=config['n_seeds'])

    # save
    config.pop('dataloader', None)
    results.update(config)
    if config['save']:
        save_result(results)
    
    # ================================================
    # setting
    config = {
        'dataloader': PseudoDataLoader1,
        'n_modals': 2,
        'n_modules': 10,
        'n_classes': 10,
        'dpc': 10,
        'T': 25,
        'recommendation': True,
        'num_trainmovement': 1,
        'save': True,
        'n_seeds': 1,
    }

    # run
    experiment = Experiment(**config)
    results = experiment.run_seeds(n_seeds=config['n_seeds'])

    # save
    config.pop('dataloader', None)
    results.update(config)
    if config['save']:
        save_result(results)
    

    # ================================================
    # setting
    config = {
        'dataloader': PseudoDataLoader,
        'n_modals': 2,
        'n_modules': 10,
        'n_classes': 10,
        'dpc': 20,
        'T': 25,
        'recommendation': True,
        'num_trainmovement': 1,
        'save': True,
        'n_seeds': 7,
    }

    # run
    experiment = Experiment(**config)
    results = experiment.run_seeds(n_seeds=config['n_seeds'])

    # save
    config.pop('dataloader', None)
    results.update(config)
    if config['save']:
        save_result(results)
    
