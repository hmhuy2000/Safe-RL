from copy import deepcopy
import numpy as np
from safe_rl.utils.mpi_tools import mpi_avg
from safe_rl.pg.utils import EPS
import safe_rl.pg.trust_region as tro

class Agent:

    def __init__(self, **kwargs):
        self.params = deepcopy(kwargs)

    def set_logger(self, logger):
        self.logger = logger

    def prepare_update(self, training_package):
        # training_package is a dict with everything we need (and more)
        # to train.
        self.training_package = training_package

    def prepare_session(self, sess):
        self.sess = sess

    def update_pi(self, inputs):
        raise NotImplementedError

    def log(self):
        pass

    def ensure_satisfiable_penalty_use(self):
        reward_penalized = self.params.get('reward_penalized', False)
        objective_penalized = self.params.get('objective_penalized', False)
        assert not(reward_penalized and objective_penalized), \
            "Can only use either reward_penalized OR objective_penalized, " + \
            "not both."

        if not(reward_penalized or objective_penalized):
            learn_penalty = self.params.get('learn_penalty', False)
            assert not(learn_penalty), \
                "If you are not using a penalty coefficient, you should " + \
                "not try to learn one."

    def ensure_satisfiable_optimization(self):
        first_order = self.params.get('first_order', False)
        trust_region = self.params.get('trust_region', False)
        assert not(first_order and trust_region), \
            "Can only use either first_order OR trust_region, " + \
            "not both."

    @property
    def cares_about_cost(self):
        return self.use_penalty or self.constrained

    @property
    def clipped_adv(self):
        return self.params.get('clipped_adv', False)

    @property
    def constrained(self):
        return self.params.get('constrained', False)

    @property
    def first_order(self):
        self.ensure_satisfiable_optimization()
        return self.params.get('first_order', False)

    @property
    def learn_penalty(self):
        # Note: can only be true if "use_penalty" is also true.
        self.ensure_satisfiable_penalty_use()
        return self.params.get('learn_penalty', False)

    @property
    def penalty_param_loss(self):
        return self.params.get('penalty_param_loss', False)

    @property
    def objective_penalized(self):
        self.ensure_satisfiable_penalty_use()
        return self.params.get('objective_penalized', False)

    @property
    def reward_penalized(self):
        self.ensure_satisfiable_penalty_use()
        return self.params.get('reward_penalized', False)

    @property
    def save_penalty(self):
        # Essentially an override for CPO so it can save a penalty coefficient
        # derived in its inner-loop optimization process.
        return self.params.get('save_penalty', False)

    @property
    def trust_region(self):
        self.ensure_satisfiable_optimization()
        return self.params.get('trust_region', False)

    @property
    def use_penalty(self):
        return self.reward_penalized or \
               self.objective_penalized


class PPOAgent(Agent):
    
    def __init__(self, clip_ratio=0.2, 
                       pi_lr=3e-4, 
                       pi_iters=80, 
                       kl_margin=1.2,
                       **kwargs):
        super().__init__(**kwargs)
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.pi_iters = pi_iters
        self.kl_margin = kl_margin
        self.params.update(dict(
            clipped_adv=True,
            first_order=True,
            constrained=False
            ))

    def update_pi(self, inputs):

        # Things we need from training package
        train_pi = self.training_package['train_pi']
        d_kl = self.training_package['d_kl']
        target_kl = self.training_package['target_kl']

        # Run the update
        for i in range(self.pi_iters):
            _, kl = self.sess.run([train_pi, d_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > self.kl_margin * target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        self.logger.store(StopIter=i)

    def log(self):
        self.logger.log_tabular('StopIter', average_only=True)

