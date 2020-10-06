import sys
import random
import numpy
import subprocess as sub
import pickle

from AFPO.afpomoo import AFPOMoo
from softbot_robot import SoftbotRobot
from utils import StructureGenotype, StructurePhenotype, get_seq_num


import matplotlib.pyplot as plt


seed = 1101
numpy.random.seed(seed)
random.seed(seed)
POP_SIZE = 32*3  # how large of a population are we using?
GENS = 5000  # how many generations are we optimizing for?

printing = True

# Setup evo run
def get_phenotype():
    new_phenotype = StructurePhenotype(StructureGenotype)
    return new_phenotype

def robot_factory():
    phenotype = get_phenotype()
    return SoftbotRobot(phenotype, get_seq_num, "run_%d" % seed)

afpo_alg = AFPOMoo(robot_factory, pop_size=POP_SIZE)

# do each generation.
best_design = []
for generation in range(GENS):
    if printing:
        print("generation %d" % (generation))

    dom_data = afpo_alg.generation()

    if printing:
        print("%d individuals are dominating" % (dom_data[0],))
        dom_inds = sorted(dom_data[1], key= lambda x: x.get_fitness(), reverse=False)
        print('\n'.join([str(d) for d in dom_inds]))

    best_fit, best_robot = afpo_alg.get_best()

    best_design = best_robot.morphology

    if (generation % 100 == 0) and (best_fit > 0.01):
        pickle_out = open("best_robot_gen{0}_fit{1}.pickle".format(generation, int(100*best_fit)), "wb")
        pickle.dump(best_robot, pickle_out)
        pickle_out.close()

        print("plotting best so far")
        plt.imshow(best_design[:, :, 0])
        plt.savefig('best_design_gen{}.png'.format(generation), bbox_inches='tight', dpi=900)
