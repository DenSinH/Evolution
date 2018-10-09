import pygame
import random
import copy
import math

# initialisations
pygame.init()
pygame.font.init()

# screen parameters
size = (width, height) = (1280, 720)
fps = 60

# necessary objects
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
myfont = pygame.font.SysFont("monospace", 12, True)

# misc parameters
GAME_LOOP = True
INIT_LOOP = True

mousepos = (0, 0)

LIFESPAN = 300  # frames\
GENESPAN = LIFESPAN/10

GEN_SIZE = 10
RANDOM_FACTOR = 1.
VEC_L = 10  # px

SHOW_BEST = False


# collision detection

def x_col(obj1, obj2):
    if obj1[0] + obj1[2] > obj2[0] > obj1[0] - obj2[2]:
        return True
    return False


def y_col(obj1, obj2):
    if obj1[1] + obj1[3] > obj2[1] > obj1[1] - obj2[3]:
        return True
    return False


def collision(obj1, obj2):
    # [x, y, width, height]
    return x_col(obj1, obj2) and y_col(obj1, obj2)


# distance calculator
def distsq(pos1, pos2):
    return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2


class Obstacle(object):

    def __init__(self, x, y, width, height):
        self.collision_box = [self.x, self.y, self.width, self.height] = [x, y, width, height]


class Goal(object):

    def __init__(self, x, y, width=20, height=20):
        self.collision_box = [self.x, self.y, self.width, self.height] = [x, y, width, height]


GOAL1 = Goal(400, 400)
GOAL2 = Goal(600, 400)
GOAL3 = Goal(600, 600)
GOAL4 = Goal(400, 600)

if not INIT_LOOP:
    ROCKET_GOAL_LIST = [GOAL1, GOAL2, GOAL3, GOAL4]
    OBSTACLE_LIST = [Obstacle(430, 330, 160, 360)]
    # [Obstacle(600, -700, 200, 750), Obstacle(600, 100, 200, 500), Obstacle(600, 650, 200, height)]
else:
    ROCKET_GOAL_LIST = []
    OBSTACLE_LIST = []


# default class for moving object
class Physics(object):

    def __init__(self, obstacles=OBSTACLE_LIST):
        self.x = 0.
        self.y = 0.
        self.xv = 0.
        self.yv = 0.
        self.xa = 0.
        self.ya = 0.
        self.width = 0.
        self.height = 0.

        self.obstacles = obstacles

    def move(self):

        self.xv += self.xa
        self.yv += self.ya

        new_x = self.x + self.xv
        new_y = self.y + self.yv

        old_x = int(self.x)
        old_y = int(self.y)

        for obstacle in self.obstacles:
            if collision([new_x, new_y, self.width, self.height], obstacle.collision_box):
                if collision([new_x, old_y, self.width, self.height], obstacle.collision_box):
                    if self.x < obstacle.x:
                        new_x = obstacle.x - self.width
                    else:
                        new_x = obstacle.x + obstacle.width
                    self.xv = 0
                if collision([old_x, new_y, self.width, self.height], obstacle.collision_box):
                    if self.y < obstacle.y:
                        new_y = obstacle.y - self.height
                    else:
                        new_y = obstacle.y + obstacle.height
                    self.yv = 0

        self.x = new_x
        self.y = new_y

        # remove """ to enable screen border
        """
        if self.x < 0:
            self.x = 0
            self.xv = 0
        elif self.x + self.width > width:
            self.x = width - self.width
            self.xv = 0

        if self.y < 0:
            self.y = 0
            self.yv = 0
        elif self.y + self.height > height:
            self.y = height - self.height
            self.yv = 0
        """


# Gene object
# contains information for direction of movement
class Gene(object):

    def __init__(self, phi, frames):
        self.phi = phi
        self.xa = math.cos(phi)
        self.ya = math.sin(phi)
        self.frames = frames


# DNA object
# contains all information for movement of rocket during lifespan
class DNA(object):

    def __init__(self, lifespan=LIFESPAN, dna=None, goalgenes_frames=None):
        self.genes = []

        # generate random genes if genes are not provided
        if dna is None:
            lifespan = 0
            while lifespan < LIFESPAN:
                genespan = random.randint(1, 6)  # 6 is good for max
                self.genes.append(Gene(2*math.pi*random.random(), genespan))
                lifespan += genespan
        else:
            self.genes = dna

        # counting variables for determining reading position in DNA
        self.currentgene = 0
        self.currentframe = 0

        if goalgenes_frames is None:
            self.goalgenes_frames = [(0, 0)]
        else:
            self.goalgenes_frames = goalgenes_frames

    def get_update(self):
        # determine movement data from current gene
        data = (self.genes[self.currentgene].xa, self.genes[self.currentgene].ya)

        # increment frame or gene
        if self.currentframe >= self.genes[self.currentgene].frames:
            self.currentgene += 1
            self.currentframe = 0
        else:
            self.currentframe += 1

        return data

    def mutate(self, stage, randomness=1.):
        # Make genes mutate slightly or very much depending on randomness
        # randomness 1. is completely new genome, randomness 0. is no mutation
        # TODO: make later genes mutate more
        # TODO: create "DNA stages", splice up DNA in parts to mutate separately (change length etc)
        try:
            start = self.goalgenes_frames[stage][0]
        except IndexError:
            start = 0

        for i in range(start, len(self.genes)):
            gene = self.genes[i]
            self.genes[i] = Gene(gene.phi + 2.*math.pi*randomness *
                                 math.exp((i - start)/(len(self.genes) - start))*(random.gauss(0., 0.5)), gene.frames)

    def record_gene(self, goal_no):
        try:
            # the 0th goal is the starting position, otherwise mutation will only occur after the
            # last reached goal, which is exactly what we don't want
            self.goalgenes_frames[goal_no] = (self.currentgene, self.currentframe)
        except IndexError:
            self.goalgenes_frames.append((self.currentgene, self.currentframe))


# Rocket object
# the actual moving part
class Rocket(Physics):

    def __init__(self, x=50, y=height/2, goal=ROCKET_GOAL_LIST, lifespan=LIFESPAN, dna=None, goalgenes_frames=None):
        Physics.__init__(self)

        # position variables
        self.x = x
        self.y = y
        self.width = 10
        self.height = 10

        # list of goals
        self.goal_list = goal
        self.goal_no = 0

        # movement variables
        self.xv = 0.
        self.yv = 0.
        self.xa = 0.
        self.ya = 0.

        # collision box, to be updated every frame
        self.collision_box = [self.x, self.y, self.width, self.height]

        self.DNA = DNA(lifespan=lifespan, dna=dna, goalgenes_frames=goalgenes_frames)

        # score and finished parameter for in the simulation
        self.score = 0
        self.finished = False

        # counting variable
        # counts frames since reaching last goal
        self.frame = 0

    def update_collision_box(self):
        self.collision_box = [self.x, self.y, self.width, self.height]

    def update(self, stage, frame=0):
        # fetch data on DNA
        (self.xa, self.ya) = self.DNA.get_update()

        # move according to data
        self.move()
        self.update_collision_box()

        # determine if goal is reached
        # give score accordingly
        # score only matters up until goal matching the stage number
        # this is to create the most eficcient path
        if self.goal_no <= stage and self.goal_no < len(self.goal_list):
            if collision(self.collision_box, self.goal_list[self.goal_no].collision_box):
                self.score += LIFESPAN*100
                if self.goal_no + 1 <= len(self.goal_list):
                    self.goal_no += 1
                # goal is recorded after, since first goalgene_frame is (0, 0) (base)
                self.DNA.record_gene(self.goal_no)

                # -1 is so it is reset to 0 after end of function
                self.frame = -1
            else:
                self.score -= math.sqrt(distsq([self.x, self.y],
                                               self.goal_list[self.goal_no].collision_box[:2]))*self.frame/1000
        if self.goal_no == len(self.goal_list):
            if not self.finished:
                self.finished = True
                return True

        self.frame += 1
        return False

    def color(self):
        # color shows the progress in the path
        max_goal = len(self.goal_list) + 1
        return 0, 255*self.goal_no/max_goal, 255 - 255*self.goal_no/max_goal


# Generation object
# contains a number of Rockets to be simulated, selected, mutated and simulated again
class Generation(object):

    def __init__(self, lifespan=LIFESPAN):
        # generate generation
        self.units = []
        for i in range(GEN_SIZE):
            self.units.append(Rocket(lifespan=lifespan))

        # copy units to use later on
        self.units_origin = copy.deepcopy(self.units)

        self.lifespan = lifespan

        # counting variables
        self.frame = 0
        self.number = 1
        self.stage = 0

        self.number_finished = 0

        self.max_score = 0.
        self.prev_change_gen = 0.
        self.randomness = RANDOM_FACTOR

    def update(self):
        # update and draw all units
        # if lifespan of generation has passed, go to the next generation
        if self.frame < self.lifespan:

            for unit in self.units:
                if not unit.finished:
                    if unit.update(self.stage, self.frame):
                        self.number_finished += 1
                        if self.number_finished == 1:
                            print "FIRST FINISHED AT", self.frame
                            self.lifespan = min(self.lifespan, int(self.frame*1.1))
                    elif unit.goal_no > self.stage:
                        self.lifespan = min(self.lifespan, int(self.frame*1.1))
                pygame.draw.rect(screen, unit.color(), unit.collision_box)
            self.frame += 1
        else:
            self.next_gen()

    def next_gen(self):

        # finish off all unfinished units
        for unit in self.units:
            # if it is not finished, we don't have to worry about an IndexError, as the goal_no will be less than
            # the length of goal_list
            if not unit.finished:
                unit.score += 1000 - 10*self.frame - distsq((unit.x, unit.y),
                                                            unit.goal_list[unit.goal_no].collision_box[:2])/1000

        # determine order for score
        indices = range(GEN_SIZE)
        ranked_indices = sorted(indices, reverse=True, key=lambda i: self.units[i].score)
        ranked_rockets = [self.units_origin[i] for i in ranked_indices]
        for i in range(GEN_SIZE):
            ranked_rockets[i].DNA.goalgenes_frames = copy.copy(self.units[ranked_indices[i]].DNA.goalgenes_frames)

        print "ORIGINAL DNA", ranked_rockets[0].DNA.goalgenes_frames

        # show best performing rocket in separate visual loop
        show_best = SHOW_BEST
        sim_frame = 0
        sim_fps = fps
        best = copy.deepcopy(ranked_rockets[0])
        trace = [(best.x, best.y)]
        vectors = []
        while show_best:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    # space to turn off showing
                    if event.key == pygame.K_SPACE:
                        show_best = False
                    elif event.key == pygame.K_LEFT:
                        sim_fps = 600 - sim_fps

            screen.fill((255, 255, 255))

            # if we are not simulating, just mark the best performing rocket
            vectors.append([(best.x, best.y), (best.x + VEC_L*best.xa, best.y + VEC_L*best.ya)])
            best.update(sim_frame)
            trace.append((best.x, best.y))
            pygame.draw.rect(screen, best.color(), best.collision_box)
            sim_frame += 1

            if sim_frame == self.lifespan:
                sim_frame = 0
                best = copy.deepcopy(ranked_rockets[0])
                trace = [(best.x, best.y)]
                vectors = []
            else:
                for vec in vectors:
                    pygame.draw.line(screen, (0, 0, 0), vec[0], vec[1])
                pygame.draw.lines(screen, (0, 0, 0), False, trace)

            # draw the goals
            for i in range(len(ROCKET_GOAL_LIST)):
                goal = ROCKET_GOAL_LIST[i]
                pygame.draw.rect(screen, (255, 0, 0), goal.collision_box)
                screen.blit(myfont.render("{0}".format(i), True, (10, 10, 10)), (goal.x, goal.y))
            for ob in OBSTACLE_LIST:
                pygame.draw.rect(screen, (0, 0, 0), ob.collision_box)

            # report generation number
            screen.blit(myfont.render("GENERATION: {0}".format(GEN.number), True, (10, 10, 10)), (10, 10))

            pygame.display.flip()
            clock.tick(sim_fps)

        # determine maximum score reached in this generation
        max_score = round(self.units[ranked_indices[0]].score, 2)
        if max_score > self.max_score or self.number == 1:
            self.max_score = max_score
            self.prev_change_gen = self.number
        elif self.number - self.prev_change_gen > 10:
            self.randomness /= 2.
        elif self.number - self.prev_change_gen > 15:
            self.randomness = RANDOM_FACTOR

        if self.randomness < RANDOM_FACTOR/(2**8):
            self.stage += 1
            self.lifespan = LIFESPAN
            self.randomness = RANDOM_FACTOR
            self.prev_change_gen = self.number

        print self.max_score, self.prev_change_gen

        # recreate all units depending on score
        # use preset ratio's to eliminate RNG
        self.units = []
        for i in [0]*(GEN_SIZE//2) + [1]*(GEN_SIZE//4) + [2]*(GEN_SIZE//8) + [3]*(GEN_SIZE/16) + [4]*(GEN_SIZE//16):
            self.units.append(Rocket(dna=copy.deepcopy(ranked_rockets[i].DNA.genes),
                                     goalgenes_frames=copy.copy(ranked_rockets[i].DNA.goalgenes_frames)))

        # add more of the best rocket until wanted generation size is reached
        while len(self.units) < GEN_SIZE:
            self.units.append(Rocket(dna=copy.deepcopy(ranked_rockets[0].DNA.genes),
                                     goalgenes_frames=copy.copy(ranked_rockets[0].DNA.goalgenes_frames)))

        # mutate all rockets, except one copy of the best performing one
        # TODO: Automate randomness
        #   - seems done
        for unit in self.units[1:]:
            unit.DNA.mutate(self.stage, randomness=self.randomness)  # random.expovariate(40))  # 0.05 seems good

        # create a copy of the new generation
        self.units_origin = copy.deepcopy(self.units)

        # counting
        self.number += 1
        self.frame = 0
        self.number_finished = 0


GEN = Generation()

# TODO: CREATE INITLOOP
actions_performed = []
init_pos = (-1, -1)
min_x, min_y, w, h = 0, 0, 0, 0

while INIT_LOOP:

    mousepos = pygame.mouse.get_pos()
    if init_pos != (-1, -1):
        min_x = min(init_pos[0], mousepos[0])
        min_y = min(init_pos[1], mousepos[1])
        w = abs(init_pos[0] - mousepos[0])
        h = abs(init_pos[1] - mousepos[1])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            INIT_LOOP = False
            GAME_LOOP = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if not any(collision(obstacle.collision_box,
                                     [mousepos[0], mousepos[1], 20, 20]) for obstacle in OBSTACLE_LIST):
                    ROCKET_GOAL_LIST.append(Goal(mousepos[0], mousepos[1]))
                    actions_performed.append("GOAL")
            elif event.button == 3:
                if init_pos == (-1, -1):
                    init_pos = mousepos
                else:
                    if not any(collision(goal.collision_box, [min_x, min_y, w, h]) for goal in ROCKET_GOAL_LIST):
                        OBSTACLE_LIST.append(Obstacle(min_x, min_y, w, h))
                        actions_performed.append("OBSTACLE")
                    init_pos = (-1, -1)
                    min_x, min_y, w, h = 0, 0, 0, 0

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if len(ROCKET_GOAL_LIST) > 0:
                    INIT_LOOP = False
            elif event.key == pygame.K_BACKSPACE:
                if len(actions_performed) == 0:
                    pass
                elif actions_performed[-1] == "GOAL":
                    ROCKET_GOAL_LIST.pop()
                    actions_performed.pop()
                elif actions_performed[-1] == "OBSTACLE":
                    OBSTACLE_LIST.pop()
                    actions_performed.pop()

        screen.fill((255, 255, 255))

    # Draw goals and obstacles
    for i in range(len(ROCKET_GOAL_LIST)):
        goal = ROCKET_GOAL_LIST[i]
        pygame.draw.rect(screen, (255, 0, 0), goal.collision_box)
        screen.blit(myfont.render("{0}".format(i), True, (10, 10, 10)), (goal.x, goal.y))
    for ob in OBSTACLE_LIST:
                pygame.draw.rect(screen, (0, 0, 0), ob.collision_box)

    pygame.draw.rect(screen, (0, 0, 0), [min_x, min_y, w, h], 1)

    pygame.display.flip()

    clock.tick(fps)

label_blits = {
    "GENERATION": myfont.render("GENERATION:", True, (10, 10, 10)),
    "FRAME": myfont.render("FRAME:", True, (10, 10, 10)),
    "RANDOMNESS": myfont.render("RANDOMNESS:", True, (10, 10, 10)),
    "STAGE": myfont.render("STAGE:", True, (10, 10, 10)),
    "FPS": myfont.render("FPS:", True, (10, 10, 10))
}

label_vals = {
    "GENERATION": "number",
    "FRAME": "frame",
    "RANDOMNESS": "randomness",
    "STAGE": "stage",
    "FPS": "FPS"
}

number_blits = {}

keys = ["GENERATION", "FRAME", "RANDOMNESS", "STAGE", "FPS"]

# MAIN LOOP
while GAME_LOOP:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            GAME_LOOP = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                SHOW_BEST = not SHOW_BEST
            elif event.key == pygame.K_DOWN:
                RANDOM_FACTOR /= 2.
            elif event.key == pygame.K_UP:
                RANDOM_FACTOR *= 2.
            elif event.key == pygame.K_LEFT:
                fps = 600 - fps
            elif event.key == pygame.K_RIGHT:
                GEN.randomness = RANDOM_FACTOR/(2**10)

    screen.fill((255, 255, 255))

    # Update generation, then draw goals after
    GEN.update()
    for i in range(len(ROCKET_GOAL_LIST)):
        goal = ROCKET_GOAL_LIST[i]
        pygame.draw.rect(screen, (255, 0, 0), goal.collision_box)
        screen.blit(myfont.render("{0}".format(i), True, (10, 10, 10)), (goal.x, goal.y))
    for ob in OBSTACLE_LIST:
                pygame.draw.rect(screen, (0, 0, 0), ob.collision_box)

    # report generation number and current frame
    for i in range(len(keys)):
        key = keys[i]
        screen.blit(label_blits[key], (10, 10 + 10*i))
        if key != "FPS":
            val = getattr(GEN, label_vals[key])
        else:
            val = int(1000./clock.get_time())
        try:
            screen.blit(label_blits[val], (100, 10 + 10*i))
        except KeyError:
            number_blits[val] = myfont.render("{0}".format(val), True, (10, 10, 10))
            screen.blit(number_blits[val], (100, 10 + 10*i))

    pygame.display.flip()

    # we might as well go to the next generation if 10% had finished
    if GEN.number_finished > GEN_SIZE/10.:
        GEN.next_gen()

    clock.tick(fps)

pygame.quit()
quit()
