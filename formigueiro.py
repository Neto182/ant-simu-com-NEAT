import pygame, neat, random, os, sys, time, math
from neat.nn import RecurrentNetwork

# -------------- CONFIGURAÇÃO --------------
WIDTH, HEIGHT, TILE = 800, 600, 4
FPS = 60
MAX_ANTS = 50
PH_DECAY, PH_DEPOSIT = 0.995, 5
FOOD_SPAWN, WATER_SPAWN = 0.01, 0.005
EVAL_TIME = 10.0
MAX_ENERGY = int(EVAL_TIME * FPS)
TARGET = 5            # meta individual
GENERATIONS = 50
CONFIG_PATH = 'neat-config.txt'

# Cores
DIRT        = (194,178,128)
ANT_COLOR   = (0,0,0)
FOOD_COLOR  = (0,200,0)
WATER_COLOR = (0,100,200)
PH_COLOR    = (200,200,255)
TEXT_COLOR  = (255,255,255)
LIFE_BG     = (100,100,100)
LIFE_FG     = (200,50,50)

def sign(x): return 1 if x>0 else -1 if x<0 else 0

class Ant:
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    def __init__(self, genome, config, x, y, terr):
        # usa rede recorrente em vez de feed‑forward
        self.net    = RecurrentNetwork.create(genome, config)
        self.g      = genome
        self.x, self.y = x, y
        self.terr   = terr
        self.prev   = self.dist_to(self.terr.foods)
        self.carry  = 0     # 0=none,1=food,2=water
        self.delivered = 0

    def dist_to(self, items):
        if not items: return self.terr.w + self.terr.h
        return min(abs(ix-self.x)+abs(iy-self.y) for ix,iy in items)

    def sense(self):
        t = self.terr; w,h = t.w, t.h
        pher   = t.phero[self.y][self.x] / 255
        f_here = (self.x,self.y) in t.set_food
        w_here = (self.x,self.y) in t.set_water
        dxh,dyh = (t.home[0]-self.x)/w, (t.home[1]-self.y)/h

        def nearest_vec(lst):
            if not lst: return 0.0, 0.0
            d,ix,iy = min(((abs(ix-self.x)+abs(iy-self.y),ix,iy) for ix,iy in lst))
            return (ix-self.x)/w, (iy-self.y)/h

        dxf,dyf = nearest_vec(t.foods)
        dxw,dyw = nearest_vec(t.water)

        local = []
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                nx,ny = self.x+dx, self.y+dy
                if 0<=nx<w and 0<=ny<h:
                    local += [
                        (nx,ny) in t.set_food,
                        (nx,ny) in t.set_water,
                        t.phero[ny][nx]/255
                    ]
                else:
                    local += [0,0,0]

        walls = [
            self.x==0, self.x==w-1,
            self.y==0, self.y==h-1
        ]

        return [pher, f_here, w_here, dxh, dyh, dxf, dyf, dxw, dyw] + local + walls

    def update(self):
        inp = self.sense()
        out = self.net.activate(inp)
        dx,dy = Ant.moves[out.index(max(out))]

        # se carregando, força retorno
        if self.carry:
            hx,hy = self.terr.home
            ndx,ndy = sign(hx-self.x), sign(hy-self.y)
            if (ndx,ndy) in Ant.moves:
                dx,dy = ndx, ndy

        # fitness shaping (food)
        nd = self.dist_to(self.terr.foods)
        self.g.fitness += (self.prev - nd) * 0.1
        self.prev = nd

        # mover
        nx,ny = self.x+dx, self.y+dy
        if not (0<=nx<self.terr.w and 0<=ny<self.terr.h):
            nx,ny = self.x-dx, self.y-dy
        self.x, self.y = nx, ny

        # pick up
        if not self.carry and (nx,ny) in self.terr.set_food:
            self.carry = 1; self.terr.remove_food(nx,ny)
        elif not self.carry and (nx,ny) in self.terr.set_water:
            self.carry = 2; self.terr.remove_water(nx,ny)

        # drop off at home
        if self.carry and (nx,ny) == self.terr.home:
            if self.carry == 1:
                self.delivered += 1
                self.g.fitness  += 5
                self.terr.food_count += 1
                self.terr.life    = min(self.terr.max_life, self.terr.life + 10)
            else:  # água
                self.terr.life = min(self.terr.max_life, self.terr.life + 15)
            self.carry = 0
            if len(self.terr.ants) < MAX_ANTS:
                self.terr.clone(self)

        # drop pheromone
        deposit = PH_DEPOSIT if self.carry else PH_DEPOSIT/2
        self.terr.phero[ny][nx] = min(255, self.terr.phero[ny][nx] + deposit)


class Terrarium:
    def __init__(self, config):
        self.w, self.h = WIDTH//TILE, HEIGHT//TILE
        self.home       = (self.w//2, self.h//2)
        self.config     = config

    def reset(self, genomes):
        self.ants       = []
        self.food_count = 0
        self.foods      = []; self.water = []
        self.set_food   = set(); self.set_water = set()
        self.phero      = [[0]*self.w for _ in range(self.h)]

        # vida do formigueiro
        self.max_life = 100
        self.life     = self.max_life

        # spawn inicial de comida e água
        for _ in range(int(self.w*self.h * 0.03)):
            x,y = random.randrange(self.w), random.randrange(self.h)
            self.add_food(x,y)
        for _ in range(int(self.w*self.h * 0.015)):
            x,y = random.randrange(self.w), random.randrange(self.h)
            self.add_water(x,y)

        # criar formigas
        for idx, g in genomes:
            angle = 2*math.pi * idx / len(genomes)
            x = self.home[0] + int(math.cos(angle)*10)
            y = self.home[1] + int(math.sin(angle)*10)
            g.fitness = 0
            self.ants.append(Ant(g, self.config, x, y, self))

    def add_food(self, x, y):
        self.foods.append((x,y)); self.set_food.add((x,y))
    def add_water(self, x, y):
        self.water.append((x,y)); self.set_water.add((x,y))
    def remove_food(self, x, y):
        self.set_food.remove((x,y)); self.foods.remove((x,y))
    def remove_water(self, x, y):
        self.set_water.remove((x,y)); self.water.remove((x,y))

    def clone(self, ant):
        self.ants.append(Ant(ant.g, self.config, *self.home, self))

    def update(self):
        # vida decai ao longo de EVAL_TIME
        decay = self.max_life / (FPS * EVAL_TIME)
        self.life = max(0, self.life - decay)

        # spawn randômico
        if random.random() < FOOD_SPAWN:
            self.add_food(random.randrange(self.w), random.randrange(self.h))
        if random.random() < WATER_SPAWN:
            self.add_water(random.randrange(self.w), random.randrange(self.h))

        # atualizar formigas
        for a in list(self.ants):
            a.update()
        # decair feromônio
        for y in range(self.h):
            for x in range(self.w):
                self.phero[y][x] *= PH_DECAY

    def draw(self, screen, font):
        screen.fill(DIRT)

        # comida e água
        for x,y in self.foods:
            pygame.draw.rect(screen, FOOD_COLOR,  (x*TILE,y*TILE,TILE,TILE))
        for x,y in self.water:
            pygame.draw.rect(screen, WATER_COLOR, (x*TILE,y*TILE,TILE,TILE))

        # feromônio
        for y in range(self.h):
            for x in range(self.w):
                if self.phero[y][x] > 1:
                    pygame.draw.rect(screen, PH_COLOR, (x*TILE,y*TILE,TILE,TILE))

        # ninho
        pygame.draw.circle(screen, (255,0,0),
                           (self.home[0]*TILE,self.home[1]*TILE),
                           TILE*2)

        # formigas + destaque líder
        best = max((a.delivered for a in self.ants), default=0)
        for a in self.ants:
            size = TILE*2 if (a.delivered == best and best > 0) else TILE
            off  = (size - TILE)//2
            pygame.draw.rect(screen, ANT_COLOR,
                             (a.x*TILE - off, a.y*TILE - off, size, size))

        # HUD: meta, líder, vida
        txt = font.render(f"Meta:{TARGET} Líder:{best} Vida:{int(self.life)}%", True, TEXT_COLOR)
        screen.blit(txt, (10,10))
        # barra de vida
        bar_w, bar_h = 200, 10
        pygame.draw.rect(screen, LIFE_BG, (10,30, bar_w, bar_h))
        fill = (self.life/self.max_life) * bar_w
        pygame.draw.rect(screen, LIFE_FG, (10,30, fill, bar_h))


def eval_genomes(genomes, config):
    terr = Terrarium(config)
    terr.reset(genomes)
    start = time.time()
    while time.time() - start < EVAL_TIME and terr.life > 0:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        terr.update()
        terr.draw(screen, font)
        pygame.display.flip()
        clock.tick(FPS)
        # meta individual
        if any(a.delivered >= TARGET for a in terr.ants):
            break


if __name__ == '__main__':
    pygame.init(); pygame.font.init()
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont(None,24)

    # configura NEAT para redes recorrentes (veja neat-config.txt)
    cfg = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        os.path.join(os.path.dirname(__file__), CONFIG_PATH)
    )

    p = neat.Population(cfg)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    winner = p.run(eval_genomes, GENERATIONS)

    # exibe o melhor indivíduo
    terr = Terrarium(cfg)
    terr.reset([(0, winner)])
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        terr.update()
        terr.draw(screen, font)
        pygame.display.flip()
        clock.tick(FPS)
