# Import packages
import pygame, sys, os, cv2, time
import numpy as np
import mediapipe as mp
from pygame.locals import *
from fastdtw import fastdtw
from CNN import CNN
from mediapipe_utls import *
from random import randrange, random, choice
from pyvidplayer import Video

LEVEL_VOCAB = {1:["abraco","ajuda", "feliz"], 2:["por favor", "amigo", "obrigado"], 3:["casa","professor","alegria"],
                       4:["brincar","bom","ruim"], 5:["feliz","parar","comecar","saber"], 6:["LIBRAS","surdo","dia"]} #"ola"

# Initiate pygame
pygame.init()
main_clock = pygame.time.Clock()
screen = pygame.display.set_mode((600,480), pygame.RESIZABLE,32)
font = pygame.font.Font('assets/fonts/FreePixel.ttf', 40)
click = False

def draw_text(text, color, surface, x, y, font=font,anchor='center'):
    '''
    Function that write some text in the screen.
    '''
    textobj = font.render(text,1,color)
    if anchor == 'center': textrect = textobj.get_rect(center=(x,y))
    if anchor == 'topleft': textrect = textobj.get_rect(topleft=(x,y))
    surface.blit(textobj, textrect)

screen.fill((0,0,0))
draw_text('loading',(255,255,255),screen,100,100)
pygame.display.update()

# Initiate opencv
video_device = 0
cap = cv2.VideoCapture(video_device)

# Initiate mediapipe
mp_hands = mp.solutions.hands           # Hand detection model
mp_pose = mp.solutions.pose             # Pose detection model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Initiate CNN
VOCABULARY_LENGTH = 20
actions = ["abraco", "amigo", "por favor", "obrigado", "casa", "ajuda", "alegria", "professor", "brincar", "bom",
                "ruim", "LIBRAS", "saber", "parar", "comecar", "dia", "surdo", "comer", "ola", "feliz"]

cnn = CNN(actions, VOCABULARY_LENGTH)
cnn.load_model("models/20_sign_model.h5")

class Button(object):
    def __init__(self, text, width=100, height=40, pos=[0,0], elevation=6, size_mod=1):
        # core attributes
        self.pressed = False
        self.click = False
        self.elevation = elevation
        self.dynamic_elevation = elevation
        self.original_y_pos = pos[1]
        self.size_mod = size_mod

        # top rectangle
        self.top_rect = pygame.Rect(pos, (width,height))
        self.top_color = (175,175,255)

        # botton rectangle
        self.bot_rect = pygame.Rect(pos,(width,self.elevation))
        self.bot_color = (75,75,255)
        
        # text
        self.font = pygame.font.Font('assets/fonts/FreePixel.ttf', int(screen.get_height()/25))

        self.text = text
        self.text_surf = self.font.render(self.text, True, (255,255,255))
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)
    
    def draw(self):
        # elevation logic
        self.top_rect.y = self.original_y_pos - self.dynamic_elevation
        self.text_rect.center = self.top_rect.center

        self.bot_rect.midtop = self.top_rect.midtop
        self.bot_rect.height = self.top_rect.height + self.dynamic_elevation
        
        # draw
        pygame.draw.rect(screen, self.bot_color, self.bot_rect, border_radius = 12)
        pygame.draw.rect(screen, self.top_color, self.top_rect, border_radius = 12)
        screen.blit(self.text_surf, self.text_rect)
        self.check_click()

    def Update(self, width, height, pos, elevation=6):
        self.original_y_pos = pos[1]

        width = width*self.size_mod
        height = height*self.size_mod
        elevation = elevation*self.size_mod
        
        
        # top rectangle
        self.top_rect = pygame.Rect(pos, (width,height))

        # botton rectangle
        self.bot_rect = pygame.Rect(pos,(width,self.elevation))

        # text
        self.font = pygame.font.Font('assets/fonts/FreePixel.ttf', int(screen.get_height()/25))
        self.text_surf = self.font.render(self.text, True, (255,255,255))
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)

        self.draw()

    def check_click(self):
        self.click = False
        mx, my = pygame.mouse.get_pos()
        if self.top_rect.collidepoint((mx,my)):
            self.top_color = (215,215,255)                # change button color when mouse colide
            if pygame.mouse.get_pressed()[0]:
                self.dynamic_elevation = 0
                self.pressed = True
            else:
                self.dynamic_elevation = self.elevation
                if self.pressed:
                    self.pressed = False
                    self.click = True
        else:
##            self.dynamic_elevation = self.elevation
            self.top_color = (175,175,255)

class Enemy(object):
    def __init__(self, ground_level, level = 1):
        self.enemy_types = {1:{"name":"skeleton", "w":22, "h":33, "spd":2},
                            2:{"name":"goblin", "w":38, "h":38, "spd":4},
                            3:{"name":"skeleton", "w":22, "h":33, "spd":2},
                            4:{"name":"goblin", "w":38, "h":38, "spd":4}}
        
        self.level = level
        self.word = choice(LEVEL_VOCAB[self.level])
        word_index = LEVEL_VOCAB[self.level].index(self.word)
        self.type = self.enemy_types[word_index+1]
        
        # load sprites
        self.sprite_sheet = pygame.image.load(os.path.join("assets","images",self.type["name"],"walk.png"))
        w,h = self.type["w"], self.type["h"]
        self.sprites = [self.sprite_sheet.subsurface((i,0,w,h)) for i in range(0,self.sprite_sheet.get_width(),w)]
        self.current_sprite = 0
        self.image = self.sprites[self.current_sprite]
        self.screen_factor = 1/9

        # set positions and rect
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.x = -self.width
        self.y = ground_level - self.height

        self.speed_variance = random()*self.level*2 - self.level
        self.speed = self.type["spd"]
        
    def Update(self, ground_level, screen):
        screen_w, screen_h = pygame.display.get_surface().get_size()
        
        self.current_sprite += 0.4  # move through the frames, animating the sprite
        if self.current_sprite >= len(self.sprites): self.current_sprite = 0
        self.image = self.sprites[int(self.current_sprite)]
        self.image = pygame.transform.scale(self.image, (screen_w*self.screen_factor, screen_h*self.screen_factor))
        
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        
        self.x = self.x + int(screen_w/400) + self.speed_variance + self.speed
        self.y = ground_level - self.height

        # text
        text_size = screen_w/40
        en_font = pygame.font.Font('assets/fonts/FreePixel.ttf', int(text_size))
        self.textobj = en_font.render(self.word,1,(200,200,200))

        # text rect
        self.textrect = self.textobj.get_rect(center=(self.x+self.width/2, self.y-text_size-10))

        # top rectangle
        top_rect = pygame.Rect((self.x,self.y) , (self.width,text_size))
        top_rect.center = self.textrect.center
        top_color = (70,70,255)

        pygame.draw.rect(screen, top_color, top_rect, border_radius = 12)

        screen.blit(self.image, (self.x, self.y))
        screen.blit(self.textobj, self.textrect)

def cv2_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "BGR")

# Define game screens
def main_menu():
    pygame.mouse.set_visible(1)
    play_btn = Button('Jogar')
    options_btn = Button('Opções')
    ref_btn = Button('Creditos')
    quit_btn = Button('Sair')
    bg = pygame.image.load(os.path.join("assets","images","backgrounds","Building6.png"))
        
    while True:
        screen_w, screen_h = pygame.display.get_surface().get_size()
##        screen.fill((0,0,0))
        bg = pygame.transform.scale(bg, (screen_w, screen_h))
        screen.blit(bg, (0,0))
        
        draw_text('MENU',(255,255,255),screen,screen_w/2,screen_h/20)

        mx, my = pygame.mouse.get_pos()

        play_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, screen_h/5))
        options_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, 2*screen_h/5))
        ref_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, 3*screen_h/5))
        quit_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, 4*screen_h/5))

        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        if play_btn.click:
            level_selection()

        if options_btn.click:
            options()

        if ref_btn.click:
            credit()

        if quit_btn.click:
            cap.release()
            pygame.quit()
            sys.exit()

        pygame.display.update()
        main_clock.tick(60)

def credit():
    pygame.mouse.set_visible(1)

    return_btn = Button("Voltar")
##    bg = pygame.image.load(os.path.join("assets","images","backgrounds","Building6.png"))

    while True:
        screen_w, screen_h = pygame.display.get_surface().get_size()
        screen.fill((40,0,65))
##        bg = pygame.transform.scale(bg, (screen_w, screen_h))
##        screen.blit(bg, (0,0))
        draw_text('Creditos',(255,255,255),screen,screen_w/2,screen_h/20)

        tut_font = pygame.font.Font('assets/fonts/FreePixel.ttf', int(screen_w/50))
        draw_text('Jogo desenvolvido para o Trabalho de Conclusão de Curso de Luciano Alves Cardona Júnior,',
                  (255,255,255),screen,screen_w/12,2*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('sendo de sua autoria os algoritmos principais do jogo e das redes neurais para',
                  (255,255,255),screen,screen_w/12,3*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('detecção de sinais em libras e de imagens.',
                  (255,255,255),screen,screen_w/12,4*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('As imagens de fundo dos menus e do jogo foram obtidas gratuitamente,',
                  (255,255,255),screen,screen_w/12,6*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('principalmente no site Itch.io',
                  (255,255,255),screen,screen_w/12,7*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('Seguem os links para os assets utilizados:',
                  (255,255,255),screen,screen_w/12,8*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('https://lornn.itch.io/backgrounds-magic-school',
                  (255,255,255),screen,screen_w/12,10*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('https://9e0.itch.io/witches-pack',
                  (255,255,255),screen,screen_w/12,11*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('https://www.pixilart.com/art/pixelart-castle-5c3c0b96f86749b',
                  (255,255,255),screen,screen_w/12,12*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('https://mamanezakon.itch.io/forest-tileset',
                  (255,255,255),screen,screen_w/12,13*screen_h/20,font=tut_font, anchor='topleft')

        mx, my = pygame.mouse.get_pos()

        return_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, 4*screen_h/5))

        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        if return_btn.click:
            break
        
        pygame.display.update()
        main_clock.tick(60)

def options():
    pygame.mouse.set_visible(1)

    return_btn = Button("Voltar")
    bg = pygame.image.load(os.path.join("assets","images","backgrounds","Building6.png"))

    while True:
        screen_w, screen_h = pygame.display.get_surface().get_size()
        screen.fill((0,0,0))
        bg = pygame.transform.scale(bg, (screen_w, screen_h))
        screen.blit(bg, (0,0))
        draw_text('Opções',(255,255,255),screen,screen_w/2,screen_h/20)

        mx, my = pygame.mouse.get_pos()

        return_btn.Update(screen_w/4, screen_h/10, ((screen_w/2-screen_w/4)/2, 4*screen_h/5))

        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        if return_btn.click:
            break
        
        pygame.display.update()
        main_clock.tick(60)
        
def level_selection():
    pygame.mouse.set_visible(1)
    return_btn = Button('Voltar')
    buttons = []
    bg = pygame.image.load(os.path.join("assets","images","backgrounds","Building6.png"))

    for i in range(6):
        btn = Button(str(i+1))
        buttons.append(btn)
        
    while True:
        screen_w, screen_h = pygame.display.get_surface().get_size()
        screen.fill((0,0,0))
        bg = pygame.transform.scale(bg, (screen_w, screen_h))
        screen.blit(bg, (0,0))
        draw_text('select level',(255,255,255),screen,screen_w/2, screen_h/20)

        mx, my = pygame.mouse.get_pos()
            
        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        for b in range(len(buttons)):
            if b<3:
                buttons[b].Update(int(screen_w/9)-10, screen_h/10, (int(screen_w/3)+screen_w/9*(b), ((b//3)+1)*screen_h/5))
                
            else:
                buttons[b].Update(int(screen_w/9)-10, screen_h/10, (int(screen_w/3)+screen_w/9*(b-3), ((b//3)+1)*screen_h/5))
                
            if buttons[b].click:
                level = b + 1
                tutorial(level)
                break

        return_btn.Update(screen_w/4, screen_h/10, ((screen_w/2-screen_w/4)/2, 4*screen_h/5))

        if return_btn.click: break

        pygame.display.update()
        main_clock.tick(60)

def tutorial(level):
    pygame.mouse.set_visible(1)
    bg = pygame.image.load(os.path.join("assets","images","backgrounds","Circle1.png"))
    witch_sheet = pygame.image.load(os.path.join("assets","images","witches","B_witch_idle.png"))
    w,h = 32, 48

    # witch animation load
    sprites = [witch_sheet.subsurface((0,i,w,h)) for i in range(0,witch_sheet.get_height(),h)]
    current_sprite = 0
    witch = sprites[current_sprite]
    rect = witch.get_rect()
    screen_factor = 1/4

    return_btn = Button("Voltar")
    play_btn = Button("Jogar")
    while True:
        screen_w, screen_h = pygame.display.get_surface().get_size()
        screen.fill((40,0,65))
##        bg = pygame.transform.scale(bg, (screen_w, screen_h))
##        screen.blit(bg, (0,0))
        draw_text('Tutorial lvl '+str(level),(255,255,255),screen,screen_w/2,screen_h/20)

        # witch animation
        current_sprite += 0.2  # move through the frames, animating the sprite
        if current_sprite >= len(sprites): current_sprite = 0
        witch = sprites[int(current_sprite)]
        witch = pygame.transform.scale(witch, (screen_w*screen_factor, screen_h*screen_factor))
        screen.blit(witch, (screen_w/8-witch.get_width()/2, screen_h/2-witch.get_height()/2))

        mx, my = pygame.mouse.get_pos()

        return_btn.Update(screen_w/4, screen_h/10, ((screen_w/2-screen_w/4)/2, 4*screen_h/5))
        play_btn.Update(screen_w/4, screen_h/10, (screen_w/2+(screen_w/2-screen_w/4)/2, 4*screen_h/5))

        tut_font = pygame.font.Font('assets/fonts/FreePixel.ttf', int(screen_w/50))
        draw_text('Defenda o castelo dos inimigos que se aproximam fazendo o sinal em LIBRAS',
                  (255,255,255),screen,screen_w/5,2*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('da palavra acima da cabeça dos inimigos',
                  (255,255,255),screen,screen_w/5,3*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('Você tem apenas três vidas, e ganha quando fizer dez (10) pontos',
                  (255,255,255),screen,screen_w/5,4*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('Sente-se à uma distância do computador de forma que a câmera consiga ver',
                  (255,255,255),screen,screen_w/5,6*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('completamente os sinais que você fizer,e ajuste a câmera para que apareça',
                  (255,255,255),screen,screen_w/5,7*screen_h/20,font=tut_font, anchor='topleft')
        draw_text(' o seu corpo da cintura para cima.',
                  (255,255,255),screen,screen_w/5,8*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('Evite sentar-se de costas para janelas ou portas abertas, a alta luminosidade ',
                  (255,255,255),screen,screen_w/5,10*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('pode atrapalhar seu jogo',
                  (255,255,255),screen,screen_w/5,11*screen_h/20,font=tut_font, anchor='topleft')
        draw_text('Aperte ESC para pausar',
                  (255,255,255),screen,screen_w/5,13*screen_h/20,font=tut_font, anchor='topleft')

        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        if return_btn.click:
            break

        if play_btn.click:
            game(level)
                
        pygame.display.update()
        main_clock.tick(60)

##def video_tutorial(level):
##    pygame.mouse.set_visible(1)
##    bg = pygame.image.load(os.path.join("assets","images","backgrounds","Circle1.png"))
##    witch_sheet = pygame.image.load(os.path.join("assets","images","witches","B_witch_idle.png"))
##    w,h = 32, 48
##    vocab = LEVEL_VOCAB[level]
##
##    # load videos
##    videos = []
##    for word in vocab:
##        videos.append(Video(os.path.join("tutorial_videos",f"{word}.mp4")))
##
##    # witch animation load
##    sprites = [witch_sheet.subsurface((0,i,w,h)) for i in range(0,witch_sheet.get_height(),h)]
##    current_sprite = 0
##    witch = sprites[current_sprite]
##    rect = witch.get_rect()
##    screen_factor = 1/4
##
##    return_btn = Button("Voltar",size_mod = 0.6)
##    play_btn = Button("Jogar",size_mod = 0.6)
##    while True:
##        screen_w, screen_h = pygame.display.get_surface().get_size()
##        screen.fill((40,0,65))
####        bg = pygame.transform.scale(bg, (screen_w, screen_h))
####        screen.blit(bg, (0,0))
##
####        draw_text('Tutorial lvl '+str(level),(255,255,255),screen,screen_w/2,screen_h/20)
##
##        # witch animation
##        current_sprite += 0.2  # move through the frames, animating the sprite
##        if current_sprite >= len(sprites): current_sprite = 0
##        witch = sprites[int(current_sprite)]
##        witch = pygame.transform.scale(witch, (screen_w*screen_factor, screen_h*screen_factor))
##        witch = pygame.transform.flip(witch, 1, 0)
##        screen.blit(witch, (6*screen_w/8-witch.get_width()/2, screen_h/2-witch.get_height()/2))
##
##        mx, my = pygame.mouse.get_pos()
##
##        return_btn.Update(screen_w/4, screen_h/10, ((screen_w/2-screen_w/4)/2, 9*screen_h/10))
##        play_btn.Update(screen_w/4, screen_h/10, (screen_w/2+(screen_w/2-screen_w/4)/2, 9*screen_h/10))
##
##        tut_font = pygame.font.Font('assets/fonts/FreePixel.ttf', int(screen_w/50))
##        draw_text('Estes são os sinais que você vai encontrar no próximo nível',
##                  (255,255,255),screen,screen_w/2,screen_h/20,font=tut_font, anchor='center')
##
##        for word in vocab:
##            index = vocab.index(word)
##            pos = [[2,5,2,5],[1,1,3,3]]
##            draw_text(word,(255,255,255),screen, pos[0][index]*screen_w/8, pos[1][index]*screen_h/6,font=tut_font, anchor='topleft')
##
##            # draw videos
##        videos[2].set_size((screen_w/3, screen_h/5))
##        videos[2].draw(screen, (0,0))
##
##        for event in pygame.event.get():
##            if event.type == QUIT:
##                for vid in videos: vid.close()
##                cap.release()
##                pygame.quit()
##                sys.exit()
##                
##        if return_btn.click:
##            for vid in videos: vid.close()
##            break
##
##        if play_btn.click:
##            for vid in videos: vid.close()
##            game(level)
##                
##        pygame.display.update()
##        main_clock.tick(60)

def camera_adj():
    '''ajuste da câmera'''
    pygame.mouse.set_visible(1)
    sample_image = pygame.image.load(os.path.join("assets","images","blank_image.png"))

    play_btn = Button("Continuar")
    try:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    screen_w, screen_h = pygame.display.get_surface().get_size()
                    screen.fill((40,0,65))
                    tut_font = pygame.font.Font('assets/fonts/FreePixel.ttf', int(screen_w/50))
                    draw_text('Teste da câmera',(255,255,255),screen,screen_w/2,screen_h/20)
                    draw_text('Sente-se à uma distância do computador de forma que a câmera consiga ver',
                              (255,255,255),screen,screen_w/10,2*screen_h/20,font=tut_font, anchor='topleft')
                    draw_text('completamente suas mãos e o seu corpo da cintura para cima.Veja também se',
                              (255,255,255),screen,screen_w/10,3*screen_h/20,font=tut_font, anchor='topleft')
                    draw_text('todos os pontos coloridos aparecem sobre as suas mãos, como na imagem da esquerda.',
                              (255,255,255),screen,screen_w/10,4*screen_h/20,font=tut_font, anchor='topleft')
                    
                    play_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, 4*screen_h/5))

                    # Capture the image from webcam
                    ret,frame = cap.read()
                    image, hand_results = mediapipe_detection(frame, hands)
                    image, pose_results = mediapipe_detection(frame, pose)
                    draw_hand_landmarks(image, hand_results)
                    draw_pose_landmarks(image, pose_results)

                    # Scale and shows the player's image in screen
                    
                    camera_image = cv2_to_pygame(image)
                    camera_image = pygame.transform.scale(camera_image, (screen_w/2, screen_h/2))

                    sample_image = pygame.transform.scale(sample_image, (screen_w/2, screen_h/2))

                    w, h = camera_image.get_size()
                    screen.blit(camera_image, (screen_w/2, screen_h/2 - h/2))
                    screen.blit(sample_image, (0, screen_h/2 - h/2))

                    if play_btn.click:
                        break
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            cap.release()
                            pygame.quit()
                            sys.exit()
                    
                    main_clock.tick(60)
                    pygame.display.update()

    except Exception as e:
        print(e)
        quit_btn = Button("sair")
        
        while True:
            screen_w, screen_h = pygame.display.get_surface().get_size()
                
            screen.fill((0,0,0))
            draw_text('Erro com a câmera, tente reiniciar',(255,255,255),screen,20,20)
            quit_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, 4*screen_h/5))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == QUIT:
                    cap.release()
                    pygame.quit()
                    sys.exit()

            if quit_btn.click:
                cap.release()
                pygame.quit()
                sys.exit()

def game(level):
    # set game variables
    running = True
    state = None
    init_time = pygame.time.get_ticks()
    spawn_time = pygame.time.get_ticks()
    variance = 1 # Variability in spawing time between enemies, in seconds
    enemies = []
    lifes = 3
    points = 0
    
    # define detection variables
    last_detections = []
    frame_num = 0
    new_sequence = []
    performing_gesture = False
    stop_gesture_count = 0
    n = 0
    sequence, sentence = [], []

    # Load the background image
    bg = pygame.image.load(os.path.join("assets","images","backgrounds","windrise-background.png"))
    bg_tile = pygame.image.load(os.path.join("assets","images","backgrounds","BG_tiles","NonParallax.png"))
    castle = pygame.image.load(os.path.join("assets","images","castle_placeholder.png"))
    cloud = pygame.image.load(os.path.join("assets","images","cloud_placeholder.png"))
    grass_tile = pygame.image.load(os.path.join("assets","images","Tiles","Tile_02.png"))
    ground_tile = pygame.image.load(os.path.join("assets","images","Tiles","Tile_14.png"))
    hearth = pygame.image.load(os.path.join("assets","images","hearth_placeholder.png"))

    pygame.mouse.set_visible(0)
    pygame.display.set_caption('Tutorial Game')

    # create hands and pose detection model, and run the game loop
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while running and cap.isOpened():
                actual_time = pygame.time.get_ticks()

                # verify end game conditions
                if lifes == 0:
                    running = False
                    state = "lose"
                    break

                if points == 15:
                    running = False
                    state = "win"
                    break

                # get anchors the anchor and proportions from the screen
                screen_w, screen_h = pygame.display.get_surface().get_size()
                ground_level = 3*screen_h/4
                screen.fill((0,0,0))

                # Capture the image from webcam
                ret,frame = cap.read()
                image, hand_results = mediapipe_detection(frame, hands)
                image, pose_results = mediapipe_detection(frame, pose)
                draw_hand_landmarks(image, hand_results)
                draw_pose_landmarks(image, pose_results)

                # Draw objects on the screen
                # background
                bg_tile = pygame.transform.scale(bg_tile, (int(screen_w/4), ground_level))
                for i in range(4):
                    screen.blit(bg_tile, (bg_tile.get_width()*i,0))

                pygame.draw.rect(screen,(0,0,75), (0,0,screen_w,screen_h/14))

                # castle
                castle = pygame.transform.scale(castle, (screen_w/4, screen_h/3))
                screen.blit(castle, (screen_w - castle.get_width(), ground_level - castle.get_height()))
                # ground
                grass_tile = pygame.transform.scale(grass_tile, (screen_w/10, screen_w/10))
                ground_tile = pygame.transform.scale(ground_tile, (screen_w/10, screen_w/10))
                for i in range(10):
                    for j in range(5):
                        if j == 0:
                            screen.blit(grass_tile, (i*grass_tile.get_width(), ground_level+j*grass_tile.get_height()))
                        else:
                            screen.blit(ground_tile, (i*ground_tile.get_width(), ground_level+j*ground_tile.get_height()))
                
                # wizard
                
                # enemies
                to_remove = []
                for i in range(len(enemies)):
                    e = enemies[i]
                    e.Update(ground_level,screen)
                    if e.x >= screen_w - castle.get_width():
                        lifes -= 1
                        to_remove.append(i)

                # Delete enemies that reached the end of the screen
                to_remove.sort(reverse=True)
                for r in to_remove:
                    enemies.pop(r)
                    
                if actual_time - spawn_time > (5000 + (random()*variance*2 - variance)*100):
                    spawn_time = pygame.time.get_ticks()
                    enemies.append(Enemy(ground_level, level=level))
                
                # lifes
                hearth = pygame.transform.scale(hearth, (screen_w/15, screen_h/15))
                for l in range(lifes):
                    screen.blit(hearth, (screen_w - (l+1)*hearth.get_width(),0))

                draw_text("pontos = "+str(points),(255,0,0),screen, screen_w/10, screen_h/22,
                          font = pygame.font.SysFont(None, 20, bold= True),anchor='center')
                
                # Scale and shows the player's image in screen
                height, width, channels = image.shape
                camera_image = cv2_to_pygame(image)
                camera_image = pygame.transform.scale(camera_image, (screen_w/4, screen_h/4))
                screen.blit(camera_image, (0, 3*screen_h/4))

                # Signal detection and classification logic
                keypoints, handedness = pose_hands_extract_keypoints(hand_results, pose_results)
                
                if not handedness == ['none','none']:
                    stop_gesture_count = 0
                    if performing_gesture == False:
                        performing_gesture = True
                        new_sequence = []
                        
                if handedness == ['none', 'none']:stop_gesture_count += 1
                if stop_gesture_count == 10:performing_gesture = False
                if performing_gesture and len(new_sequence) < 30:new_sequence.append(keypoints)
                elif performing_gesture == False and new_sequence != []:
                    sequence = np.array(new_sequence)

                    ### aply fastdtw
                    distance, path = fastdtw(np.arange(0,30,1), np.arange(0,len(sequence),1)) #, dist=euclidean)
                    new_fastdtw= np.zeros([len(path), 258])
                    
                    for i in range(len(path)):
                        new_fastdtw[i,:] = sequence[path[i][1], :]
                    sequence = new_fastdtw
                    ### end of fastdtw
                    
                    res = cnn.model.predict(np.reshape(sequence,(-1,30,258,1)), verbose=0)
                    now_sign = np.argmax(res[0])
                    perc = max(res[0])
                    last_detections.append(now_sign)
                    if len(last_detections) > 10: last_detections = last_detections[-30:]
                    
                    text = 'Prediction: %.3g%% %s'%(perc * 100, actions[now_sign])
                    draw_text(text,(0,0,255),screen, screen_w/2, screen_h/22, font = pygame.font.SysFont(None, 20, bold= True),anchor='center')
                    
                    # Delete enemies if the sign performed is equal to enemy word
                    to_remove = []
                    for j in range(len(enemies)):
                        if actions[now_sign] == enemies[j].word:
                            to_remove.append(j)

                    to_remove.sort(reverse=True)
                    for r in to_remove:
                        enemies.pop(r)
                        points += 1

                    new_squence = []

                # Debug
                ground_line = pygame.draw.line(screen, (255,0,0), (0,ground_level), (screen_w,ground_level))
##                test_rect = pygame.Rect(0,ground_level,screen_w, screen_h/3)
##                pygame.draw.rect(screen, (150,0,0), test_rect)
                
                mx, my = pygame.mouse.get_pos()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        cap.release()
                        pygame.quit()
                        sys.exit()

                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            running = pause()
                            
                main_clock.tick(60)
                pygame.display.update()

            print("exit game")
            if state != None: end_game(state, level)

def end_game(state, level):
    pygame.mouse.set_visible(1)

    if state == 'lose':
        play_btn = Button('Jogar Novamente')
    else:
        replay_btn = Button('Próximo nível')
    return_btn = Button('Voltar para o menu')
    while True:
        screen_w, screen_h = pygame.display.get_surface().get_size()
        screen.fill((0,0,0))

        mx, my = pygame.mouse.get_pos()

        play_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, screen_h/5))
        return_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, 2*screen_h/5))
        
        if state == "lose":
            draw_text('você perdeu :(',(255,255,255),screen,20,20)
        if state == "win":
            draw_text('você ganhou :)',(255,255,255),screen,20,20)

        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        if play_btn.click:
            if state == "lose":
                game(level)
                break
            if state == "win":
                game(level + 1)
                break

        if return_btn.click: break

        pygame.display.update()
        main_clock.tick(60)

def pause():
    pygame.mouse.set_visible(1)
    resume_btn = Button('Continuar')
    quit_btn = Button('Voltar para o menu')
    
    while True:
        screen_w, screen_h = pygame.display.get_surface().get_size()
        screen.fill((0,0,0))
        draw_text('Jogo Pausado',(255,255,255),screen,20,20)

        mx, my = pygame.mouse.get_pos()

        resume_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, screen_h/5))
        quit_btn.Update(screen_w/4, screen_h/10, (screen_w/2-screen_w/8, 2*screen_h/5))
        
        
        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        if resume_btn.click:
            return True

        if quit_btn.click:
            return False
                
        pygame.display.update()
        main_clock.tick(60)


if __name__ == "__main__":
    camera_adj()
    main_menu()

