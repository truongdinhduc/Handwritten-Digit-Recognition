import pygame, sys
from keras.models import load_model
import numpy as np
from pygame.locals import *
import cv2

WINDOW_SIZE = (640, 480)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

LINE_WIDTH = 5

PADDING = (10,10)

MODEL = load_model('HandwrittenDigitRecognition.h5')

LABELS = {
    0:'Zero',
    1:'One',
    2:'Two',
    3:'Three',
    4:'Four',
    5:'Five',
    6:'Six',
    7:'Seven',
    8:'Eight',
    9:'Nine'
}

pixel_x = []
pixel_y = []

pygame.init()

DISPLAYSURFACE = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('Digit Board')
FONT = pygame.font.Font('freesansbold.ttf', 20)

iswriting = False

while True:
    for event in pygame.event.get():

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_c:
                DISPLAYSURFACE.fill(BLACK)

        if event.type == MOUSEMOTION and iswriting:
            x, y = event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (x,y), LINE_WIDTH, 0)
            pixel_x.append(x)
            pixel_y.append(y)

        if event.type == MOUSEBUTTONDOWN:
            pixel_x = []
            pixel_y = []
            iswriting = True
            
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            pixel_x = sorted(pixel_x)
            pixel_y = sorted(pixel_y)

            min_x = max(0, pixel_x[0] - LINE_WIDTH)
            max_x = min(pixel_x[-1] + LINE_WIDTH, WINDOW_SIZE[0])
            min_y = max(0, pixel_y[0] - LINE_WIDTH)
            max_y = min(pixel_y[-1] + LINE_WIDTH, WINDOW_SIZE[1])

            pygame.draw.rect(
                DISPLAYSURFACE, 
                RED, 
                pygame.Rect(min_x, min_y, max_x-min_x, max_y-min_y), 
                1, 
                1
            )

            img_arr = np.array(pygame.PixelArray(DISPLAYSURFACE))[min_x:max_x, min_y:max_y].T.astype(np.float32)    
            image = cv2.resize(img_arr, (28,28))
            image = np.pad(image, PADDING, 'constant', constant_values = 0)
            image = cv2.resize(image, (28,28))/255.0

            result = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
            text = FONT.render(result, True, RED, BLACK)
            textRect = text.get_rect()
            textRect.center = ((pixel_x[0]+pixel_x[-1])/2.0, (pixel_y[0]+pixel_y[-1])/2.0)
            DISPLAYSURFACE.blit(text, textRect)
        
        pygame.display.update()