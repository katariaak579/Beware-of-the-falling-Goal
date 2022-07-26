import cv2
from cv2 import FONT_HERSHEY_DUPLEX
import numpy as np
from cv2 import COLOR_BGR2GRAY
from cv2 import THRESH_BINARY
from collections import deque


# Classes Used 
class BgExtract:
    def __init__(self,width,height,scale,maxlen=10): 
        self.maxlen=maxlen
        self.width=width//scale
        self.scale=scale
        self.height=height//scale
        self.buffer=deque(maxlen=maxlen)
        self.bg=None
    
    def cal_if_notfull(self):
        self.bg=np.zeros((self.height,self.width,),dtype='float32')
        for i in self.buffer:
            self.bg+=i
        self.bg//=len(self.buffer)

    def cal_if_full(self,old,new):
        self.bg-=old/self.maxlen
        self.bg+=new/self.maxlen

    def add_frame(self,frame):
        if self.maxlen>len(self.buffer):
            self.buffer.append(frame)
            self.cal_if_notfull()
        else:
            old=self.buffer.popleft()
            self.buffer.append(frame)
            self.cal_if_full(old,frame)

    def output_frame(self):       
        return self.bg.astype('uint8')

    def apply(self,frame):
        down_scale=cv2.resize(frame,(self.width,self.height))
        # Creating the gray frame to act as background 
        gray=cv2.cvtColor(down_scale,COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(5,5),0)
    
        self.add_frame(gray)
        absdifference=cv2.absdiff(gray,self.output_frame())
        _,maskabs=cv2.threshold(absdifference,50,255,THRESH_BINARY)
        return cv2.resize(maskabs,(self.width*self.scale,self.height*self.scale))

    def get_frame(self,frame):
        down_scale=cv2.resize(frame,(self.width,self.height))
        # Creating the gray frame to act as background 
        gray=cv2.cvtColor(down_scale,COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(5,5),0)
    
        # self.add_frame(gray)
        absdifference=cv2.absdiff(gray,self.output_frame())
        _,maskabs=cv2.threshold(absdifference,50,255,THRESH_BINARY)
        return cv2.resize(maskabs,(self.width*self.scale,self.height*self.scale))


class Game:
    def __init__(self,width,height,size=50):
        self.width=width
        self.height=height
        self.size=size
        self.img=cv2.imread("./Goal.jpg")
        self.img=cv2.resize(self.img,(self.size,self.size))
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        
        self.mask=cv2.threshold(gray,40,255,cv2.THRESH_BINARY)[1]
        # print(self.mask)
        self.x=np.random.randint(0,self.width-self.size)
        self.y=0
        self.speed=np.random.randint(5,15)
        self.score=0
        # return self.mask

    def add_frame(self,frame):
        roi=frame[self.y:self.y+self.size,self.x:self.x+self.size]
        roi[np.where(self.mask)]=0
        roi+=self.img

    def new_drop(self,fg_frame):
        self.y+=self.speed
        if self.y+self.size>=self.height:
            self.score+=1
            self.speed=np.random.randint(5,15)
            self.y=0
            self.x=np.random.randint(0,self.width-self.size)
        
        roi=fg_frame[self.y:self.y+self.size,self.x:self.x+self.size]

        check_if_it=np.any(roi[np.where(self.mask)])
        if check_if_it:
            self.score-=1
            self.speed=np.random.randint(5,15)
            self.y=0
            self.x=np.random.randint(0,self.width-self.size)
        
        return check_if_it

# Variables 
width=640
height=480
scale_down=2


cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

bg_buffer=BgExtract(width,height,scale_down)
game=Game(width,height)

while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    fg_frame=bg_buffer.apply(frame)
    play=f"Click s to Start the Game"
    cv2.putText(frame,play,(10,40),FONT_HERSHEY_DUPLEX,1.0,(255,0,0),2)
    cv2.imshow("Get Background",frame)

    if cv2.waitKey(1) == ord('s'):
        break

while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    fg_frame=bg_buffer.get_frame(frame)
    hit=game.new_drop(fg_frame)
    game.add_frame(frame)

    if hit:
        frame[:,:,2]=255

    text = f"Score: {game.score}"
    cv2.putText(frame,text,(10,30),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,0,0),2)
    # cv2.imshow("Game",fg_frame)
    cv2.imshow("Actual",frame)
   
    if cv2.waitKey(1) == ord('q'):
        break
