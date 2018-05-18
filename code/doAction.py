
from keyboard import *
import subprocess

alt = False

def act(Class):
    
    global alt 

    subprocess.Popen(r"../action/actionControl.exe")
    
    VK_VOLUME_MUTE = 0xAD
    VK_VOLUME_DOWN = 0xAE
    VK_VOLUME_UP = 0xAF
    VK_TAB = 0x09
    VK_ALT = 0x12
    VK_CTRL = 0x11
    VK_UP = 0x26
    VK_DOWN = 0x28
    
    try:
        if Class == 0:
            if alt:
                Keyboard.keyUp(VK_ALT)
                alt = False
                
            else:                
                Keyboard.keyDown(VK_ALT)
                alt = True            
            return True                

        elif Class == 1:
            Keyboard.key(VK_VOLUME_DOWN)
            return True
        
        elif Class == 2:            
            
            Keyboard.keyDown(VK_CTRL)
            Keyboard.keyDown(VK_UP)
            time.sleep(1)
            Keyboard.keyUp(VK_CTRL)
            Keyboard.keyUp(VK_UP)
            return True

        
        elif Class == 3:
            
            Keyboard.keyDown(VK_CTRL)
            Keyboard.keyDown(VK_DOWN)
            time.sleep(1)
            Keyboard.keyUp(VK_CTRL)
            Keyboard.keyUp(VK_DOWN)
            return True

        elif Class == 4:
            Keyboard.key(VK_VOLUME_MUTE)            
            return True
        
        elif Class == 5:
            Keyboard.key(VK_TAB)
            return True
        
        elif Class == 6:
            Keyboard.key(VK_VOLUME_UP)
            return True
        
        else :
            print("Not in actions")
            return False
    
    except:
        print("Please Enter a number from 1 --> 7")                
        return None
        