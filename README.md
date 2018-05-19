# Pattern Recongintion to Control PC 

This a College Project in Tanta University, computer and control Department 

## Idea 
draw a specific symbols on the screen using a special color attached to your finger 
where there are several Symbols and each one perform some functonality in PC

This is done in three phases :
* Pattern Recognition
* Classification 
* Action performing 

## Pattern Recognition Phase 

Using Opencv to detect the special color in Hand once it appears in front of camera 
Then make contineous line tracing the finger until it gets far away the screen 
Then save this drawn figure as img with background color Black and White color in front 

## Classification Phase 

using Convelutional Neural Network (CNN) to classify the output img of detection phase 
we trained a model on 3500 images belong to 7 classes (500 for each class)
it outputs the class number of the image.
### This is the architecture of nueral Network we Used
![CNN Model](https://github.com/dar4kamal/Pattern-Recognition-to-Control-Pc/blob/master/CNN-example-block-diagram.jpg?raw=true)

## Action performing Phase 

using keyboard we perform some action like 

![ALT](https://github.com/dar4kamal/Pattern-Recognition-to-Control-Pc/blob/master/action/alt-Symbol.PNG?raw=true)
![UP](https://github.com/dar4kamal/Pattern-Recognition-to-Control-Pc/blob/master/action/up-Symbol.PNG?raw=true)
![MAXIMIZE](https://github.com/dar4kamal/Pattern-Recognition-to-Control-Pc/blob/master/action/maxi-Symbol.PNG?raw=true)
![DOWN](https://github.com/dar4kamal/Pattern-Recognition-to-Control-Pc/blob/master/action/down-Symbol.PNG?raw=true)
![MINIMIZE](https://github.com/dar4kamal/Pattern-Recognition-to-Control-Pc/blob/master/action/mini-Symbol.PNG?raw=true)
![MUTE](https://github.com/dar4kamal/Pattern-Recognition-to-Control-Pc/blob/master/action/mute-Symbol.PNG?raw=true)
![TAB](https://github.com/dar4kamal/Pattern-Recognition-to-Control-Pc/blob/master/action/tab-Symbol.PNG?raw=true)

## Team Memebrs

* Mostafa Kamal
* Mostafa Tarek 
* [Abdulrahman Mosad](https://github.com/AbdulrahmanMosad)
* Ibrahim Shaaban
* [Abdel7y EL-Nakeep](https://github.com/Abdel7y)
* Ahmed Rezk
* Nourhan Ayman 
* Ahmed ElSaed
