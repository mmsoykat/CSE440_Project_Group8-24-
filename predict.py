from ultralytics import YOLO

model=YOLO("best.pt")
#for image
#model.predict(source="1.png",show=True, save=True, conf=0.7, line_width=2,save_crop=True,save_txt=True,show_labels=True,show_conf=True,classes=[0,1,2])

#for video 
model.predict(source="2.mp4",show=True, save=True, conf=0.7, line_width=2,save_crop=False,save_txt= False,show_labels=True,show_conf=True,classes=[0,1,2])


