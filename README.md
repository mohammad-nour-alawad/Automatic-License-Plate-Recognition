# YOLO4_Automatic_License_Plate_Recognition

THIS PROJECT IS MADE TO WORK ON A *KHADAS VIM3 AI EDGE DEVICE*.

Achived a REALTIME processing at 21 frames per second.

In this project I am deploying a YOLOV4 model for License Plate Recognition of THE Indian Numbers (The ones used in the Arabic Language).

*NOTE*: I also send a stream of the detection results UDP port, so I can catch these packets and decode it to see a LIVE STREAM of the detection results. 

The Process:
- find the car.
- find the numbers (six numbers) and order them depending on the X axis positions.
- save the detection to a MongoDB.
- snap a photo of the plate.
- compress the photo.
- encode the photo as a string and save it to the cloud.

All these results are also connected to a FULTTER app for monitoring the live stream, and cheching the results.
