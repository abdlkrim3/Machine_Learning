import serial.tools.list_ports
import serial
import time


print("Looking for COM ports...")
ports = serial.tools.list_ports.comports(include_links=False)


if (len(ports) != 0): # if a COM port has been found

    print (str(len(ports)) + " active ports has been found") 

    for count, port in enumerate(ports, start=1) :  # Show name of all ports
        print(str(count) + ' : ' + port.device)

    Choosed_port = int(input('Choose the used port: '))

    print('1: 9600   2: 38400    3: 115200')

    baud = int(input('Choose a baud rate: '))

    if (baud == 1):
        baud = 9600
    if (baud == 2):
        baud = 38400
    if (baud == 3):
        baud = 115200

    port = ports[Choosed_port - 1].device

    # Connecting to the Pololu Zumo robot
    print('Connecting to ' + str(port) + ' with a baud rate of ' + str(baud))
    robot = serial.Serial(port, baud, timeout=1)
    
else: 
    print("No serial port has been found")

def Move(middle, width, th):
    if width/2 - middle > th/2:
        Left()
    elif width/2 - middle < - th/2:
        Right()
    else:
        Forward()


def Forward():
    print("Direction : Forward")
    robot.write(bytes("F", 'utf-8'))

def Backward():
    print("Direction : Backward")
    robot.write(bytes("B", 'utf-8'))

def Right():
    print("Direction : Right")
    robot.write(bytes("R", 'utf-8'))

def Left():
    print("Direction : Left")
    robot.write(bytes("L", 'utf-8'))

def Stop():
    print("Direction : Stop")
    robot.write(bytes("0", 'utf-8'))
