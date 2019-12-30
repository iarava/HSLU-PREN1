# Read and Print VL6180x distance sensor range every 0.01 second.
# Author: Michael Ruppen
import time

import board
import busio

import adafruit_vl6180x

# Create I2C bus.
i2c = busio.I2C(board.SCL, board.SDA)

# Create sensor instance.
#s = adafruit_vl6180x.VL6180X(i2c, adafruit_vl6180x.__VL6180X_I2C_SLAVE_DEVICE_ADDRESS = 0x0041)

sensor = adafruit_vl6180x._VL6180X_DEFAULT_I2C_ADDR = 0x0040
sensor1 = adafruit_vl6180x.VL6180X(i2c)
sensor1.__VL6180X_I2C_SLAVE_DEVICE_ADDRESS = 0x0040
sensor1.get_register(0x0040)
sensor_top = adafruit_vl6180x.VL6180X(i2c, 0x0029)

# sensor_bottom = adafruit_vl6180x.VL6180X(i2c, address=27)
# sensor_weight = adafruit_vl6180x.VL6180X(i2c, address=27)

# Main loop prints the range and lux every second:
while True:
    # Read the range in millimeters and print it.
    range_mm = sensor_top.range
    if(range_mm != 255):
        print("yeah got it")

    print('Range: {0}mm'.format(range_mm))

    # Delay for a second.
    time.sleep(0.1)


# maybe for UART
# uart_read = busio.UART.readline()
# uart_wright = busio.UART.write()