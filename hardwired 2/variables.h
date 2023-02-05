const char* ssid = "pi";
const char* password = "Virat@kohli18";
const uint16_t serverPort = 11411;   

#define left_sensor   D8
#define right_sensor  D7
#define left_pwm      D6
#define left_motor1   D3
#define left_motor2   D4
#define right_pwm     D5
#define right_motor1  D2
#define right_motor2  D1

bool cond = false;
float kp = 0.07;
float ki = 0.008;
float kd = 0.06;
int thresh = 10;
int s = 100; 
 
float error, prev_error, P, I, D;
int PID;
