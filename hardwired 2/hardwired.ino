#include <ESP8266WiFi.h>    
#include <ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include "variables.h"

IPAddress server(192,168,66,40);
void setupWiFi(){
  Serial.print("Connecting to ");
  Serial.print(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status()!=WL_CONNECTED) delay(500);
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());}

ros::NodeHandle nh;
void yaw_related_cb(const std_msgs::Float32& msg){
  error = msg.data;
}
void cond_related_cb(const std_msgs::Bool& msg){
  cond = msg.data;
}
ros::Subscriber<std_msgs::Float32> sub1("yaw", &yaw_related_cb);
ros::Subscriber<std_msgs::Bool> sub2("condition", &cond_related_cb);

void setup(){
  Serial.begin(115200); delay(100);
  setupWiFi(); delay(100);
  error=prev_error=P=I=D=PID=0; delay(100);
  nh.getHardware()->setConnection(server, serverPort);
  nh.initNode();
  nh.subscribe(sub1);
  nh.subscribe(sub2);
  pinMode(left_sensor, INPUT); pinMode(right_sensor, INPUT);
  pinMode(left_motor1, OUTPUT); pinMode(right_motor1, OUTPUT);
  pinMode(left_motor2, OUTPUT); pinMode(right_motor2, OUTPUT);
}

void loop(){
  if(cond==true){
    if(error>-thresh && error<thresh){
      set_speed(s, s);
    }
    else{
      set_speed(0, 0);
      P = error;
      I = I + error;
      D = error - prev_error;
      prev_error = error;
      PID = kp*P + ki*I + kd*D;
      set_speed(constrain(-PID, -100, 100), constrain(PID, -100, 100));
    }
  }
  else if(cond==false){
    set_speed(0, 0);
  }
  nh.spinOnce();
}

void set_speed(int l_speed, int r_speed){
  analogWrite(left_pwm, abs(l_speed)); analogWrite(right_pwm, abs(r_speed));
  digitalWrite(left_motor1, (l_speed>=0)?HIGH:LOW); digitalWrite(left_motor2, (l_speed>=0)?HIGH:LOW);
  digitalWrite(right_motor1, (r_speed>=0)?HIGH:LOW); digitalWrite(right_motor2, (r_speed>=0)?HIGH:LOW);
}
