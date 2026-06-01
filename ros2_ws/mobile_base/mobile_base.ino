#include <Encoder.h>

// ===== 모터 핀 정의: 네가 확인한 정상 동작 핀 기준 =====
#define PWMA 4
#define DIRA1 A5
#define DIRA2 A4

#define PWMB 6
#define DIRB1 43
#define DIRB2 42

#define PWMC 11
#define DIRC1 35
#define DIRC2 34

#define PWMD 7
#define DIRD1 37
#define DIRD2 36

// ===== Encoder 핀 =====
// 실제 확인된 encoder 커넥터 기준
Encoder encA(18, 31);   // 실제 RF
Encoder encB(19, 38);   // 실제 LF, 부호 반대
Encoder encC(3, 49);    // 실제 RR
Encoder encD(2, A1);    // 실제 LR, 부호 반대

// ===== 속도 설정 =====
int Motor_PWM = 30;

// encoder 출력 주기
const unsigned long ENC_PERIOD_MS = 100;
unsigned long lastEncMs = 0;

// 명령 timeout: 키 입력 끊기면 자동 정지
const unsigned long CMD_TIMEOUT_MS = 800;
unsigned long lastCmdMs = 0;

// ===== 모터 제어 매크로: 네 기존 코드 그대로 =====
#define MOTORA_FORWARD(pwm)    do{digitalWrite(DIRA1,HIGH); digitalWrite(DIRA2,LOW);  analogWrite(PWMA,pwm);}while(0)
#define MOTORA_BACKOFF(pwm)    do{digitalWrite(DIRA1,LOW);  digitalWrite(DIRA2,HIGH); analogWrite(PWMA,pwm);}while(0)
#define MOTORA_STOP()          do{digitalWrite(DIRA1,LOW);  digitalWrite(DIRA2,LOW);  analogWrite(PWMA,0);}while(0)

#define MOTORB_FORWARD(pwm)    do{digitalWrite(DIRB1,LOW);  digitalWrite(DIRB2,HIGH); analogWrite(PWMB,pwm);}while(0)
#define MOTORB_BACKOFF(pwm)    do{digitalWrite(DIRB1,HIGH); digitalWrite(DIRB2,LOW);  analogWrite(PWMB,pwm);}while(0)
#define MOTORB_STOP()          do{digitalWrite(DIRB1,LOW);  digitalWrite(DIRB2,LOW);  analogWrite(PWMB,0);}while(0)

#define MOTORC_FORWARD(pwm)    do{digitalWrite(DIRC1,HIGH); digitalWrite(DIRC2,LOW);  analogWrite(PWMC,pwm);}while(0)
#define MOTORC_BACKOFF(pwm)    do{digitalWrite(DIRC1,LOW);  digitalWrite(DIRC2,HIGH); analogWrite(PWMC,pwm);}while(0)
#define MOTORC_STOP()          do{digitalWrite(DIRC1,LOW);  digitalWrite(DIRC2,LOW);  analogWrite(PWMC,0);}while(0)

#define MOTORD_FORWARD(pwm)    do{digitalWrite(DIRD1,LOW);  digitalWrite(DIRD2,HIGH); analogWrite(PWMD,pwm);}while(0)
#define MOTORD_BACKOFF(pwm)    do{digitalWrite(DIRD1,HIGH); digitalWrite(DIRD2,LOW);  analogWrite(PWMD,pwm);}while(0)
#define MOTORD_STOP()          do{digitalWrite(DIRD1,LOW);  digitalWrite(DIRD2,LOW);  analogWrite(PWMD,0);}while(0)

// ===== 기본 이동 =====
void moveForward() {
  MOTORA_FORWARD(Motor_PWM);
  MOTORB_FORWARD(Motor_PWM);
  MOTORC_FORWARD(Motor_PWM);
  MOTORD_FORWARD(Motor_PWM);
}

void moveBackward() {
  MOTORA_BACKOFF(Motor_PWM);
  MOTORB_BACKOFF(Motor_PWM);
  MOTORC_BACKOFF(Motor_PWM);
  MOTORD_BACKOFF(Motor_PWM);
}

void stopMotors() {
  MOTORA_STOP();
  MOTORB_STOP();
  MOTORC_STOP();
  MOTORD_STOP();
}

// ===== 회전 =====
void turnLeft() {
  MOTORA_BACKOFF(Motor_PWM);
  MOTORB_FORWARD(Motor_PWM);
  MOTORC_BACKOFF(Motor_PWM);
  MOTORD_FORWARD(Motor_PWM);
}

void turnRight() {
  MOTORA_FORWARD(Motor_PWM);
  MOTORB_BACKOFF(Motor_PWM);
  MOTORC_FORWARD(Motor_PWM);
  MOTORD_BACKOFF(Motor_PWM);
}

// ===== 메카넘 평행이동 =====
void strafeLeft() {
  MOTORA_BACKOFF(Motor_PWM);
  MOTORB_FORWARD(Motor_PWM);
  MOTORC_FORWARD(Motor_PWM);
  MOTORD_BACKOFF(Motor_PWM);
}

void strafeRight() {
  MOTORA_FORWARD(Motor_PWM);
  MOTORB_BACKOFF(Motor_PWM);
  MOTORC_BACKOFF(Motor_PWM);
  MOTORD_FORWARD(Motor_PWM);
}

// ===== 대각선 =====
void diagFrontLeft() {
  MOTORA_STOP();
  MOTORB_FORWARD(Motor_PWM);
  MOTORC_FORWARD(Motor_PWM);
  MOTORD_STOP();
}

void diagFrontRight() {
  MOTORA_FORWARD(Motor_PWM);
  MOTORB_STOP();
  MOTORC_STOP();
  MOTORD_FORWARD(Motor_PWM);
}

void diagBackLeft() {
  MOTORA_BACKOFF(Motor_PWM);
  MOTORB_STOP();
  MOTORC_STOP();
  MOTORD_BACKOFF(Motor_PWM);
}

void diagBackRight() {
  MOTORA_STOP();
  MOTORB_BACKOFF(Motor_PWM);
  MOTORC_BACKOFF(Motor_PWM);
  MOTORD_STOP();
}

// ===== Encoder 출력 =====
// 최종 출력: ENC LF RF LR RR
void publishEncoder() {
  long rawA = encA.read();     // 실제 RF
  long rawB = -encB.read();    // 실제 LF
  long rawC = encC.read();     // 실제 RR
  long rawD = -encD.read();    // 실제 LR

  long lf = rawB;
  long rf = rawA;
  long lr = rawD;
  long rr = rawC;

  Serial.print("ENC ");
  Serial.print(lf);
  Serial.print(" ");
  Serial.print(rf);
  Serial.print(" ");
  Serial.print(lr);
  Serial.print(" ");
  Serial.println(rr);
}

// ===== 명령 처리 =====
void handleCommand(char cmd) {
  if (cmd == '\n' || cmd == '\r') return;

  switch (cmd) {
    case 'w': case 'W':
      moveForward();
      lastCmdMs = millis();
      break;

    case 's': case 'S':
      moveBackward();
      lastCmdMs = millis();
      break;

    case 'a': case 'A':
      turnLeft();
      lastCmdMs = millis();
      break;

    case 'd': case 'D':
      turnRight();
      lastCmdMs = millis();
      break;

    case 'q': case 'Q':
      strafeLeft();
      lastCmdMs = millis();
      break;

    case 'e': case 'E':
      strafeRight();
      lastCmdMs = millis();
      break;

    case 'r': case 'R':
      diagFrontLeft();
      lastCmdMs = millis();
      break;

    case 't': case 'T':
      diagFrontRight();
      lastCmdMs = millis();
      break;

    case 'f': case 'F':
      diagBackLeft();
      lastCmdMs = millis();
      break;

    case 'g': case 'G':
      diagBackRight();
      lastCmdMs = millis();
      break;

    case 'x': case 'X':
      stopMotors();
      break;

    case 'l': case 'L':
      Motor_PWM += 10;
      if (Motor_PWM > 255) Motor_PWM = 255;
      break;

    case 'm': case 'M':
      Motor_PWM -= 10;
      if (Motor_PWM < 0) Motor_PWM = 0;
      break;
  }
}

// ===== 초기 설정 =====
void setup() {
  Serial.begin(115200);

  pinMode(PWMA, OUTPUT);
  pinMode(DIRA1, OUTPUT);
  pinMode(DIRA2, OUTPUT);

  pinMode(PWMB, OUTPUT);
  pinMode(DIRB1, OUTPUT);
  pinMode(DIRB2, OUTPUT);

  pinMode(PWMC, OUTPUT);
  pinMode(DIRC1, OUTPUT);
  pinMode(DIRC2, OUTPUT);

  pinMode(PWMD, OUTPUT);
  pinMode(DIRD1, OUTPUT);
  pinMode(DIRD2, OUTPUT);

  stopMotors();

  lastCmdMs = millis();
  lastEncMs = millis();

  Serial.println("READY TELEOP_WITH_ENCODER");
  Serial.println("FORMAT: ENC LF RF LR RR");
}

// ===== 메인 루프 =====
void loop() {
  unsigned long now = millis();

  while (Serial.available()) {
    char cmd = Serial.read();
    handleCommand(cmd);
  }

  // 명령 끊기면 자동 정지
  if (now - lastCmdMs > CMD_TIMEOUT_MS) {
    stopMotors();
  }

  // encoder 출력
  if (now - lastEncMs >= ENC_PERIOD_MS) {
    lastEncMs = now;
    publishEncoder();
  }
}
