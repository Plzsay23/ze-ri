#include <Encoder.h>

// ===== 모터 핀 정의 =====
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
Encoder encA(18, 31);   // 실제 RF
Encoder encB(19, 38);   // 실제 LF, 부호 반대
Encoder encC(3, 49);    // 실제 RR
Encoder encD(2, A1);    // 실제 LR, 부호 반대

// ===== 속도 설정 =====
// 기존 30은 Linux/Python 테스트에서 너무 약할 수 있으므로 일단 180으로 고정
int Motor_PWM = 180;

// encoder 출력 주기
const unsigned long ENC_PERIOD_MS = 100;
unsigned long lastEncMs = 0;

// 명령 timeout
const unsigned long CMD_TIMEOUT_MS = 800;
unsigned long lastCmdMs = 0;

// ===== ROS Twist 명령용 설정 =====
const float MAX_LINEAR_X = 0.30;
const float MAX_LINEAR_Y = 0.30;
const float MAX_ANGULAR_Z = 1.50;

const int MIN_ACTIVE_PWM = 60;
const int MAX_CMD_PWM = 180;

// ===== 모터 제어 매크로 =====
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

// ===== ROS Twist 형식 처리 =====
// 입력 예: V 0.100 0.000 0.000
int velocityToPwm(float value) {
  float a = fabs(value);

  if (a < 0.03) {
    return 0;
  }

  int pwm = (int)(a * MAX_CMD_PWM);

  if (pwm > 0 && pwm < MIN_ACTIVE_PWM) {
    pwm = MIN_ACTIVE_PWM;
  }

  if (pwm > MAX_CMD_PWM) {
    pwm = MAX_CMD_PWM;
  }

  return pwm;
}

void driveMotorA(float value) {
  int pwm = velocityToPwm(value);

  if (pwm == 0) {
    MOTORA_STOP();
  } else if (value > 0.0) {
    MOTORA_FORWARD(pwm);
  } else {
    MOTORA_BACKOFF(pwm);
  }
}

void driveMotorB(float value) {
  int pwm = velocityToPwm(value);

  if (pwm == 0) {
    MOTORB_STOP();
  } else if (value > 0.0) {
    MOTORB_FORWARD(pwm);
  } else {
    MOTORB_BACKOFF(pwm);
  }
}

void driveMotorC(float value) {
  int pwm = velocityToPwm(value);

  if (pwm == 0) {
    MOTORC_STOP();
  } else if (value > 0.0) {
    MOTORC_FORWARD(pwm);
  } else {
    MOTORC_BACKOFF(pwm);
  }
}

void driveMotorD(float value) {
  int pwm = velocityToPwm(value);

  if (pwm == 0) {
    MOTORD_STOP();
  } else if (value > 0.0) {
    MOTORD_FORWARD(pwm);
  } else {
    MOTORD_BACKOFF(pwm);
  }
}

void driveMecanum(float vx, float vy, float wz) {
  float nx = vx / MAX_LINEAR_X;
  float ny = vy / MAX_LINEAR_Y;
  float nw = wz / MAX_ANGULAR_Z;

  nx = constrain(nx, -1.0, 1.0);
  ny = constrain(ny, -1.0, 1.0);
  nw = constrain(nw, -1.0, 1.0);

  // A = RF
  // B = LF
  // C = RR
  // D = LR
  float lf = nx + ny + nw;
  float rf = nx - ny - nw;
  float lr = nx - ny + nw;
  float rr = nx + ny - nw;

  float maxAbs = max(max(fabs(lf), fabs(rf)), max(fabs(lr), fabs(rr)));

  if (maxAbs > 1.0) {
    lf /= maxAbs;
    rf /= maxAbs;
    lr /= maxAbs;
    rr /= maxAbs;
  }

  driveMotorA(rf);
  driveMotorB(lf);
  driveMotorC(rr);
  driveMotorD(lr);
}

void handleVelocityLine(char *line) {
  float vx = 0.0;
  float vy = 0.0;
  float wz = 0.0;

  int matched = sscanf(line, "V %f %f %f", &vx, &vy, &wz);

  Serial.print("RX_V matched=");
  Serial.print(matched);
  Serial.print(" vx=");
  Serial.print(vx, 3);
  Serial.print(" vy=");
  Serial.print(vy, 3);
  Serial.print(" wz=");
  Serial.println(wz, 3);

  if (matched == 3) {
    driveMecanum(vx, vy, wz);
    lastCmdMs = millis();
  }
}

// ===== 문자 명령 처리 =====
void handleCommand(char cmd) {
  if (cmd == '\n' || cmd == '\r') return;

  Serial.print("RX_CHAR ");
  Serial.print(cmd);
  Serial.print(" PWM=");
  Serial.println(Motor_PWM);

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
      lastCmdMs = millis();
      break;

    case 'l': case 'L':
      Motor_PWM += 10;
      if (Motor_PWM > 255) Motor_PWM = 255;
      lastCmdMs = millis();
      Serial.print("PWM_UP ");
      Serial.println(Motor_PWM);
      break;

    case 'm': case 'M':
      Motor_PWM -= 10;
      if (Motor_PWM < 0) Motor_PWM = 0;
      lastCmdMs = millis();
      Serial.print("PWM_DOWN ");
      Serial.println(Motor_PWM);
      break;

    default:
      Serial.print("UNKNOWN_CMD ");
      Serial.println(cmd);
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

  Serial.println("READY TELEOP_DEBUG_ROS_TWIST");
  Serial.println("FORMAT: ENC LF RF LR RR");
  Serial.println("CMD: w/s/a/d/q/e/x/l/m or V vx vy wz");
}

// ===== 메인 루프 =====
void loop() {
  unsigned long now = millis();

  static char lineBuf[80];
  static int lineIndex = 0;

  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (lineIndex > 0) {
        lineBuf[lineIndex] = '\0';

        if (lineBuf[0] == 'V') {
          handleVelocityLine(lineBuf);
        } else {
          handleCommand(lineBuf[0]);
        }

        lineIndex = 0;
      }
    } else {
      if (lineIndex < 79) {
        lineBuf[lineIndex++] = c;
      } else {
        lineIndex = 0;
        Serial.println("LINE_OVERFLOW");
      }
    }
  }

  if (now - lastCmdMs > CMD_TIMEOUT_MS) {
    stopMotors();
  }

  if (now - lastEncMs >= ENC_PERIOD_MS) {
    lastEncMs = now;
    publishEncoder();
  }
}
