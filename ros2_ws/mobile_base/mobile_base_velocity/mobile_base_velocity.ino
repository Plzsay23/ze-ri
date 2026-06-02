#include <Encoder.h>

// ============================================================
// Ze-Ri Mobile Base Arduino Firmware
// Protocol:
//   V vx vy wz\n     vx, vy, wz in [-1.0, 1.0]
//   P pwm\n          set max PWM
//   A accel decel\n  set ramp step
//   X\n              stop
//
// Legacy keys:
//   w/s : forward/backward
//   a/d : turn left/right
//   q/e : strafe left/right
//   r/t/f/g : diagonal
//   x : stop
//   l/m : PWM +10 / -10
//
// Encoder output:
//   ENC LF RF LR RR
// ============================================================


// ===== Motor pins: existing confirmed wiring =====
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


// ===== Encoder pins =====
// Actual motor mapping:
//   A = RF
//   B = LF, inverted
//   C = RR
//   D = LR, inverted
Encoder encA(18, 31);   // RF
Encoder encB(19, 38);   // LF, inverted
Encoder encC(3, 49);    // RR
Encoder encD(2, A1);    // LR, inverted


// ===== Speed / timing =====
int Motor_PWM = 60;

// 너무 낮으면 모터가 웅웅거리기만 하고 못 움직임.
// 너무 높으면 시작이 튐.
const int MIN_MOTION_PWM = 18;

const unsigned long ENC_PERIOD_MS = 100;
const unsigned long CMD_TIMEOUT_MS = 800;
const unsigned long RAMP_PERIOD_MS = 20;

unsigned long lastEncMs = 0;
unsigned long lastCmdMs = 0;
unsigned long lastRampMs = 0;


// ===== Smooth ramp =====
// 20ms마다 wheel command가 이 정도씩 변함.
// 더 부드럽게: ACCEL_STEP/DECEL_STEP 낮추기
// 더 빠르게: ACCEL_STEP/DECEL_STEP 올리기
float ACCEL_STEP = 0.025f;
float DECEL_STEP = 0.040f;

// Wheel order: LF RF LR RR
float targetLF = 0.0f;
float targetRF = 0.0f;
float targetLR = 0.0f;
float targetRR = 0.0f;

float currentLF = 0.0f;
float currentRF = 0.0f;
float currentLR = 0.0f;
float currentRR = 0.0f;


// ===== Serial line buffer =====
char lineBuf[96];
uint8_t lineLen = 0;
bool lineMode = false;


// ===== Motor low-level macros =====
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


// ============================================================
// Utility
// ============================================================

float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

bool oppositeSign(float a, float b, float eps) {
  return fabs(a) > eps && fabs(b) > eps && (a * b) < 0.0f;
}

int scaledPwm(float v) {
  float a = fabs(v);

  if (a < 0.001f) {
    return 0;
  }

  int pwm = (int)(a * Motor_PWM + 0.5f);

  if (pwm > 0 && pwm < MIN_MOTION_PWM) {
    pwm = MIN_MOTION_PWM;
  }

  if (pwm > 255) {
    pwm = 255;
  }

  return pwm;
}


// ============================================================
// Motor signed output
// ============================================================

void motorA_signed(float v) { // RF
  int pwm = scaledPwm(v);

  if (pwm == 0) {
    MOTORA_STOP();
  } else if (v > 0.0f) {
    MOTORA_FORWARD(pwm);
  } else {
    MOTORA_BACKOFF(pwm);
  }
}

void motorB_signed(float v) { // LF
  int pwm = scaledPwm(v);

  if (pwm == 0) {
    MOTORB_STOP();
  } else if (v > 0.0f) {
    MOTORB_FORWARD(pwm);
  } else {
    MOTORB_BACKOFF(pwm);
  }
}

void motorC_signed(float v) { // RR
  int pwm = scaledPwm(v);

  if (pwm == 0) {
    MOTORC_STOP();
  } else if (v > 0.0f) {
    MOTORC_FORWARD(pwm);
  } else {
    MOTORC_BACKOFF(pwm);
  }
}

void motorD_signed(float v) { // LR
  int pwm = scaledPwm(v);

  if (pwm == 0) {
    MOTORD_STOP();
  } else if (v > 0.0f) {
    MOTORD_FORWARD(pwm);
  } else {
    MOTORD_BACKOFF(pwm);
  }
}


// ============================================================
// Wheel ramp
// ============================================================

float slewWheel(float current, float target) {
  target = clampf(target, -1.0f, 1.0f);

  // 방향이 바로 반대로 바뀌면 급반전하지 말고 먼저 0으로 감속
  if (oppositeSign(current, target, 0.02f)) {
    target = 0.0f;
  }

  float diff = target - current;

  if (fabs(diff) < 0.0005f) {
    return target;
  }

  float step;

  // 목표 절댓값이 작아지는 중이면 감속 step 사용
  if (fabs(target) < fabs(current)) {
    step = DECEL_STEP;
  } else {
    step = ACCEL_STEP;
  }

  if (fabs(diff) <= step) {
    return target;
  }

  if (diff > 0.0f) {
    return current + step;
  }

  return current - step;
}

void setWheelTargets(float lf, float rf, float lr, float rr) {
  targetLF = clampf(lf, -1.0f, 1.0f);
  targetRF = clampf(rf, -1.0f, 1.0f);
  targetLR = clampf(lr, -1.0f, 1.0f);
  targetRR = clampf(rr, -1.0f, 1.0f);
}

void applyCurrentWheels() {
  // Actual motor mapping:
  //   A = RF
  //   B = LF
  //   C = RR
  //   D = LR
  motorB_signed(currentLF);
  motorA_signed(currentRF);
  motorD_signed(currentLR);
  motorC_signed(currentRR);
}

void updateMotorRamp() {
  unsigned long now = millis();

  if (now - lastRampMs < RAMP_PERIOD_MS) {
    return;
  }

  lastRampMs = now;

  currentLF = slewWheel(currentLF, targetLF);
  currentRF = slewWheel(currentRF, targetRF);
  currentLR = slewWheel(currentLR, targetLR);
  currentRR = slewWheel(currentRR, targetRR);

  applyCurrentWheels();
}

void immediateStopMotors() {
  targetLF = 0.0f;
  targetRF = 0.0f;
  targetLR = 0.0f;
  targetRR = 0.0f;

  currentLF = 0.0f;
  currentRF = 0.0f;
  currentLR = 0.0f;
  currentRR = 0.0f;

  MOTORA_STOP();
  MOTORB_STOP();
  MOTORC_STOP();
  MOTORD_STOP();
}

void smoothStopMotors() {
  setWheelTargets(0.0f, 0.0f, 0.0f, 0.0f);
}


// ============================================================
// Mecanum velocity command
// vx: forward +
// vy: left +
// wz: CCW / left turn +
// ============================================================

void driveVelocity(float vx, float vy, float wz) {
  vx = clampf(vx, -1.0f, 1.0f);
  vy = clampf(vy, -1.0f, 1.0f);
  wz = clampf(wz, -1.0f, 1.0f);

  // Wheel order: LF RF LR RR
  float lf = vx + vy + wz;
  float rf = vx - vy - wz;
  float lr = vx - vy + wz;
  float rr = vx + vy - wz;

  float maxAbs = fabs(lf);

  if (fabs(rf) > maxAbs) maxAbs = fabs(rf);
  if (fabs(lr) > maxAbs) maxAbs = fabs(lr);
  if (fabs(rr) > maxAbs) maxAbs = fabs(rr);

  if (maxAbs > 1.0f) {
    lf /= maxAbs;
    rf /= maxAbs;
    lr /= maxAbs;
    rr /= maxAbs;
  }

  setWheelTargets(lf, rf, lr, rr);
  lastCmdMs = millis();
}


// ============================================================
// Legacy key command
// 이제 키 입력도 직접 모터를 때리지 않고 velocity target만 설정함.
// 그래서 한 번 누르면 CMD_TIMEOUT_MS 동안 유지되고 ramp가 걸림.
// ============================================================

void handleKeyCommand(char cmd) {
  if (cmd == '\n' || cmd == '\r') {
    return;
  }

  switch (cmd) {
    case 'w':
    case 'W':
      driveVelocity(1.0f, 0.0f, 0.0f);
      break;

    case 's':
    case 'S':
      driveVelocity(-1.0f, 0.0f, 0.0f);
      break;

    case 'a':
    case 'A':
      driveVelocity(0.0f, 0.0f, 1.0f);
      break;

    case 'd':
    case 'D':
      driveVelocity(0.0f, 0.0f, -1.0f);
      break;

    case 'q':
    case 'Q':
      driveVelocity(0.0f, 1.0f, 0.0f);
      break;

    case 'e':
    case 'E':
      driveVelocity(0.0f, -1.0f, 0.0f);
      break;

    case 'r':
    case 'R':
      driveVelocity(0.7f, 0.7f, 0.0f);
      break;

    case 't':
    case 'T':
      driveVelocity(0.7f, -0.7f, 0.0f);
      break;

    case 'f':
    case 'F':
      driveVelocity(-0.7f, 0.7f, 0.0f);
      break;

    case 'g':
    case 'G':
      driveVelocity(-0.7f, -0.7f, 0.0f);
      break;

    case 'x':
    case 'X':
      smoothStopMotors();
      lastCmdMs = millis();
      break;

    case 'z':
    case 'Z':
      immediateStopMotors();
      lastCmdMs = millis();
      break;

    case 'l':
    case 'L':
      Motor_PWM += 10;
      if (Motor_PWM > 255) Motor_PWM = 255;
      Serial.print("PWM ");
      Serial.println(Motor_PWM);
      break;

    case 'm':
    case 'M':
      Motor_PWM -= 10;
      if (Motor_PWM < 0) Motor_PWM = 0;
      Serial.print("PWM ");
      Serial.println(Motor_PWM);
      break;
  }
}


// ============================================================
// Encoder publish
// Output order must match base node:
//   ENC LF RF LR RR
// ============================================================

void publishEncoder() {
  long rawA = encA.read();     // RF
  long rawB = -encB.read();    // LF inverted
  long rawC = encC.read();     // RR
  long rawD = -encD.read();    // LR inverted

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


// ============================================================
// Line protocol
// ============================================================

void handleLine(char *line) {
  char *cmd = strtok(line, " ,\t");

  if (cmd == NULL) {
    return;
  }

  // V vx vy wz
  if (cmd[0] == 'V' || cmd[0] == 'v') {
    char *sx = strtok(NULL, " ,\t");
    char *sy = strtok(NULL, " ,\t");
    char *sw = strtok(NULL, " ,\t");

    if (sx == NULL || sy == NULL || sw == NULL) {
      smoothStopMotors();
      return;
    }

    float vx = atof(sx);
    float vy = atof(sy);
    float wz = atof(sw);

    driveVelocity(vx, vy, wz);
    return;
  }

  // P pwm
  if (cmd[0] == 'P' || cmd[0] == 'p') {
    char *spwm = strtok(NULL, " ,\t");

    if (spwm != NULL) {
      Motor_PWM = atoi(spwm);

      if (Motor_PWM < 0) Motor_PWM = 0;
      if (Motor_PWM > 255) Motor_PWM = 255;

      Serial.print("PWM ");
      Serial.println(Motor_PWM);
    }

    return;
  }

  // A accel decel
  if (cmd[0] == 'A' || cmd[0] == 'a') {
    char *sa = strtok(NULL, " ,\t");
    char *sd = strtok(NULL, " ,\t");

    if (sa != NULL) {
      ACCEL_STEP = atof(sa);
      ACCEL_STEP = clampf(ACCEL_STEP, 0.001f, 0.200f);
    }

    if (sd != NULL) {
      DECEL_STEP = atof(sd);
      DECEL_STEP = clampf(DECEL_STEP, 0.001f, 0.300f);
    }

    Serial.print("RAMP ");
    Serial.print(ACCEL_STEP, 4);
    Serial.print(" ");
    Serial.println(DECEL_STEP, 4);
    return;
  }

  // X or Z as line command
  if (cmd[0] == 'X' || cmd[0] == 'x') {
    smoothStopMotors();
    lastCmdMs = millis();
    return;
  }

  if (cmd[0] == 'Z' || cmd[0] == 'z') {
    immediateStopMotors();
    lastCmdMs = millis();
    return;
  }

  // Single-key fallback
  if (cmd[1] == '\0') {
    handleKeyCommand(cmd[0]);
  }
}

void readSerialCommands() {
  while (Serial.available()) {
    char c = Serial.read();

    if (lineMode) {
      if (c == '\n' || c == '\r') {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
        lineLen = 0;
        lineMode = false;
      } else if (lineLen < sizeof(lineBuf) - 1) {
        lineBuf[lineLen++] = c;
      } else {
        lineLen = 0;
        lineMode = false;
        immediateStopMotors();
      }

      continue;
    }

    // Line-based commands
    if (c == 'V' || c == 'v' ||
        c == 'P' || c == 'p' ||
        c == 'A') {
      lineMode = true;
      lineLen = 0;
      lineBuf[lineLen++] = c;
      continue;
    }

    // Single-char key commands
    handleKeyCommand(c);
  }
}


// ============================================================
// Setup / loop
// ============================================================

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

  immediateStopMotors();

  lastCmdMs = millis();
  lastEncMs = millis();
  lastRampMs = millis();

  Serial.println("READY ZE_RI_BASE_VELOCITY_SMOOTH");
  Serial.println("CMD: V vx vy wz  where vx,vy,wz in [-1,1]");
  Serial.println("CMD: P pwm");
  Serial.println("CMD: A accel decel");
  Serial.println("KEYS: w/s/a/d/q/e/r/t/f/g/x/z/l/m");
  Serial.println("FORMAT: ENC LF RF LR RR");
}

void loop() {
  unsigned long now = millis();

  readSerialCommands();
  updateMotorRamp();

  // 명령이 끊기면 안전상 즉시 정지
  if (now - lastCmdMs > CMD_TIMEOUT_MS) {
    immediateStopMotors();
    lastCmdMs = now;
  }

  if (now - lastEncMs >= ENC_PERIOD_MS) {
    lastEncMs = now;
    publishEncoder();
  }
}